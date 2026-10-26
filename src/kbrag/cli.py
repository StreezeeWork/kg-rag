"""Командная строка kb-rag: ``kbrag <команда>`` или ``python -m kbrag <команда>``.

Команды:

    validate-data   проверить корпус и эталонный набор, показать статистику
    build-gold      собрать data/eval/gold.jsonl из частей data/eval/parts/*.jsonl
    ask             задать вопрос из терминала
    eval            оценить один конфиг → reports/<name>/ (+ quality gate --fail-under)
    grid            прогнать сетку экспериментов → reports/grid/
    serve           запустить HTTP-сервис с веб-страницей

Коды возврата (на них опираются CI и Makefile):

    0 — успех;
    1 — ошибка данных или конфига: битый эталон, нет файла, неизвестная метрика…;
    2 — не пройден quality gate ``eval --fail-under``. Тот же код argparse возвращает
        при неверных аргументах командной строки — в обоих случаях «дальше не идём».

Тяжёлые модули (пайплайн со sklearn и моделями, FastAPI, uvicorn) импортируются внутри
команд: ``kbrag --help`` и ``kbrag validate-data`` не должны ждать загрузки того, что им не нужно.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from kbrag import __version__
from kbrag.config import AppConfig, load_config
from kbrag.eval.dataset import (
    QUESTION_TYPES,
    SPLITS,
    GoldItem,
    assign_splits,
    load_gold,
    save_gold,
    validate_gold,
)
from kbrag.eval.report import format_number, safe_file_name, write_report
from kbrag.eval.results import ExperimentResult, metric_names
from kbrag.generation.llm import LLMError
from kbrag.ingest import chunk_corpus, load_corpus, make_chunker

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_QUALITY_GATE = 2
EXIT_INTERRUPTED = 130  # общепринятый код «прервано по Ctrl+C» (128 + номер сигнала SIGINT)

DEFAULT_PARTS_DIR = "data/eval/parts"
DEFAULT_GOLD_OUT = "data/eval/gold.jsonl"
DEFAULT_EXPERIMENTS = "configs/experiments.yaml"
DEFAULT_GRID_OUT = "reports/grid"

# Сплит, по которому проверяется --fail-under: итоговое качество меряем только на test.
GATE_SPLIT = "test"
# Метрики, которые печатает `kbrag eval` (те же столбцы, что в leaderboard.md).
SUMMARY_METRICS = (
    "recall@5",
    "mrr",
    "ndcg@5",
    "hit@1",
    "fact_recall",
    "citation_precision",
    "abstain_f1",
    "latency_p50_ms",
    "latency_p95_ms",
)
# Для этих метрик «меньше — лучше», и порог «не ниже» для них бессмыслен: --fail-under
# latency_p95_ms=500 читался бы как «упасть, если быстрее 500 мс». Отклоняем явно.
LOWER_IS_BETTER = frozenset(
    {"latency_p50_ms", "latency_p95_ms", "false_abstention_rate", "invalid_citation_rate"}
)


# ---------------------------------------------------------------------------
# Общие помощники
# ---------------------------------------------------------------------------


def _print_error(message: str) -> None:
    # Ошибки — в stderr: так `kbrag ask --json > answer.json` не смешает их с результатом.
    print(f"Ошибка: {message}", file=sys.stderr)


def _load_cfg(path: str | None, needs_index: bool = False) -> AppConfig:
    """Явный ``--config`` обязан существовать; без него — как у сервиса (``$KBRAG_CONFIG`` или default.yaml).

    ``needs_index=True`` (ask, eval, serve): сразу проверяем, что для конфига установлены
    нейросетевые пакеты, — а не после минуты сборки индекса.
    """
    if path:
        if not Path(path).is_file():
            raise FileNotFoundError(f"конфиг не найден: {path}")
        cfg = load_config(path)
    else:
        # Правило выбора конфига по умолчанию живёт в одном месте — у сервиса.
        from kbrag.api.app import config_from_env

        cfg = config_from_env()
    if needs_index:
        from kbrag.api.app import require_installed_extras

        require_installed_extras(cfg)
    return cfg


def _config_stem(path: str | None) -> str:
    """Имя прогона по умолчанию — имя файла конфига, которым реально воспользовались.

    Без ``--config`` это ``$KBRAG_CONFIG`` или configs/default.yaml — как у сервиса.
    """
    if path is None:
        from kbrag.api.app import config_path_from_env

        path = config_path_from_env()
    return Path(path).stem if path else "default"


def _use_utf8_output() -> None:
    """Переключает stdout/stderr на UTF-8, если они в другой кодировке.

    На Windows, когда вывод идёт не в консоль (pipe в Git Bash, ``> answer.json``, IDE, CI),
    Python пишет в кодировке локали — cp1251 или cp1252. В них нет «→», «≥», а в cp1252 и всей
    кириллицы, и ``kbrag ask --json > answer.json`` падал с UnicodeEncodeError, оставляя пустой файл.
    В настоящей консоли кодировка уже UTF-8, и тогда ничего не меняем.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
        if hasattr(stream, "reconfigure") and encoding != "utf8":
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _format_counter(counter: Counter) -> str:
    """``Counter({"HR": 3, "IT": 1})`` → ``"HR 3, IT 1"`` (сначала самые частые)."""
    items = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return ", ".join(f"{key} {count}" for key, count in items) or "—"


def _print_gold_stats(gold: Sequence[GoldItem]) -> None:
    """Таблица «тип вопроса × split»: сразу видно, хватает ли dev-вопросов каждого типа."""
    by_type_and_split = Counter((item.qtype, item.split) for item in gold)
    by_type = Counter(item.qtype for item in gold)
    by_split = Counter(item.split for item in gold)
    # Сначала известные типы в каноническом порядке, затем неизвестные (их потом отметит validate_gold).
    types = [t for t in QUESTION_TYPES if t in by_type] + sorted(set(by_type) - set(QUESTION_TYPES))

    print(f"Эталонный набор: {len(gold)} вопросов")
    print(f"  {'тип':<14}" + "".join(f"{split:>7}" for split in SPLITS) + f"{'всего':>8}")
    for qtype in types:
        cells = "".join(f"{by_type_and_split[(qtype, split)]:>7}" for split in SPLITS)
        print(f"  {qtype:<14}{cells}{by_type[qtype]:>8}")
    totals = "".join(f"{by_split[split]:>7}" for split in SPLITS)
    print(f"  {'итого':<14}{totals}{len(gold):>8}")


def _print_errors(title: str, errors: Sequence[str]) -> None:
    print(f"{title} ({len(errors)}):", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)


# ---------------------------------------------------------------------------
# validate-data
# ---------------------------------------------------------------------------


def cmd_validate_data(args: argparse.Namespace) -> int:
    """Читает корпус и эталон, печатает статистику; ошибки разметки → код 1."""
    cfg = _load_cfg(args.config)
    docs = load_corpus(cfg.corpus_dir)
    if not docs:
        _print_error(f"в каталоге {cfg.corpus_dir} нет ни одного документа *.md")
        return EXIT_ERROR
    chunks = chunk_corpus(docs, make_chunker(cfg.chunking))

    departments = Counter(str(doc.metadata.get("department", "—")) for doc in docs)
    statuses = Counter(str(doc.metadata.get("status", "—")) for doc in docs)
    print(f"Корпус: {cfg.corpus_dir}")
    print(f"  документов: {len(docs)}")
    print(f"  по отделам: {_format_counter(departments)}")
    print(f"  по статусам: {_format_counter(statuses)}")
    print(f"  средняя длина документа: {_mean([len(doc.text.split()) for doc in docs]):.0f} слов")
    print(
        f"  фрагментов: {len(chunks)} (нарезка {cfg.chunking.strategy}, chunk_size={cfg.chunking.chunk_size}), "
        f"в среднем {_mean([len(chunk.text.split()) for chunk in chunks]):.0f} слов"
    )

    gold_path = Path(cfg.gold_path)
    if not gold_path.is_file():
        _print_error(f"эталонный набор {gold_path} не найден — соберите его командой `kbrag build-gold`")
        return EXIT_ERROR
    gold = load_gold(gold_path)
    print()
    _print_gold_stats(gold)

    errors = validate_gold(gold, docs)
    if errors:
        _print_errors(f"Ошибки в эталонном наборе {gold_path}", errors)
        return EXIT_ERROR
    print("\nКорпус и эталонный набор в порядке.")
    return EXIT_OK


# ---------------------------------------------------------------------------
# build-gold
# ---------------------------------------------------------------------------


def cmd_build_gold(args: argparse.Namespace) -> int:
    """Склеивает части эталона, назначает dev/test, проверяет и пишет (или сверяет с --check)."""
    parts_dir = Path(args.parts)
    # sorted: порядок файлов на диске зависит от ОС, а итоговый файл должен быть одинаковым везде.
    part_files = sorted(parts_dir.glob("*.jsonl"))
    if not part_files:
        _print_error(f"в каталоге {parts_dir} нет файлов *.jsonl")
        return EXIT_ERROR
    items = [item for path in part_files for item in load_gold(path)]
    items = assign_splits(items, dev_fraction=args.dev_fraction)

    cfg = _load_cfg(args.config)
    errors = validate_gold(items, load_corpus(cfg.corpus_dir))
    if errors:
        _print_errors("Ошибки в частях эталонного набора", errors)
        return EXIT_ERROR

    out = Path(args.out)
    if args.check:
        return _check_gold_file(out, items)
    save_gold(items, out)
    print(f"Собрано {len(items)} вопросов из {len(part_files)} файлов → {out}")
    _print_gold_stats(items)
    return EXIT_OK


def _check_gold_file(path: Path, expected: list[GoldItem]) -> int:
    """Совпадает ли файл на диске с тем, что собрал бы build-gold. Нужен в CI.

    Сравниваем разобранные записи, а не байты: git на Windows (autocrlf) может поменять
    переводы строк в файле, и побайтовое сравнение ложно сообщало бы об устаревшем файле.
    """
    hint = "запустите `kbrag build-gold` и закоммитьте результат"
    if not path.is_file():
        _print_error(f"файл {path} не найден — {hint}")
        return EXIT_ERROR
    try:
        on_disk = load_gold(path)
    except ValueError as exc:  # битый JSON, нет обязательных полей, не UTF-8 (тоже ValueError)
        _print_error(f"файл {path} повреждён ({exc}) — {hint}")
        return EXIT_ERROR
    if on_disk == expected:
        print(f"OK: {path} совпадает с тем, что собирается из частей ({len(expected)} вопросов)")
        return EXIT_OK

    expected_by_id = {item.qid: item for item in expected}
    disk_by_id = {item.qid: item for item in on_disk}
    details = []
    if added := sorted(expected_by_id.keys() - disk_by_id.keys()):
        details.append(f"нет в файле: {', '.join(added[:5])}")
    if removed := sorted(disk_by_id.keys() - expected_by_id.keys()):
        details.append(f"лишние в файле: {', '.join(removed[:5])}")
    changed = sorted(
        q for q in expected_by_id.keys() & disk_by_id.keys() if expected_by_id[q] != disk_by_id[q]
    )
    if changed:
        details.append(f"изменились: {', '.join(changed[:5])}")
    if not details:
        details.append("отличается порядок вопросов")
    _print_error(f"файл {path} устарел ({'; '.join(details)}) — {hint}")
    return EXIT_ERROR


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


def cmd_ask(args: argparse.Namespace) -> int:
    """Отвечает на один вопрос. ``--json`` печатает ровно тот же формат, что ``POST /ask``."""
    question = args.question.strip()
    if not question:
        _print_error("пустой вопрос")
        return EXIT_ERROR
    cfg = _load_cfg(args.config, needs_index=True)

    from kbrag.api.app import build_ask_response
    from kbrag.pipeline import RAGPipeline

    pipeline = RAGPipeline.from_config(cfg)
    answer = pipeline.answer(question, top_k=args.top_k)
    payload = build_ask_response(answer, pipeline.score_source).model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_answer(payload)
    return EXIT_OK


def _print_answer(payload: dict[str, Any]) -> None:
    print(payload["answer"])
    print()
    if payload["abstained"]:
        reason = "низкая уверенность поиска" if "low_confidence" in payload["flags"] else "решение генератора"
        print(f"Отказ: да ({reason})")
    confidence = payload["confidence"]
    total_ms = payload["timings_ms"].get("total_ms", 0.0)
    print(f"Уверенность: {'—' if confidence is None else f'{confidence:.3f}'}; время: {total_ms:.0f} мс")
    # Сверяем по chunk_id, а не по номеру: при guardrails.action=drop_chunk генератор видит
    # не все фрагменты, и номер [n] в ответе может не совпасть с позицией фрагмента в выдаче.
    cited = {citation["chunk_id"] for citation in payload["citations"]}
    print("Источники:" if payload["contexts"] else "Источники: поиск ничего не нашёл")
    for ctx in payload["contexts"]:
        place = " > ".join(part for part in (ctx["title"], ctx["section"]) if part)
        mark = "  <- в ответе" if ctx["chunk_id"] in cited else ""
        # В гибриде score — ранговый RRF (≈0.016 у всех), он ничего не говорит о релевантности;
        # показываем score того компонента, по которому считается уверенность.
        value = ctx["confidence_score"] if ctx.get("confidence_score") is not None else ctx["score"]
        score = "—" if value is None else f"{value:.3f}"
        print(f"  [{ctx['index']}] {place} ({ctx['chunk_id']}, score {score}){mark}")
    if payload["flags"]:
        print(f"Флаги: {', '.join(payload['flags'])}")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    """Оценка одного конфига: отчёт в reports/<name>/, опционально порог и quality gate."""
    from kbrag.eval.runner import run_experiment

    cfg = _load_cfg(args.config, needs_index=True)
    gates: list[tuple[str, float]] = args.fail_under or []
    # Всё, что можно проверить ДО прогона, проверяем до него: узнать об опечатке в метрике,
    # о невозможности сохранить порог или о неверном --out через 10 минут оценки обидно.
    if args.save_calibration and (reason := _calibration_not_saveable(cfg)):
        _print_error(f"--save-calibration: {reason}")
        return EXIT_ERROR
    known = metric_names(cfg.eval.k_values)
    unknown = [name for name, _ in gates if name not in known]
    if unknown:
        _print_error(
            f"неизвестные метрики в --fail-under: {', '.join(unknown)}; доступны: {', '.join(known)}"
        )
        return EXIT_ERROR
    inverted = [name for name, _ in gates if name in LOWER_IS_BETTER]
    if inverted:
        _print_error(
            f"для метрик {', '.join(inverted)} «меньше — лучше», порог --fail-under к ним неприменим"
        )
        return EXIT_ERROR
    if not Path(cfg.gold_path).is_file():
        _print_error(f"эталонный набор {cfg.gold_path} не найден — соберите его командой `kbrag build-gold`")
        return EXIT_ERROR

    name = args.name or _config_stem(args.config)
    out_dir = Path(args.out or cfg.reports_dir) / safe_file_name(name)
    out_dir.mkdir(parents=True, exist_ok=True)  # неверный путь — ошибка сразу, а не после прогона

    result = run_experiment(cfg, name=name)
    leaderboard = write_report([result], out_dir, split=cfg.eval.report_split)
    _print_eval_summary(result)
    print(f"Отчёт: {leaderboard}")

    code = _check_quality_gates(result, gates)
    if args.save_calibration:
        # Порог из прогона, не прошедшего quality gate, в сервис не отдаём.
        if code != EXIT_OK:
            print("Порог отказа не сохранён: quality gate не пройден.", file=sys.stderr)
            return code
        if result.threshold is None:
            _print_error(
                "порог отказа не подобран на dev (нет неотвечаемых dev-вопросов или ни один порог "
                "не лучше случайного) — файл калибровки не изменён"
            )
            return EXIT_ERROR
        path = _save_calibration(cfg, result.name, result.threshold)
        print(f"Порог отказа сохранён: {path}")
    return code


def _print_eval_summary(result: ExperimentResult) -> None:
    threshold = "не задан" if result.threshold is None else format_number(result.threshold)
    print(f"Эксперимент: {result.name}")
    print(f"Порог отказа: {threshold}")
    n_questions = format_number(result.metric("n_questions", GATE_SPLIT))
    print(f"Метрики на {GATE_SPLIT} ({n_questions} вопросов):")
    for metric in SUMMARY_METRICS:
        print(f"  {metric:<20}{format_number(result.metric(metric, GATE_SPLIT)):>10}")


def _calibration_not_saveable(cfg: AppConfig) -> str | None:
    """Почему порог этого конфига нельзя сохранить для сервиса (None — можно)."""
    abstention = cfg.abstention
    if not abstention.enabled:
        return "отказ по порогу выключен (abstention.enabled: false) — сохранять нечего"
    if abstention.threshold != "auto":
        return (
            f"порог не подбирается: abstention.threshold={abstention.threshold!r} задан в конфиге, "
            "сервис возьмёт его оттуда (подбор на dev — только при threshold: auto)"
        )
    if not abstention.calibration_path:
        return "в конфиге не задан abstention.calibration_path — некуда сохранить порог"
    return None


def _save_calibration(cfg: AppConfig, experiment: str, threshold: float) -> Path:
    """Пишет порог для сервиса в ``abstention.calibration_path``.

    Формат файла задаёт ``pipeline.save_calibration`` — тот же модуль, что его читает
    (``pipeline.load_calibration``), поэтому писатель и читатель не разойдутся.
    ``experiment`` попадает в файл как подсказка человеку: откуда и когда взялся порог.
    """
    from kbrag.pipeline import save_calibration

    return save_calibration(cfg, threshold, experiment=experiment)


def _check_quality_gates(result: ExperimentResult, gates: Sequence[tuple[str, float]]) -> int:
    """Quality gate: каждая метрика на test должна быть не ниже порога, иначе код 2.

    Неопределённая метрика (None, например abstain_f1 без единого отказа) считается
    непройденной: «не смогли посчитать» не должно молча превращаться в «всё хорошо».
    """
    if not gates:
        return EXIT_OK
    print(f"Quality gate (split={GATE_SPLIT}):")
    failed = []
    for metric, minimum in gates:
        value = result.metric(metric, GATE_SPLIT)
        passed = value is not None and value >= minimum
        print(f"  {metric} = {format_number(value)}, нужно >= {minimum:g}: {'OK' if passed else 'FAIL'}")
        if not passed:
            failed.append(metric)
    if failed:
        _print_error(f"качество ниже порога: {', '.join(failed)}")
        return EXIT_QUALITY_GATE
    return EXIT_OK


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------


def cmd_grid(args: argparse.Namespace) -> int:
    """Прогоняет сетку экспериментов и пишет общий leaderboard."""
    from kbrag.eval.runner import run_grid

    results = run_grid(args.experiments, only=args.only, include_heavy=args.include_heavy)
    done = [r for r in results if not r.is_skipped]
    # Сплит отчёта — из конфига экспериментов (у всех он общий, приходит из base).
    split = done[0].config.get("eval", {}).get("report_split", GATE_SPLIT) if done else GATE_SPLIT
    leaderboard = write_report(results, args.out, split=split)

    print(f"Экспериментов: {len(results)}, выполнено {len(done)}, пропущено {len(results) - len(done)}")
    width = max((len(r.name) for r in results), default=0)
    for r in results:
        if r.is_skipped:
            print(f"  {r.name:<{width}}  пропущен: {r.skipped}")
        else:
            print(f"  {r.name:<{width}}  recall@5={format_number(r.metric('recall@5', split))}")
    print(f"Отчёт: {leaderboard}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    """Запускает uvicorn с приложением из ``kbrag.api``; индекс строится при старте."""
    import uvicorn

    from kbrag.api.app import create_app

    cfg = _load_cfg(args.config, needs_index=True)
    app = create_app(cfg=cfg)
    print(
        f"kb-rag: индекс строится при старте, затем сервис будет доступен на http://{args.host}:{args.port}"
    )
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    except SystemExit as exc:
        # Если старт упал (нет корпуса, битый конфиг), uvicorn сам вызывает sys.exit(3). Переводим
        # это в наш документированный код 1, чтобы скрипты и CI видели обычную «ошибку данных/конфига».
        return EXIT_ERROR if exc.code else EXIT_OK
    return EXIT_OK


# ---------------------------------------------------------------------------
# Разбор аргументов
# ---------------------------------------------------------------------------


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"ожидается целое число, получено {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"ожидается число >= 1, получено {value}")
    return value


def _fraction(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"ожидается число, получено {text!r}") from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"ожидается число от 0 до 1, получено {value}")
    return value


def _port(text: str) -> int:
    value = _positive_int(text)
    if value > 65535:
        raise argparse.ArgumentTypeError(f"номер порта должен быть от 1 до 65535, получено {value}")
    return value


def _gate(text: str) -> tuple[str, float]:
    """``"recall@5=0.7"`` → ``("recall@5", 0.7)``."""
    name, sep, raw_value = text.partition("=")
    if not sep or not name.strip():
        raise argparse.ArgumentTypeError(
            f"ожидается метрика=значение (например recall@5=0.7), получено {text!r}"
        )
    try:
        value = float(raw_value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"порог должен быть числом: {text!r}") from None
    return name.strip(), value


def build_parser() -> argparse.ArgumentParser:
    """Описание всех команд и их аргументов (его же показывает ``kbrag --help``)."""
    parser = argparse.ArgumentParser(
        prog="kbrag",
        description="kb-rag — RAG-ассистент по корпоративной базе знаний с воспроизводимой оценкой качества.",
        epilog="Подробнее о команде: kbrag <команда> --help",
    )
    parser.add_argument("--version", action="version", version=f"kbrag {__version__}")

    # Общие аргументы подкоманд: так `-v` и `--config` пишутся после имени команды (kbrag eval -v).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v", "--verbose", action="count", default=0, help="подробные логи: -v — INFO, -vv — DEBUG"
    )
    with_config = argparse.ArgumentParser(add_help=False, parents=[common])
    with_config.add_argument(
        "--config", default=None, help="YAML-конфиг (по умолчанию $KBRAG_CONFIG или configs/default.yaml)"
    )

    commands = parser.add_subparsers(dest="command", metavar="<команда>", required=True)

    p = commands.add_parser(
        "validate-data", parents=[with_config], help="проверить корпус и эталонный набор, показать статистику"
    )
    p.set_defaults(handler=cmd_validate_data)

    p = commands.add_parser(
        "build-gold", parents=[with_config], help="собрать gold.jsonl из частей и назначить dev/test"
    )
    p.add_argument(
        "--parts",
        default=DEFAULT_PARTS_DIR,
        help=f"каталог с частями *.jsonl (по умолчанию {DEFAULT_PARTS_DIR})",
    )
    p.add_argument(
        "--out", default=DEFAULT_GOLD_OUT, help=f"куда записать набор (по умолчанию {DEFAULT_GOLD_OUT})"
    )
    p.add_argument("--dev-fraction", type=_fraction, default=0.3, help="доля dev-вопросов каждого типа (0.3)")
    p.add_argument(
        "--check", action="store_true", help="ничего не писать, только проверить, что файл актуален (для CI)"
    )
    p.set_defaults(handler=cmd_build_gold)

    p = commands.add_parser("ask", parents=[with_config], help="задать вопрос из терминала")
    p.add_argument("question", help="текст вопроса (в кавычках)")
    p.add_argument("--top-k", type=_positive_int, default=None, help="сколько фрагментов передать генератору")
    p.add_argument("--json", action="store_true", help="вывести ответ в JSON (тот же формат, что POST /ask)")
    p.set_defaults(handler=cmd_ask)

    p = commands.add_parser("eval", parents=[with_config], help="оценить конфиг на эталонном наборе")
    p.add_argument("--name", default=None, help="имя прогона (по умолчанию — имя файла конфига)")
    p.add_argument("--out", default=None, help="каталог отчётов (по умолчанию reports_dir из конфига)")
    p.add_argument(
        "--save-calibration",
        action="store_true",
        help="сохранить подобранный на dev порог отказа в abstention.calibration_path (его читает сервис)",
    )
    p.add_argument(
        "--fail-under",
        type=_gate,
        nargs="+",
        action="extend",
        metavar="МЕТРИКА=ЗНАЧЕНИЕ",
        help="quality gate по метрикам test: код 2, если хоть одна ниже порога (например recall@5=0.7)",
    )
    p.set_defaults(handler=cmd_eval)

    p = commands.add_parser("grid", parents=[common], help="прогнать сетку экспериментов")
    p.add_argument(
        "--experiments", default=DEFAULT_EXPERIMENTS, help=f"файл сетки (по умолчанию {DEFAULT_EXPERIMENTS})"
    )
    p.add_argument(
        "--only", nargs="+", action="extend", metavar="ИМЯ", help="запустить только эти эксперименты"
    )
    p.add_argument(
        "--include-heavy", action="store_true", help="запускать и тяжёлые эксперименты (heavy: true)"
    )
    p.add_argument(
        "--out", default=DEFAULT_GRID_OUT, help=f"каталог отчёта (по умолчанию {DEFAULT_GRID_OUT})"
    )
    p.set_defaults(handler=cmd_grid)

    p = commands.add_parser("serve", parents=[with_config], help="запустить HTTP-сервис с веб-страницей")
    p.add_argument(
        "--host", default="127.0.0.1", help="адрес (127.0.0.1 — только с этой машины; 0.0.0.0 — для всех)"
    )
    p.add_argument("--port", type=_port, default=8080, help="порт (по умолчанию 8080)")
    p.set_defaults(handler=cmd_serve)
    return parser


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    if verbosity <= 0:
        # Без -v прячем шум библиотек HuggingFace: «Could not reach the Hub» при офлайн-режиме
        # (дважды на каждую модель) и полосы прогресса «Loading weights». Именно переменные
        # окружения, а не logging.getLogger(...).setLevel: huggingface_hub при импорте сам
        # выставляет уровень своему логгеру и перетёр бы наш. Импорт моделей ленивый, поэтому
        # сюда мы успеваем раньше. setdefault: явно заданное пользователем значение важнее.
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа: ``kbrag ...`` (pyproject.toml) и ``python -m kbrag ...``. Возвращает код выхода."""
    _use_utf8_output()
    args = build_parser().parse_args(argv)
    # У сервиса строки INFO («порог отказа загружен», «сервис готов») — его основной журнал,
    # поэтому для serve уровень INFO и без -v.
    _setup_logging(max(args.verbose, 1) if args.command == "serve" else args.verbose)
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        print("Прервано.", file=sys.stderr)
        return EXIT_INTERRUPTED
    # Ожидаемые ошибки (нет файла, битые данные, недоступна LLM, нет нейросетевых пакетов) —
    # короткое сообщение без трассировки. Неожиданные (баги) пусть падают с полным traceback.
    except (OSError, ValueError, yaml.YAMLError, ImportError, LLMError) as exc:
        _print_error(str(exc))
        return EXIT_ERROR
