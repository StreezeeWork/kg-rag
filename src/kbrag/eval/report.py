"""Отчёт по экспериментам: машиночитаемый JSON + человекочитаемый leaderboard.md.

Что пишется в ``out_dir``:

* ``results.json`` — конфиги, метрики, пороги и статистика индекса всех экспериментов,
  отпечаток данных каждого прогона (``provenance``) и версии Python/пакетов (``environment``);
* ``per_question/<name>.jsonl`` — ответ и метрики по каждому вопросу (для разбора ошибок);
* ``leaderboard.md`` — две таблицы (поиск; ответы и отказы), 95% интервалы Recall@5 и
  отличие каждой строки от лидера (bootstrap), разбивка Recall@5 по типам вопросов,
  отдельная проверка типа confusable и легенда «как читать».

Почему таблиц две. Метрики поиска (Recall@5, MRR, …) от порога отказа не зависят. Метрики
ответа — зависят: отказ на отвечаемый вопрос даёт FactRecall = 0, а порог у каждого
эксперимента свой. Если положить их в одну строку без FAR, разница в FactRecall выглядит
как разница в качестве генерации, хотя на деле это разница в пороге.

Что воспроизводимо, а что нет. Отчёт не содержит даты, поэтому при повторном прогоне с тем же
конфигом **метрики качества, пороги и ответы** совпадают байт в байт. Тайминги — нет:
``latency_ms``, ``latency_p50_ms``, ``latency_p95_ms`` и ``index_stats.index_time_ms``
(список — ``results.TIMING_KEYS``) меряются одним проходом без прогрева и зависят от
нагрузки машины. При сравнении двух отчётов в git их строки надо игнорировать.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kbrag.eval.dataset import QUESTION_TYPES
from kbrag.eval.metrics import bootstrap_ci, paired_bootstrap_diff
from kbrag.eval.results import BY_TYPE_KEY, ExperimentResult

# (заголовок столбца, имя метрики) — порядок задаёт порядок столбцов.
# Поиск: от порога отказа не зависит.
RECALL_CI = "recall@5:ci95"  # псевдо-метрика: интервал считается в отчёте из per_question
SEARCH_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Recall@5", "recall@5"),
    ("Recall@5, 95% ДИ", RECALL_CI),
    ("MRR", "mrr"),
    ("nDCG@5", "ndcg@5"),
    ("Hit@1", "hit@1"),
    ("p50, мс", "latency_p50_ms"),
    ("p95, мс", "latency_p95_ms"),
)
# Ответы и отказы: зависят от порога, поэтому рядом всегда FAR и сам порог.
ANSWER_COLUMNS: tuple[tuple[str, str], ...] = (
    ("FactRecall (отвеч.)", "fact_recall_answered"),
    ("FactRecall", "fact_recall"),
    ("CitationPrecision", "citation_precision"),
    ("Архивные ссылки", "stale_citation_rate"),
    ("Abstention F1", "abstain_f1"),
    ("FAR", "false_abstention_rate"),
)
# Что сравниваем с лидером парным bootstrap (раздел «Отличие от лидера»).
DIFF_COLUMNS: tuple[tuple[str, str], ...] = (
    ("ΔRecall@5 [95% ДИ]", "recall@5"),
    ("ΔMRR [95% ДИ]", "mrr"),
)
SORT_METRIC = "recall@5"
CONFUSABLE_TYPE = "confusable"
# Пакеты, версии которых пишутся в results.json (блок environment).
ENVIRONMENT_PACKAGES = (
    "kbrag",
    "numpy",
    "scikit-learn",
    "snowballstemmer",
    "sentence-transformers",
    "transformers",
    "torch",
)
MISSING = "—"

_LEGEND = """## Как читать таблицы

Поиск:
- **Recall@5** — доля доказательств (цитат из эталона), найденных среди первых 5 фрагментов. Главная метрика поиска.
- **Recall@5, 95% ДИ** — доверительный интервал (bootstrap по вопросам, seed фиксирован): где могла бы оказаться метрика на другом наборе вопросов того же рода. Если интервалы строк сильно перекрываются, порядок строк — не вывод.
- **Отличие от лидера** — разница «эксперимент − первая строка» на тех же вопросах с 95% интервалом (парный bootstrap); «≈» — интервал включает 0, то есть по этим данным строки неотличимы.
- **MRR** — 1 / позиция первого релевантного фрагмента, усреднённое по вопросам. Считается по всей оцениваемой выдаче (max(k_values, top_k) фрагментов, по умолчанию 10), то есть фактически это MRR@10.
- **nDCG@5** — учитывает позиции всех найденных доказательств: чем выше, тем лучше.
- **Hit@1** — доля вопросов, где первый же фрагмент релевантен.
- **p50 / p95, мс** — медианная и «хвостовая» задержка ответа. Это один прогон на этой машине без прогрева: от запуска к запуску цифры заметно гуляют, поэтому годятся только для грубого сравнения строк одного отчёта.
- **N** — вопросов в сплите: отвечаемые + неотвечаемые. Метрики поиска считаются только по отвечаемым.

Ответы и отказы:
- **FactRecall (отвеч.)** — доля ключевых фактов эталона в ответе, только по вопросам, где система ответила. По этому столбцу сравнивают качество генерации.
- **FactRecall** — то же, но сквозное: отказ на отвечаемый вопрос считается за 0. Поэтому оно зависит от порога, а порог у каждого эксперимента свой: высокий FAR опускает FactRecall при хорошем качестве ответов.
- **CitationPrecision** — доля ссылок `[n]`, ведущих на фрагмент с доказательством. Считается по отвеченным отвечаемым вопросам (их доля = 1 − FAR); ответ без единой ссылки = 0.
- **Архивные ссылки** — доля отвеченных вопросов, где ответ ссылается на архивную редакцию документа, которой нет в эталоне (меньше — лучше).
- **Abstention F1** — насколько точно система отказывается отвечать на вопросы, ответа на которые в базе нет.
- **FAR** (false abstention rate) — доля отвечаемых вопросов, на которые система зря отказалась (меньше — лучше).
- **Порог** — шкала и значение порога уверенности для отказа; «—» — без порога. Подбирается на dev по balanced accuracy отдельно для каждого эксперимента. Шкалы разные: `bm25` — сырой BM25-score (растёт с длиной вопроса, порог по нему грубый), `dense` — косинус, `rerank` — score cross-encoder; числа разных шкал несравнимы.

Общее:
- Это описательное сравнение (абляция) на test. Боевые конфиги (`configs/offline.yaml`, `configs/default.yaml`) зафиксированы заранее, а не выбраны по этой таблице. Если выбрать конфиг по test-строкам, его test-метрики станут оптимистично смещёнными: выбирают по dev (dev-метрики есть в `results.json`), а test используют один раз — для подтверждения. Выборки маленькие, поэтому разница в 1–2 вопроса — шум.
- **—** — метрика не определена (например, не было ни одного отказа).
"""


def format_number(value: Any) -> str:
    """Число → строка с 3 знаками после точки; целые — как есть; None/NaN → «—»."""
    if hasattr(value, "item") and callable(value.item):  # скаляры numpy → числа Python
        value = value.item()
    if value is None or isinstance(value, bool):
        return MISSING if value is None else str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return MISSING if not math.isfinite(value) else f"{value:.3f}"
    return str(value)


def safe_file_name(name: str) -> str:
    """Имя эксперимента → безопасное имя файла (Windows запрещает ``: / \\ * ? " < > |``)."""
    cleaned = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("._")
    return cleaned or "experiment"


def _escape_cell(text: str) -> str:
    # Вертикальная черта внутри ячейки сломала бы markdown-таблицу.
    return text.replace("|", "\\|").replace("\n", " ")


def _to_jsonable(obj: Any) -> Any:
    """Приводит данные к тому, что точно сериализуется в JSON.

    * NaN/inf → None: стандарт JSON их не поддерживает, а многие парсеры падают;
    * числа numpy (``np.float32`` и т.п.) → обычные числа Python;
    * всё неизвестное → строка: падать в самом конце долгого прогона из-за отчёта обидно.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    if hasattr(obj, "item") and callable(obj.item):  # скаляры numpy
        return _to_jsonable(obj.item())
    if isinstance(obj, Path):
        return obj.as_posix()
    return str(obj)


def _dump_json(obj: Any, indent: int | None = None) -> str:
    return json.dumps(_to_jsonable(obj), ensure_ascii=False, indent=indent)


def _check_unique_names(results: Sequence[ExperimentResult]) -> None:
    seen: dict[str, str] = {}
    for result in results:
        file_name = safe_file_name(result.name)
        if file_name in seen:
            raise ValueError(
                f"эксперименты {seen[file_name]!r} и {result.name!r} дают одинаковое имя файла {file_name!r}"
            )
        seen[file_name] = result.name


# ---------------------------------------------------------------------------
# leaderboard.md
# ---------------------------------------------------------------------------


def _sort_key(result: ExperimentResult, split: str) -> tuple[float, str]:
    value = result.metric(SORT_METRIC, split)
    # Лучший Recall@5 — первым; эксперименты без значения — в конце; при равенстве — по имени.
    if value is None or not math.isfinite(value):
        return (math.inf, result.name)
    return (-value, result.name)


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _types_in_report(results: Sequence[ExperimentResult]) -> list[str]:
    """Типы вопросов, для которых хоть у одного эксперимента есть Recall@5.

    ``unanswerable`` обычно выпадает сам: у вопросов без доказательств recall не определён.
    """
    present: set[str] = set()
    for result in results:
        for qtype, metrics in result.metrics.get(BY_TYPE_KEY, {}).items():
            if metrics.get(SORT_METRIC) is not None:
                present.add(qtype)
    known = [qtype for qtype in QUESTION_TYPES if qtype in present]
    return known + sorted(present - set(QUESTION_TYPES))


def _counts(metrics: dict) -> tuple[int, int] | None:
    """(отвечаемых, неотвечаемых) из метрик сплита; None для старых отчётов без n_answerable."""
    n_questions, n_answerable = metrics.get("n_questions"), metrics.get("n_answerable")
    if isinstance(n_questions, int) and isinstance(n_answerable, int):
        return n_answerable, n_questions - n_answerable
    return None


def _format_n(metrics: dict) -> str:
    """«124 + 15»: отвечаемые (знаменатель метрик поиска) + неотвечаемые."""
    counts = _counts(metrics)
    return f"{counts[0]} + {counts[1]}" if counts else format_number(metrics.get("n_questions"))


def _format_threshold(result: ExperimentResult) -> str:
    """«dense: 0.499» — без шкалы порог 20.5 (BM25) рядом с 0.5 (косинус) сбивает с толку."""
    value = format_number(result.threshold)
    if result.threshold is None or not result.score_source:
        return value
    return f"{result.score_source}: {value}"


def _intro(done: Sequence[ExperimentResult], split: str) -> str:
    text = f"Метрики посчитаны на сплите `{split}`"
    counts = _counts(done[0].metrics.get(split, {})) if done else None
    if counts and counts[0]:
        answerable, unanswerable = counts
        text += (
            f": {answerable + unanswerable} вопросов, из них {answerable} отвечаемых (по ним считаются"
            f" метрики поиска) и {unanswerable} неотвечаемых. Один отвечаемый вопрос ≈ "
            f"{1 / answerable:.3f} Recall@5"
        )
    return text + ". Строки отсортированы по Recall@5 (по убыванию)."


def _threshold_note(done: Sequence[ExperimentResult], split: str) -> list[str]:
    """Напоминание, на какой маленькой выборке подобран порог (он общий для всего сплита dev)."""
    counts = _counts(done[0].metrics.get("dev", {})) if done and split != "dev" else None
    if not counts:
        return []
    answerable, unanswerable = counts
    return [
        f"Порог подбирается на dev: {answerable} отвечаемых и {unanswerable} неотвечаемых вопросов."
        " По такой выборке он шумный, поэтому сравнивайте FactRecall вместе с FAR.",
        "",
    ]


def _split_rows(result: ExperimentResult, split: str) -> list[dict]:
    """Строки per_question нужного сплита (``all`` — все)."""
    return [row for row in result.per_question if split == "all" or row.get("split") == split]


def _search_cell(result: ExperimentResult, metric: str, split: str) -> str:
    if metric == RECALL_CI:
        interval = bootstrap_ci([row.get(SORT_METRIC) for row in _split_rows(result, split)])
        return MISSING if interval is None else f"{interval[0]:.3f}–{interval[1]:.3f}"
    return format_number(result.metric(metric, split))


def _format_diff(leader: ExperimentResult, other: ExperimentResult, metric: str, split: str) -> str:
    """«-0.012 [-0.040; +0.016] ≈»: разница other − leader на общих вопросах и её интервал."""
    other_by_qid = {row.get("qid"): row.get(metric) for row in _split_rows(other, split)}
    leader_rows = _split_rows(leader, split)
    diff = paired_bootstrap_diff(
        [other_by_qid.get(row.get("qid")) for row in leader_rows],
        [row.get(metric) for row in leader_rows],
    )
    if diff is None:
        return MISSING
    mean, low, high = diff
    mark = " ≈" if low <= 0 <= high else ""
    return f"{mean:+.3f} [{low:+.3f}; {high:+.3f}]{mark}"


def _leader_diff_section(done: Sequence[ExperimentResult], split: str) -> list[str]:
    """Насколько каждый эксперимент отличается от первой строки — с поправкой на шум."""
    if len(done) < 2:
        return []
    leader = done[0]
    rows = [
        [_escape_cell(r.name)] + [_format_diff(leader, r, metric, split) for _, metric in DIFF_COLUMNS]
        for r in done[1:]
    ]
    if all(cell == MISSING for row in rows for cell in row[1:]):
        return []  # старый отчёт без per_question — сравнивать не по чему
    return [
        "",
        f"## Отличие от лидера ({_escape_cell(leader.name)})",
        "",
        "Разница «эксперимент − лидер» на тех же вопросах и её 95% доверительный интервал (парный"
        " bootstrap по вопросам). «≈» — интервал включает 0: по этим данным строки неотличимы, и их"
        " порядок в таблице задан шумом. Сравнений много, поэтому одиночный результат у самой"
        " границы интервала — ещё не твёрдый вывод.",
        "",
        *_table(["Эксперимент", *(title for title, _ in DIFF_COLUMNS)], rows),
    ]


def _by_type_section(done: Sequence[ExperimentResult]) -> list[str]:
    types = _types_in_report(done)
    if not types:
        return []
    headers: list[str] = []
    sizes: list[int] = []
    for qtype in types:
        # n — отвечаемые вопросы типа (знаменатель Recall@5); у всех экспериментов он обычно общий.
        found = {r.metrics.get(BY_TYPE_KEY, {}).get(qtype, {}).get("n_answerable") for r in done} - {None}
        if len(found) == 1:
            size = int(found.pop())
            sizes.append(size)
            headers.append(f"{qtype} (n={size})")
        else:
            headers.append(qtype)  # старый отчёт без n_answerable или разные эталоны
    rows = [
        [_escape_cell(r.name)]
        + [format_number(r.metrics.get(BY_TYPE_KEY, {}).get(qtype, {}).get(SORT_METRIC)) for qtype in types]
        for r in done
    ]
    lines = ["", "## Recall@5 по типам вопросов", ""]
    if sizes and min(sizes) > 0:
        smallest = min(sizes)
        lines += [
            f"В маленьких группах один вопрос заметно двигает метрику: при n = {smallest} это ≈ "
            f"{1 / smallest:.2f} Recall@5, так что различия в 1–2 вопроса здесь — шум.",
            "",
        ]
    return lines + _table(["Эксперимент", *headers], rows)


def _confusable_section(done: Sequence[ExperimentResult]) -> list[str]:
    """Отдельная проверка типа confusable: Recall@5 не видит путаницы версий документа."""

    def metrics(result: ExperimentResult) -> dict:
        return result.metrics.get(BY_TYPE_KEY, {}).get(CONFUSABLE_TYPE, {})

    if not any(metrics(r).get("stale_citation_rate") is not None for r in done):
        return []
    rows = [
        [_escape_cell(r.name)]
        + [
            format_number(metrics(r).get(name))
            for name in (SORT_METRIC, "fact_recall_answered", "stale_citation_rate")
        ]
        for r in done
    ]
    return [
        "",
        "## Тип confusable: не путает ли система редакции документа",
        "",
        "Recall@5 здесь обычно 1.0: нужный документ найден. Но если ответ ссылается ещё и на архивную"
        " редакцию, пользователь получает две противоречивые цифры, а FactRecall этого не замечает"
        " (нужный факт в ответе есть). Видно это только по «Архивным ссылкам».",
        "",
        *_table(["Эксперимент", "Recall@5", "FactRecall (отвеч.)", "Архивные ссылки"], rows),
    ]


def render_leaderboard(results: Sequence[ExperimentResult], split: str = "test") -> str:
    """Собирает текст ``leaderboard.md`` (без записи на диск — удобно тестировать)."""
    done = sorted((r for r in results if not r.is_skipped), key=lambda r: _sort_key(r, split))
    skipped = [r for r in results if r.is_skipped]

    lines = ["# Leaderboard", "", _intro(done, split), ""]
    if done:
        lines += ["## Поиск и задержка (от порога отказа не зависят)", ""]
        header = ["Эксперимент"] + [title for title, _ in SEARCH_COLUMNS] + ["N"]
        rows = [
            [_escape_cell(r.name)]
            + [_search_cell(r, metric, split) for _, metric in SEARCH_COLUMNS]
            + [_format_n(r.metrics.get(split, {}))]
            for r in done
        ]
        lines += _table(header, rows)
        lines += _leader_diff_section(done, split)

        lines += ["", "## Ответы и отказы (зависят от порога отказа)", "", *_threshold_note(done, split)]
        header = ["Эксперимент"] + [title for title, _ in ANSWER_COLUMNS] + ["Порог"]
        rows = [
            [_escape_cell(r.name)]
            + [format_number(r.metric(metric, split)) for _, metric in ANSWER_COLUMNS]
            + [_format_threshold(r)]
            for r in done
        ]
        lines += _table(header, rows)
    else:
        lines.append("Нет завершённых экспериментов.")

    lines += _by_type_section(done)
    lines += _confusable_section(done)

    if skipped:
        lines += ["", "## Пропущенные эксперименты", ""]
        # Причина может быть многострочной (текст исключения) — схлопываем в одну строку списка.
        lines += [f"- {_escape_cell(r.name)}: {' '.join(str(r.skipped).split())}" for r in skipped]

    lines += ["", _LEGEND.rstrip()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Запись и чтение отчёта
# ---------------------------------------------------------------------------


def environment_info() -> dict[str, Any]:
    """Где получен отчёт: версии Python и пакетов, влияющих на числа.

    Обновление torch, transformers или sentence-transformers может сдвинуть эмбеддинги,
    а scikit-learn — LSA; без этих строк расхождение двух отчётов с одинаковым конфигом
    не объяснить. Ревизии моделей Hugging Face здесь пока не фиксируются.
    """
    packages: dict[str, str | None] = {}
    for name in ENVIRONMENT_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None  # не установлен (например, офлайн-режим без нейросетей)
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def write_report(results: Sequence[ExperimentResult], out_dir: str | Path, split: str = "test") -> Path:
    """Пишет ``results.json``, ``per_question/<name>.jsonl``, ``leaderboard.md``.

    Возвращает путь к ``leaderboard.md`` — его удобно сразу открыть или напечатать.
    ``split`` — по какому сплиту строится таблица (``cfg.eval.report_split``).
    """
    _check_unique_names(results)
    out = Path(out_dir)
    per_question_dir = out / "per_question"
    per_question_dir.mkdir(parents=True, exist_ok=True)

    experiments: list[dict[str, Any]] = []
    for result in results:
        record = result.to_dict()
        # Ответы по вопросам — в отдельных JSONL: results.json остаётся компактным и читаемым.
        record.pop("per_question")
        if not result.is_skipped:
            relative = f"per_question/{safe_file_name(result.name)}.jsonl"
            lines = [_dump_json(row) + "\n" for row in result.per_question]
            (out / relative).write_text("".join(lines), encoding="utf-8", newline="\n")
            record["per_question_file"] = relative
        experiments.append(record)

    payload = {"report_split": split, "environment": environment_info(), "experiments": experiments}
    (out / "results.json").write_text(_dump_json(payload, indent=2) + "\n", encoding="utf-8", newline="\n")

    leaderboard = out / "leaderboard.md"
    leaderboard.write_text(render_leaderboard(results, split), encoding="utf-8", newline="\n")
    return leaderboard


def load_results(out_dir: str | Path) -> list[ExperimentResult]:
    """Читает отчёт обратно (results.json + per_question/*.jsonl) — например, чтобы сравнить прогоны."""
    out = Path(out_dir)
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    results: list[ExperimentResult] = []
    for record in payload.get("experiments", []):
        record = dict(record)
        relative = record.pop("per_question_file", None)
        if relative:
            text = (out / relative).read_text(encoding="utf-8")
            record["per_question"] = [json.loads(line) for line in text.splitlines() if line.strip()]
        results.append(ExperimentResult.from_dict(record))
    return results
