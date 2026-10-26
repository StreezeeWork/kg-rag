"""Отчёт по экспериментам: машиночитаемый JSON + человекочитаемый leaderboard.md.

Что пишется в ``out_dir``:

* ``results.json`` — конфиги, метрики, пороги и статистика индекса всех экспериментов;
* ``per_question/<name>.jsonl`` — ответ и метрики по каждому вопросу (для разбора ошибок);
* ``leaderboard.md`` — сводная таблица и разбивка Recall@5 по типам вопросов.

Что воспроизводимо, а что нет. Отчёт не содержит даты, поэтому при повторном прогоне с тем же
конфигом **метрики качества, пороги и ответы** совпадают байт в байт. Тайминги — нет:
``latency_ms``, ``latency_p50_ms``, ``latency_p95_ms`` и ``index_stats.index_time_ms``
(список — ``results.TIMING_KEYS``) меряются одним проходом без прогрева и зависят от
нагрузки машины. При сравнении двух отчётов в git их строки надо игнорировать.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kbrag.eval.dataset import QUESTION_TYPES
from kbrag.eval.results import BY_TYPE_KEY, ExperimentResult

# (заголовок столбца, имя метрики) — порядок задаёт порядок столбцов.
LEADERBOARD_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Recall@5", "recall@5"),
    ("MRR", "mrr"),
    ("nDCG@5", "ndcg@5"),
    ("Hit@1", "hit@1"),
    ("FactRecall", "fact_recall"),
    ("CitationPrecision", "citation_precision"),
    ("Abstention F1", "abstain_f1"),
    ("p50, мс", "latency_p50_ms"),
    ("p95, мс", "latency_p95_ms"),
)
SORT_METRIC = "recall@5"
MISSING = "—"

_LEGEND = """## Как читать таблицу

- **Recall@5** — доля доказательств (цитат из эталона), найденных среди первых 5 фрагментов. Главная метрика поиска.
- **MRR** — 1 / позиция первого релевантного фрагмента, усреднённое по вопросам.
- **nDCG@5** — учитывает позиции всех найденных доказательств: чем выше, тем лучше.
- **Hit@1** — доля вопросов, где первый же фрагмент релевантен.
- **FactRecall** — доля ключевых фактов эталона, которые есть в ответе.
- **CitationPrecision** — доля ссылок `[n]` в ответе, ведущих на фрагмент с доказательством.
- **Abstention F1** — насколько точно система отказывается отвечать на вопросы, ответа на которые в базе нет.
- **p50 / p95, мс** — медианная и «хвостовая» задержка ответа. Это один прогон на этой машине без прогрева: от запуска к запуску цифры заметно гуляют, поэтому годятся только для грубого сравнения строк одного отчёта.
- **Порог** — порог уверенности поиска для отказа (подобран на dev; «—» — без порога).
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


def render_leaderboard(results: Sequence[ExperimentResult], split: str = "test") -> str:
    """Собирает текст ``leaderboard.md`` (без записи на диск — удобно тестировать)."""
    done = sorted((r for r in results if not r.is_skipped), key=lambda r: _sort_key(r, split))
    skipped = [r for r in results if r.is_skipped]

    lines = [
        "# Leaderboard",
        "",
        f"Метрики посчитаны на сплите `{split}`. Строки отсортированы по Recall@5 (по убыванию).",
        "",
    ]
    if done:
        header = ["Эксперимент"] + [title for title, _ in LEADERBOARD_COLUMNS] + ["Порог", "N"]
        rows = [
            [_escape_cell(r.name)]
            + [format_number(r.metric(metric, split)) for _, metric in LEADERBOARD_COLUMNS]
            + [format_number(r.threshold), format_number(r.metric("n_questions", split))]
            for r in done
        ]
        lines += _table(header, rows)
    else:
        lines.append("Нет завершённых экспериментов.")

    types = _types_in_report(done)
    if types:
        lines += ["", "## Recall@5 по типам вопросов", ""]
        rows = [
            [_escape_cell(r.name)]
            + [
                format_number(r.metrics.get(BY_TYPE_KEY, {}).get(qtype, {}).get(SORT_METRIC))
                for qtype in types
            ]
            for r in done
        ]
        lines += _table(["Эксперимент", *types], rows)

    if skipped:
        lines += ["", "## Пропущенные эксперименты", ""]
        # Причина может быть многострочной (текст исключения) — схлопываем в одну строку списка.
        lines += [f"- {_escape_cell(r.name)}: {' '.join(str(r.skipped).split())}" for r in skipped]

    lines += ["", _LEGEND.rstrip()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Запись и чтение отчёта
# ---------------------------------------------------------------------------


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

    payload = {"report_split": split, "experiments": experiments}
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
