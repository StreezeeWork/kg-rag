"""Экспорт эталонных таблиц сетки экспериментов в docs/results/.

Зачем. Каталог ``reports/`` в git не хранится (это черновики прогонов), а README и ONBOARDING
ссылаются на конкретные цифры — их нужно где-то зафиксировать вместе с условиями прогона.
Раньше таблицы копировали из ``leaderboard.md`` и правили руками; скрипт делает это
воспроизводимо:

* ``grid_neural.md`` — вся сетка; ``grid_offline.md`` — только конфигурации без нейросетей;
* из таблиц убираются столбцы задержки: p50/p95 зависят от машины и её загрузки в момент прогона,
  поэтому для эталона, который сравнивают между машинами, они вводят в заблуждение.

Запуск (после ``kbrag grid --out reports/grid_final``):

    python scripts/export_reference_results.py reports/grid_final docs/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kbrag.eval.report import load_results, render_leaderboard  # noqa: E402
from kbrag.eval.results import ExperimentResult  # noqa: E402

_LATENCY_COLUMNS = {"p50, мс", "p95, мс"}
_HEADER_RENAMES = {
    "## Поиск и задержка (от порога отказа не зависят)": "## Поиск (от порога отказа не зависит)",
}


def is_offline(result: ExperimentResult) -> bool:
    """Конфигурация без нейросетей: BM25 и/или LSA, без реранкера."""
    retrieval = result.config.get("retrieval", {})
    uses_neural_dense = (
        retrieval.get("mode") != "bm25" and retrieval.get("dense", {}).get("embedder") != "lsa"
    )
    return not uses_neural_dense and not result.config.get("rerank", {}).get("enabled", False)


def drop_latency_columns(markdown: str) -> str:
    """Удаляет столбцы p50/p95 из всех markdown-таблиц и переименовывает заголовок раздела."""
    lines = markdown.splitlines()
    out: list[str] = []
    drop: set[int] = set()
    for line in lines:
        if not line.startswith("|"):
            drop = set()  # таблица закончилась
            out.append(_HEADER_RENAMES.get(line, line))
            continue
        cells = line.strip().strip("|").split("|")
        if not drop and any(cell.strip() in _LATENCY_COLUMNS for cell in cells):
            drop = {i for i, cell in enumerate(cells) if cell.strip() in _LATENCY_COLUMNS}
        kept = [cell for i, cell in enumerate(cells) if i not in drop]
        out.append("|" + "|".join(kept) + "|")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="папка прогона kbrag grid (с results.json)")
    parser.add_argument("out_dir", type=Path, help="куда писать таблицы (обычно docs/results)")
    args = parser.parse_args()

    # Пропущенные эксперименты (например, тяжёлые) остаются в полной таблице: отчёт честно
    # показывает, что было запланировано, но не запускалось, и почему.
    results = load_results(args.run_dir)
    environment = json.loads((args.run_dir / "results.json").read_text(encoding="utf-8")).get(
        "environment", {}
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (
        ("grid_neural.md", results),
        ("grid_offline.md", [r for r in results if not r.is_skipped and is_offline(r)]),
    ):
        (args.out_dir / name).write_text(
            drop_latency_columns(render_leaderboard(subset)), encoding="utf-8", newline="\n"
        )
        print(f"{args.out_dir / name}: {len(subset)} экспериментов")
    sha = environment.get("git_sha", "?")
    print(
        f"git_sha прогона: {sha[:7]} (dirty={environment.get('git_dirty')}); укажите его в docs/results/README.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
