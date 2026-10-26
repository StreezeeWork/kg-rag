"""Проверяет, что порог отказа в git не устарел: сравнивает закоммиченный файл калибровки со свежим.

Зачем это нужно. ``reports/calibration_offline.json`` лежит в git, чтобы сервис из свежего клона
и Docker-образ сразу умели честно отвечать «нет ответа» (``abstention.threshold: auto``).
Но у закоммиченного артефакта есть риск устареть:

* ``fingerprint`` в файле защищает только от смены настроек поиска (секции chunking/retrieval/rerank):
  с чужим отпечатком сервис порог просто не применит;
* от правки корпуса или эталона отпечаток НЕ меняется, а порог (подобранный на dev) — может.

Поэтому CI пересчитывает порог (``kbrag eval --save-calibration``) и этим скриптом сравнивает его
с закоммиченным — так же, как ``kbrag build-gold --check`` сверяет gold.jsonl с частями.
Сравниваются только смысловые поля: ``fingerprint``, ``score_source`` и ``threshold``
(с допуском — на другой ОС или версии numpy последние знаки float могут отличаться).
``experiment`` и ``created_at`` — подсказки для человека, они меняются при каждом прогоне.

Запуск:

    python scripts/check_calibration.py ЗАКОММИЧЕННЫЙ.json СВЕЖИЙ.json [--tolerance 0.001]

Код возврата: 0 — порог актуален, 1 — устарел или файл не читается.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# Порог в leaderboard печатается с 3 знаками: расхождение меньше 0.001 на поведение сервиса не влияет.
DEFAULT_TOLERANCE = 1e-3
# Поля, которые обязаны совпадать точно.
EXACT_FIELDS = ("fingerprint", "score_source")

FIX_HINT = (
    "Обновите калибровку: `make eval` (или `kbrag eval --config configs/offline.yaml --save-calibration`) "
    "и закоммитьте reports/calibration_offline.json вместе с изменением, которое сдвинуло порог."
)


def compare_calibrations(
    committed: dict[str, Any], fresh: dict[str, Any], tolerance: float = DEFAULT_TOLERANCE
) -> list[str]:
    """Возвращает список расхождений (пустой — файлы равнозначны для сервиса)."""
    problems: list[str] = []
    for field in EXACT_FIELDS:
        if committed.get(field) != fresh.get(field):
            problems.append(f"{field}: в git {committed.get(field)!r}, сейчас {fresh.get(field)!r}")

    old, new = committed.get("threshold"), fresh.get("threshold")
    if old is None or new is None:
        # None — «порог не подобран». Совпасть должны оба: иначе сервис ведёт себя по-разному.
        if old is not new:
            problems.append(f"threshold: в git {old!r}, сейчас {new!r}")
    elif not math.isclose(float(old), float(new), rel_tol=0.0, abs_tol=tolerance):
        problems.append(f"threshold: в git {float(old):.4f}, сейчас {float(new):.4f} (допуск {tolerance})")
    return problems


def _read(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("ожидался JSON-объект")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сверить закоммиченный порог отказа со свежим.")
    parser.add_argument("committed", type=Path, help="файл калибровки из git")
    parser.add_argument(
        "fresh", type=Path, help="файл, только что записанный `kbrag eval --save-calibration`"
    )
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE, help="допустимое расхождение порога"
    )
    args = parser.parse_args(argv)

    try:
        committed, fresh = _read(args.committed), _read(args.fresh)
    except (OSError, ValueError) as exc:
        print(f"Не удалось прочитать файл калибровки: {exc}", file=sys.stderr)
        return 1

    problems = compare_calibrations(committed, fresh, args.tolerance)
    if problems:
        print("Калибровка порога отказа в git устарела:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(FIX_HINT, file=sys.stderr)
        return 1
    print(f"Калибровка актуальна: {args.committed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
