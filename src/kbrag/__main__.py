"""Позволяет запускать CLI как модуль: ``python -m kbrag <команда>``.

Удобно, когда пакет не установлен (нет команды ``kbrag`` в PATH) или нужно явно выбрать
интерпретатор: ``.venv/Scripts/python -m kbrag eval``.
"""

from kbrag.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
