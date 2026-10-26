"""Позволяет запускать CLI как модуль: ``python -m kbrag <команда>``.

Удобно, когда пакет установлен (``pip install -e .``), но скрипт ``kbrag`` не в PATH
(venv не активирован), или когда нужно явно выбрать интерпретатор:
``.venv/Scripts/python -m kbrag eval``. Без установки пакета (src-layout) модуль не найдётся.
"""

from kbrag.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
