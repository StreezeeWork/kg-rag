"""HTTP-сервис kb-rag (FastAPI): ``POST /ask``, ``GET /health``, ``GET /metrics`` и веб-страница.

Вся логика — в ``kbrag.api.app``. Типичное использование::

    from kbrag.api import create_app
    app = create_app(cfg)              # индекс строится при старте сервера
    app = create_app(pipeline=fake)    # в тестах: готовый (или фейковый) пайплайн

Для uvicorn используйте модульную переменную: ``uvicorn kbrag.api.app:app``.
"""

from kbrag.api.app import create_app

__all__ = ["create_app"]
