# syntax=docker/dockerfile:1
#
# Образ сервиса kb-rag: FastAPI-приложение (POST /ask, GET /health, GET /metrics) и веб-страница на порту 8080.
#
# Сборка:
#   docker build -t kbrag .                                            # ядро: BM25 + LSA, без torch
#   docker build -t kbrag-neural --build-arg INSTALL_NEURAL=true .     # + sentence-transformers, torch (CPU) и catboost
#                                                                      #   (модель отказа configs/default.yaml — CatBoost)
# Запуск (127.0.0.1 — порт открыт только для этой машины: у сервиса нет ни авторизации, ни лимита
# запросов, поэтому наружу его выставляют только за reverse proxy с авторизацией и rate limit):
#   docker run --rm -p 127.0.0.1:8080:8080 kbrag
#   docker run --rm -p 127.0.0.1:8080:8080 -v kbrag-cache:/app/.cache \
#       -e KBRAG_CONFIG=configs/default.yaml kbrag-neural
# Том kbrag-cache хранит скачанные модели (~1 ГБ): без него они качаются при каждом запуске.
# Проще всего — через docker compose (см. docker-compose.yml) или `make docker-up`.
# Нужен BuildKit (по умолчанию в Docker 23+ и Docker Desktop): ниже используется RUN --mount.

FROM python:3.12-slim

# PYTHONDONTWRITEBYTECODE — не создаём .pyc в образе; PYTHONUNBUFFERED — логи сразу видны в `docker logs`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Непривилегированный пользователь: даже если в сервисе найдётся уязвимость,
# у атакующего не будет прав root внутри контейнера.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home app

# true → ставим нейросетевые зависимости (нужны для configs/default.yaml и экспериментов с e5/bge).
ARG INSTALL_NEURAL=false

# torch ставим отдельным слоем и ДО копирования кода: он весит сотни мегабайт, а так Docker
# берёт его из кэша при каждой правке кода. Колёса CPU-only в разы меньше сборки с CUDA.
# --mount=type=cache: скачанные колёса pip живут в кэше сборки, а не в слое образа. Образ не толстеет,
# а после правки кода `pip install .` берёт numpy, scikit-learn и др. из кэша, не скачивая заново.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_NEURAL" = "true" ]; then \
        pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6"; \
    fi

# Сначала только то, что нужно для установки пакета (README.md указан в pyproject.toml как readme).
COPY pyproject.toml README.md ./
COPY src ./src
# torch уже стоит (если нужен), поэтому pip не будет тянуть тяжёлую CUDA-сборку с PyPI.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_NEURAL" = "true" ]; then \
        pip install ".[neural,classic]"; \
    else \
        pip install .; \
    fi

# Данные и конфиги — отдельными слоями: правка документа не заставляет переустанавливать пакет.
# Из data/ в образ попадает только корпус: эталон data/eval сервису не нужен (см. .dockerignore).
COPY configs ./configs
COPY data ./data
# Из reports/ в образ попадают только файлы калибровки порога отказа (см. .dockerignore).
# reports/calibration_offline.json лежит в git, поэтому офлайн-образ даже из свежего клона сразу
# умеет отвечать «нет ответа». Для configs/default.yaml порог появится после
# `make eval CONFIG=configs/default.yaml NAME=neural` (reports/calibration.json) и пересборки образа.
# Порог не подхватился — GET /health покажет "abstention_threshold": null.
COPY reports/ ./reports/

# Каталоги, куда сервис пишет во время работы (кэш моделей и эмбеддингов, отчёты), принадлежат app.
RUN mkdir -p .cache reports && chown -R app:app .cache reports

# Пути относительно /app — так же, как при локальном запуске из корня проекта.
# Офлайн-конфиг по умолчанию, потому что базовый образ собран без torch;
# для образа с INSTALL_NEURAL=true передайте -e KBRAG_CONFIG=configs/default.yaml.
ENV KBRAG_CONFIG=configs/offline.yaml \
    HF_HOME=/app/.cache/huggingface

USER app

EXPOSE 8080

# В slim-образе нет curl, поэтому проверяем /health стандартной библиотекой Python.
# start-period с запасом: при первом старте нейро-образ скачивает модели.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"]

# Один процесс uvicorn: индекс и модели живут в памяти процесса, лишние воркеры только умножат RAM.
CMD ["uvicorn", "kbrag.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
