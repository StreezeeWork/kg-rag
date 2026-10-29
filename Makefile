# Makefile — короткие команды для частых действий: `make <цель>`. Список целей: `make help`.
#
# На Windows запускайте из Git Bash (make можно поставить: `choco install make` или `winget install GnuWin32.Make`).
# Любую переменную ниже можно переопределить в командной строке:
#   make eval CONFIG=configs/default.yaml NAME=neural_threshold METHOD=threshold
#   make test PYTHON=.venv/Scripts/python

# Интерпретатор Python (по умолчанию — тот, что первый в PATH; удобно активировать venv заранее).
PYTHON ?= python
# Конфиг для eval и serve. Офлайн по умолчанию: работает без нейросетей и интернета.
CONFIG ?= configs/offline.yaml
# Имя прогона оценки: под ним результат появится в reports/.
NAME ?= offline
# Способ отказа для eval: пусто — из конфига. В configs/default.yaml отказывает классификатор, а его порог
# хранится в файле модели, поэтому запасной порог по уверенности (reports/calibration.json) считают с METHOD=threshold.
METHOD ?=
# Файл сетки экспериментов и дополнительные флаги для неё.
EXPERIMENTS ?= configs/experiments.yaml
GRID_ARGS ?=
# Адрес локального сервера (127.0.0.1 — доступен только с этой машины).
HOST ?= 127.0.0.1
PORT ?= 8080
# Индекс колёс torch: CPU-сборка в разы меньше CUDA. Для GPU: TORCH_INDEX=https://download.pytorch.org/whl/cu124
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu
# Docker: нейросети в образе (true/false), имя образа, доп. флаги compose (например, --profile llm).
INSTALL_NEURAL ?= false
# Имена как в шапке Dockerfile: kbrag (ядро) и kbrag-neural (с torch) — сборки не затирают друг друга.
IMAGE ?= $(if $(filter true,$(INSTALL_NEURAL)),kbrag-neural,kbrag):latest
COMPOSE_ARGS ?=

# `make` без аргументов показывает справку, а не запускает что-то случайное.
.DEFAULT_GOAL := help
# Цели — это команды, а не файлы: без .PHONY файл с именем `test` сломал бы `make test`.
.PHONY: help install install-neural lint format test data eval grid serve docker-build docker-up

help: ## Показать список команд
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Установить пакет в режиме разработки: ядро + pytest и ruff
	$(PYTHON) -m pip install -e ".[dev]"

# Сначала torch из CPU-индекса, иначе pip скачает с PyPI сборку с CUDA на несколько ГБ.
install-neural: ## То же + sentence-transformers и torch (CPU) — нужно для configs/default.yaml
	$(PYTHON) -m pip install --index-url $(TORCH_INDEX) "torch>=2.6"
	$(PYTHON) -m pip install -e ".[dev,neural]"

# Те же две проверки, что в CI: ошибки и стиль (ruff check) + единое форматирование (ruff format --check).
lint: ## Проверить код: типичные ошибки и форматирование (ruff), ничего не меняя
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format: ## Отформатировать код и исправить то, что ruff умеет чинить сам
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

test: ## Запустить тесты (нейросетевые пропускаются, пока не задано KBRAG_RUN_NEURAL=1)
	$(PYTHON) -m pytest

data: ## Собрать data/eval/gold.jsonl из data/eval/parts и проверить корпус и эталон
	$(PYTHON) -m kbrag build-gold
	$(PYTHON) -m kbrag validate-data

# --save-calibration: заодно сохранить подобранный на dev порог отказа в abstention.calibration_path
# конфига (offline → reports/calibration_offline.json, default → reports/calibration.json).
# Этот файл читает сервис: без него `make serve` и Docker-образ никогда не отказываются от ответа.
# Файлы калибровки лежат в git — после изменения поиска, корпуса или эталона закоммитьте обновлённый.
# Для configs/default.yaml (отказ — классификатор) нужен METHOD=threshold: иначе kbrag eval завершится с кодом 1.
eval: ## Оценить конфиг и сохранить порог отказа для сервиса (нейро: CONFIG=configs/default.yaml METHOD=threshold)
	KBRAG_ABSTENTION_METHOD=$(METHOD) $(PYTHON) -m kbrag eval --config $(CONFIG) --name $(NAME) --save-calibration

grid: ## Прогнать сетку экспериментов → reports/grid/leaderboard.md (GRID_ARGS="--include-heavy")
	$(PYTHON) -m kbrag grid --experiments $(EXPERIMENTS) $(GRID_ARGS)

serve: ## Запустить API и веб-страницу на http://127.0.0.1:8080 с перезапуском при правке кода
	KBRAG_CONFIG=$(CONFIG) $(PYTHON) -m uvicorn kbrag.api.app:app --host $(HOST) --port $(PORT) --reload

docker-build: ## Собрать Docker-образ (make docker-build INSTALL_NEURAL=true — с нейросетями)
	docker build --build-arg INSTALL_NEURAL=$(INSTALL_NEURAL) -t $(IMAGE) .

# INSTALL_NEURAL, KBRAG_CONFIG и др. compose читает из окружения или файла .env;
# переменные из командной строки make (`make docker-up INSTALL_NEURAL=true`) тоже попадают в окружение.
docker-up: ## Поднять сервис в docker compose (с LLM: make docker-up COMPOSE_ARGS="--profile llm")
	docker compose $(COMPOSE_ARGS) up --build
