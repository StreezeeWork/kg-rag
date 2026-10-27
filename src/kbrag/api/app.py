"""HTTP-сервис kb-rag: ``POST /ask``, ``GET /health``, ``GET /metrics`` и веб-страница на ``GET /``.

Зачем отдельный слой, если ответ на вопрос целиком считает ``RAGPipeline``: сервис отвечает
за то, чего в пайплайне нет, — проверку входных данных, коды ошибок HTTP, метрики для
мониторинга и загрузку индекса при старте. Сам ответ даёт тот же пайплайн, что и оценка
качества (``kbrag eval``), поэтому цифры из отчёта относятся ровно к этому коду.

Запуск::

    KBRAG_CONFIG=configs/offline.yaml uvicorn kbrag.api.app:app --port 8080
    kbrag serve --config configs/offline.yaml          # то же самое через CLI

Переменные окружения: ``KBRAG_ALLOWED_HOSTS`` — допустимые Host через запятую (по умолчанию только
локальные имена; при доступе по IP или имени в сети его нужно добавить), ``KBRAG_LOG_LEVEL`` —
уровень логов kbrag (INFO). Swagger UI на ``/docs`` загружает скрипты с CDN, поэтому в закрытом
контуре без интернета он не откроется; веб-страница на ``/`` внешних ресурсов не использует.

Жизненный цикл: при импорте модуля создаётся только объект ``app`` — быстро и без индекса.
Индекс строится в ``lifespan`` один раз при старте сервера, до первого запроса. Если сборка
упала (нет корпуса, битый конфиг), сервер не стартует: лучше упасть сразу и громко,
чем часами отвечать 503 на каждый запрос.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import math
import os
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.middleware.trustedhost import TrustedHostMiddleware

from kbrag import __version__
from kbrag.config import AppConfig, load_config
from kbrag.generation import LLMError
from kbrag.types import Answer

if TYPE_CHECKING:
    # Только для подсказок типов. Настоящий импорт пайплайна (sklearn, модели) — лениво
    # в build_pipeline, чтобы `import kbrag.api.app` оставался быстрым (его использует и CLI).
    from kbrag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Какой конфиг загружает сервис. Переменная окружения удобна в Docker и docker compose.
CONFIG_ENV_VAR = "KBRAG_CONFIG"
DEFAULT_CONFIG_PATH = "configs/default.yaml"

INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"

MAX_QUESTION_CHARS = 1000
MAX_TOP_K = 20
# Тело POST /ask: вопрос до 1000 символов даже в \uXXXX-экранировании — около 6 КБ. 16 КиБ с запасом.
MAX_BODY_BYTES = 16 * 1024
_BODY_TOO_LARGE_DETAIL = (
    f"Тело запроса больше {MAX_BODY_BYTES} байт: вопрос — не длиннее {MAX_QUESTION_CHARS} символов."
)

# Защита от DNS rebinding (см. allowed_hosts_from_env). "app" — имя сервиса в docker-compose,
# "testserver" — хост, с которым ходит FastAPI TestClient в тестах.
ALLOWED_HOSTS_ENV_VAR = "KBRAG_ALLOWED_HOSTS"
DEFAULT_ALLOWED_HOSTS = ("localhost", "127.0.0.1", "[::1]", "app", "testserver")
LOG_LEVEL_ENV_VAR = "KBRAG_LOG_LEVEL"

# Флаги ответа, которые означают подозрение на prompt injection (см. kbrag.generation.guardrails).
CONTEXT_INJECTION_PREFIX = "prompt_injection"
QUESTION_INJECTION_FLAG = "question_injection"

# Prometheus различает версии текстового формата по Content-Type.
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
# JSON по стандарту (RFC 8259) всегда в UTF-8, и браузеры это знают, но не все клиенты: Windows
# PowerShell 5.1 (Invoke-RestMethod) без явного charset читает ответ как Latin-1 и показывает
# кириллицу как «ÐÑÐ¾…». Явный charset ничего не ломает и чинит таких клиентов.
JSON_CONTENT_TYPE = "application/json; charset=utf-8"
# Верхние границы бакетов гистограммы латентности, мс (+Inf добавляется автоматически).
LATENCY_BUCKETS_MS: tuple[float, ...] = (50, 100, 250, 500, 1000, 2500, 5000)
# Метка endpoint — только из этого списка. Если писать в метку любой путь из запроса,
# сканер, перебирающий /admin, /wp-login.php, …, создаст тысячи временных рядов
# и раздует память Prometheus («взрыв кардинальности»). Всё остальное — "other".
TRACKED_ENDPOINTS = frozenset({"/", "/ask", "/health", "/metrics"})
OTHER_ENDPOINT = "other"
LATENCY_ENDPOINT = "/ask"


# ---------------------------------------------------------------------------
# Конфиг сервиса
# ---------------------------------------------------------------------------


def config_path_from_env() -> str | None:
    """Какой файл конфига возьмёт сервис: ``$KBRAG_CONFIG``, иначе ``configs/default.yaml``, если он есть.

    None — файла нет, будут значения по умолчанию. Это правило в проекте одно: им пользуются
    и ``config_from_env``, и CLI (например, чтобы назвать прогон ``kbrag eval`` по имени файла).
    """
    explicit_path = os.environ.get(CONFIG_ENV_VAR)
    if explicit_path:
        return explicit_path
    return DEFAULT_CONFIG_PATH if Path(DEFAULT_CONFIG_PATH).is_file() else None


def config_from_env() -> AppConfig:
    """Конфиг из ``$KBRAG_CONFIG``, иначе ``configs/default.yaml``, иначе значения по умолчанию.

    Явно указанный в переменной файл обязан существовать (``FileNotFoundError``): опечатка
    в пути не должна молча запускать сервис с другими настройками. Если же нет и файла по
    умолчанию (запуск не из корня проекта), берутся значения по умолчанию из ``AppConfig``.
    Учтите: пути в них относительные (``data/corpus``), поэтому вне корня проекта сервис
    стартует, только если задан ``$KBRAG_CORPUS_DIR`` (или ``$KBRAG_CONFIG`` с абсолютными путями).
    """
    path = config_path_from_env()
    if path is not None:
        logger.info("Конфиг: %s", path)
        return load_config(path)
    logger.warning(
        "Файл %s не найден, а %s не задана: используются значения по умолчанию "
        "(корпус — %s или $KBRAG_CORPUS_DIR)",
        DEFAULT_CONFIG_PATH,
        CONFIG_ENV_VAR,
        AppConfig().corpus_dir,
    )
    return load_config(None)


def require_installed_extras(cfg: AppConfig) -> None:
    """Падает сразу и с подсказкой, если конфигу нужны нейросетевые пакеты, а их нет.

    Базовая установка (``make install``, CI, Docker) — офлайн, а ``configs/default.yaml`` —
    нейросетевой. Без этой проверки новичок узнавал бы об этом только после сборки индекса,
    а подсказка не говорила бы, что есть офлайн-конфиг.
    """
    from kbrag.eval.runner import config_needs_neural
    from kbrag.retrieval import neural_available

    if config_needs_neural(cfg) and not neural_available():
        raise ImportError(
            "этому конфигу нужны нейросетевые модели (sentence-transformers), а они не установлены: "
            "make install-neural (pip install -e .[neural]) — или запустите офлайн-вариант: "
            "--config configs/offline.yaml (KBRAG_CONFIG=configs/offline.yaml)"
        )


def build_pipeline(cfg: AppConfig) -> RAGPipeline:
    """Собирает пайплайн по конфигу: проверка зависимостей, чтение корпуса, нарезка, индекс, модели."""
    from kbrag.pipeline import RAGPipeline

    require_installed_extras(cfg)
    return RAGPipeline.from_config(cfg)


# ---------------------------------------------------------------------------
# Схемы запроса и ответа
# ---------------------------------------------------------------------------

# strip_whitespace срабатывает ДО проверки длины: вопрос из одних пробелов — пустой (422).
Question = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
]
# strict: true, "5" и 5.5 не превращаются молча в число — клиент с ошибкой узнает о ней сразу.
TopK = Annotated[int, Field(strict=True, ge=1, le=MAX_TOP_K)]


class AskRequest(BaseModel):
    """Тело ``POST /ask``. Невалидный запрос FastAPI сам превращает в ответ 422 с причиной."""

    # Лишнее поле («topk» вместо «top_k») — ошибка 422, а не тихо проигнорированный параметр.
    model_config = ConfigDict(extra="forbid")

    question: Question = Field(
        description="Вопрос сотрудника (пробелы по краям обрезаются)",
        examples=["Сколько дней ежегодного отпуска положено сотруднику?"],
    )
    top_k: TopK | None = Field(
        default=None, description="Сколько фрагментов передать генератору; null — значение из конфига"
    )


class CitationOut(BaseModel):
    """Ссылка ``[n]`` из текста ответа на фрагмент-источник."""

    index: int = Field(description="Номер n из ссылки [n] в тексте ответа")
    # index — номер в списке, который видел генератор; после guardrails (drop_chunk) он может не
    # совпасть с позицией в contexts. Чтобы клиенту не сверять самому по chunk_id, отдаём и её.
    context_index: int | None = Field(
        default=None, description="Позиция фрагмента-источника в contexts (ContextOut.index)"
    )
    chunk_id: str
    doc_id: str
    title: str


class ContextOut(BaseModel):
    """Найденный фрагмент — то, на чём основан ответ (показывается пользователю как источник)."""

    index: int = Field(description="Позиция фрагмента в выдаче, начиная с 1")
    chunk_id: str
    doc_id: str
    title: str
    section: str
    score: float | None = Field(
        description="Score на последнем этапе поиска (в гибриде это ранговый RRF: годится для порядка, "
        "а не для оценки релевантности)"
    )
    confidence_score: float | None = Field(
        default=None,
        description="Score того компонента поиска, по которому считается уверенность (score_source в /health)",
    )
    text: str


class AskResponse(BaseModel):
    """Ответ ``POST /ask``. Эти же поля читает веб-страница ``static/index.html``."""

    answer: str
    abstained: bool = Field(description="True — система отказалась отвечать («в базе нет ответа»)")
    confidence: float | None = Field(description="Уверенность поиска, по ней принимается решение об отказе")
    abstain_score: float | None = Field(
        default=None,
        description="Вероятность «в базе нет ответа» от классификатора отказа; null — классификатор не используется",
    )
    citations: list[CitationOut]
    contexts: list[ContextOut]
    timings_ms: dict[str, float] = Field(
        description="Время этапов: retrieve_ms, rerank_ms, generate_ms, total_ms"
    )
    flags: list[str] = Field(
        description="Служебные пометки: low_confidence, abstain_classifier, prompt_injection:<chunk_id>, …"
    )


def _finite_or_none(value: float | None) -> float | None:
    # JSON не умеет NaN и бесконечность, а numpy.float32 — не float: приводим к обычному числу.
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def build_ask_response(answer: Answer, score_source: str | None = None) -> AskResponse:
    """``Answer`` пайплайна → ответ API. Тот же формат печатает ``kbrag ask --json``.

    ``score_source`` — компонент поиска, по которому считается уверенность (``RAGPipeline.score_source``):
    его score у каждого фрагмента попадает в ``ContextOut.confidence_score``.
    """
    positions = {item.chunk.chunk_id: position for position, item in enumerate(answer.contexts, start=1)}
    citations = [
        CitationOut(
            index=c.index,
            context_index=positions.get(c.chunk_id),
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            title=c.title,
        )
        for c in answer.citations
    ]
    contexts = [
        ContextOut(
            index=position,
            chunk_id=item.chunk.chunk_id,
            doc_id=item.chunk.doc_id,
            title=item.chunk.title,
            section=item.chunk.section,
            score=_finite_or_none(item.score),
            confidence_score=_finite_or_none(item.details.get(score_source)) if score_source else None,
            text=item.chunk.text,
        )
        for position, item in enumerate(answer.contexts, start=1)
    ]
    return AskResponse(
        answer=answer.text,
        abstained=answer.abstained,
        confidence=_finite_or_none(answer.confidence),
        abstain_score=_finite_or_none(answer.abstain_score),
        citations=citations,
        contexts=contexts,
        timings_ms={name: round(float(ms), 3) for name, ms in answer.timings_ms.items()},
        flags=list(answer.flags),
    )


def count_injection_flags(flags: list[str]) -> int:
    """Сколько подозрений на prompt injection в ответе: по флагу на фрагмент и на вопрос."""
    return sum(
        1 for flag in flags if flag == QUESTION_INJECTION_FLAG or flag.startswith(CONTEXT_INJECTION_PREFIX)
    )


# ---------------------------------------------------------------------------
# Метрики Prometheus
# ---------------------------------------------------------------------------


def _format_number(value: float) -> str:
    # 50.0 → "50": так границы бакетов выглядят привычно (le="50", а не le="50.0").
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _escape_label(value: str) -> str:
    # Правила текстового формата Prometheus для значений меток.
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class ServiceMetrics:
    """Счётчики и гистограмма латентности в текстовом формате Prometheus.

    Почему без библиотеки ``prometheus_client``: метрик всего четыре, формат — простой текст,
    а лишняя зависимость — это ещё одна вещь, которую надо обновлять. Собственная
    реализация на 60 строк заодно показывает, как устроены счётчики и гистограммы.

    Почему с блокировкой: FastAPI выполняет синхронные обработчики в пуле потоков, и два
    запроса могут обновить метрики одновременно. ``x += 1`` в Python — это «прочитать →
    прибавить → записать», без блокировки часть увеличений потерялась бы.
    """

    def __init__(self, buckets_ms: tuple[float, ...] = LATENCY_BUCKETS_MS) -> None:
        self._lock = threading.Lock()
        self._buckets_ms = tuple(sorted(buckets_ms))
        self._requests: dict[tuple[str, str], int] = {}
        self._abstained = 0
        self._injection_flags = 0
        # Бакеты Prometheus накопительные: запрос за 80 мс попадает в le=100, le=250, … le=5000.
        self._bucket_counts = [0] * len(self._buckets_ms)
        self._latency_sum_ms = 0.0
        self._latency_count = 0

    def observe_request(self, endpoint: str, status: int) -> None:
        """+1 к ``kbrag_requests_total{endpoint, status}``."""
        key = (endpoint, str(status))
        with self._lock:
            self._requests[key] = self._requests.get(key, 0) + 1

    def observe_latency(self, latency_ms: float) -> None:
        """Добавляет одно наблюдение в гистограмму ``kbrag_request_latency_ms``."""
        with self._lock:
            for i, upper_bound in enumerate(self._buckets_ms):
                if latency_ms <= upper_bound:
                    self._bucket_counts[i] += 1
            self._latency_sum_ms += latency_ms
            self._latency_count += 1

    def observe_answer(self, abstained: bool, n_injection_flags: int) -> None:
        """Учитывает исход ответа: отказ и флаги prompt injection (фрагменты в выдаче и вопрос)."""
        with self._lock:
            self._abstained += int(abstained)
            self._injection_flags += n_injection_flags

    def render(self) -> str:
        """Текст для ``GET /metrics``. Строки отсортированы — вывод детерминирован."""
        with self._lock:  # снимок под блокировкой: значения согласованы между собой
            requests = sorted(self._requests.items())
            abstained, injections = self._abstained, self._injection_flags
            bucket_counts = list(self._bucket_counts)
            latency_sum, latency_count = self._latency_sum_ms, self._latency_count

        lines = [
            "# HELP kbrag_requests_total Число HTTP-запросов по эндпоинтам и кодам ответа.",
            "# TYPE kbrag_requests_total counter",
        ]
        for (endpoint, status), count in requests:
            labels = f'endpoint="{_escape_label(endpoint)}",status="{_escape_label(status)}"'
            lines.append(f"kbrag_requests_total{{{labels}}} {count}")
        lines += [
            "# HELP kbrag_abstained_total Число ответов-отказов («в базе знаний нет ответа»).",
            "# TYPE kbrag_abstained_total counter",
            f"kbrag_abstained_total {abstained}",
            # Это не «число атак»: один и тот же помеченный фрагмент считается в каждом ответе,
            # где он оказался в выдаче. Рост счётчика — повод посмотреть, какие фрагменты помечены.
            "# HELP kbrag_injection_flags_total Флаги prompt injection в ответах: по одному на каждый"
            " помеченный фрагмент в выдаче и на вопрос.",
            "# TYPE kbrag_injection_flags_total counter",
            f"kbrag_injection_flags_total {injections}",
            f"# HELP kbrag_request_latency_ms Время обработки запроса {LATENCY_ENDPOINT}, мс.",
            "# TYPE kbrag_request_latency_ms histogram",
        ]
        for upper_bound, count in zip(self._buckets_ms, bucket_counts, strict=True):
            lines.append(f'kbrag_request_latency_ms_bucket{{le="{_format_number(upper_bound)}"}} {count}')
        lines += [
            f'kbrag_request_latency_ms_bucket{{le="+Inf"}} {latency_count}',
            f"kbrag_request_latency_ms_sum {round(latency_sum, 3)}",
            f"kbrag_request_latency_ms_count {latency_count}",
        ]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Приложение
# ---------------------------------------------------------------------------


@dataclass
class ServiceState:
    """Всё, что живёт между запросами. У каждого приложения своё состояние — без глобальных переменных."""

    cfg: AppConfig | None
    pipeline: RAGPipeline | None
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)


def create_app(cfg: AppConfig | None = None, pipeline: RAGPipeline | None = None) -> FastAPI:
    """Создаёт FastAPI-приложение.

    * ``pipeline`` передан — используется как есть (в тестах подойдёт любой объект с методом
      ``answer(question, top_k=None) -> Answer`` и атрибутом ``index_stats``);
    * иначе пайплайн строится при старте сервера (``lifespan``) по ``cfg``, а если и его нет —
      по конфигу из ``config_from_env()``.
    """
    state = ServiceState(cfg=cfg if cfg is not None else getattr(pipeline, "cfg", None), pipeline=pipeline)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Код до yield выполняется один раз при старте сервера, до приёма запросов. Сборка индекса
        # блокирует поток, но здесь это нормально: пока lifespan не дошёл до yield, uvicorn всё
        # равно не принимает соединения. Исключение здесь останавливает запуск сервера.
        _configure_service_logging()
        if state.pipeline is None:
            if state.cfg is None:
                state.cfg = config_from_env()
            started = time.perf_counter()
            state.pipeline = build_pipeline(state.cfg)
            logger.info("Сервис готов к работе за %.1f с", time.perf_counter() - started)
        yield

    app = FastAPI(
        title="kb-rag",
        version=__version__,
        description="RAG-ассистент по корпоративной базе знаний: ответы со ссылками на документы.",
        lifespan=lifespan,
    )
    # Доступ к состоянию из тестов и отладчика: app.state.kbrag.metrics, app.state.kbrag.pipeline.
    app.state.kbrag = state
    # Порядок важен: каждый add_middleware оборачивает всё, что добавлено до него. Метрики —
    # самый внешний слой, поэтому они считают и отказы нижних слоёв (400 за чужой Host, 413).
    app.add_middleware(BodySizeLimitMiddleware, max_body_size=MAX_BODY_BYTES)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts_from_env())
    _add_common_headers(app)
    _add_metrics_middleware(app, state.metrics)
    _add_validation_error_handler(app)
    _add_routes(app, state)
    return app


# ---------------------------------------------------------------------------
# Защита сервиса: размер тела, Host, заголовки безопасности, ошибки валидации
# ---------------------------------------------------------------------------


class BodySizeLimitMiddleware:
    """Отвечает 413, если тело запроса больше ``max_body_size`` байт.

    Ни Starlette, ни uvicorn по умолчанию размер тела не ограничивают: запрос на 20 МБ
    читался целиком, разбирался как JSON и только потом отклонялся проверкой длины вопроса —
    несколько таких запросов параллельно легко съедают память. Здесь две проверки: по заголовку
    ``Content-Length`` (сразу, тело даже не читаем) и по реально пришедшим байтам — для тел без
    этого заголовка (chunked). Во втором случае бросаем ``HTTPException(413)``: FastAPI
    пропускает её из чтения тела как есть, и клиент получает обычный ответ ``{"detail": …}``.
    """

    def __init__(self, app: Any, max_body_size: int) -> None:
        self.app = app
        self.max_body_size = max_body_size

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers") or []).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_body_size:
            response = JSONResponse({"detail": _BODY_TOO_LARGE_DETAIL}, status_code=413)
            await response(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> dict:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_size:
                    raise HTTPException(status_code=413, detail=_BODY_TOO_LARGE_DETAIL)
            return message

        await self.app(scope, limited_receive, send)


def allowed_hosts_from_env() -> list[str]:
    """Допустимые значения заголовка Host: ``$KBRAG_ALLOWED_HOSTS`` (через запятую) или локальные имена.

    Зачем: у сервиса нет аутентификации, и без проверки Host его можно прочитать со стороннего
    сайта через DNS rebinding — браузер считает запрос «своим», а сервер принимает любой Host.
    При доступе по IP или имени в сети добавьте его в переменную; ``*`` выключает проверку.
    """
    raw = os.environ.get(ALLOWED_HOSTS_ENV_VAR, "")
    hosts = [host.strip() for host in raw.split(",") if host.strip()]
    return hosts or list(DEFAULT_ALLOWED_HOSTS)


class _InlineScriptCollector(HTMLParser):
    """Собирает текст встроенных ``<script>`` ровно так, как его видит браузер."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.scripts: list[str] = []
        self._parts: list[str] | None = None  # не None — мы внутри встроенного <script>

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Внешний скрипт (<script src=…>) по хешу не разрешается — его текст нам не нужен.
        if tag == "script" and not any(name == "src" for name, _ in attrs):
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._parts is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._parts is not None:
            self.scripts.append("".join(self._parts))
            self._parts = None


def inline_scripts(html: str) -> list[str]:
    """Тексты встроенных ``<script>`` страницы — для хешей в Content-Security-Policy.

    Почему не регулярка ``<script>(.*?)</script>``: она не знает про HTML-комментарии. В шапке
    index.html есть комментарий со словом «<script>», регулярка начинала совпадение с него,
    и хеш считался от «комментарий + стили + скрипт». Браузер хеширует только сам скрипт,
    хеши не совпадали — и браузер блокировал скрипт: страница переставала отправлять вопросы.
    HTMLParser, как и браузер, пропускает комментарии, а содержимое <script> отдаёт дословно.
    """
    collector = _InlineScriptCollector()
    collector.feed(html)
    collector.close()
    return collector.scripts


def _index_csp() -> str:
    """Content-Security-Policy для веб-страницы: только свои ресурсы и только наш встроенный скрипт.

    Хеш скрипта считается из самого файла, поэтому правка index.html не ломает страницу.
    Стили разрешены inline: страница задаёт ширину полос таймингов атрибутом style.
    """
    html = INDEX_HTML.read_text(encoding="utf-8")
    hashes = [
        "'sha256-" + base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii") + "'"
        for script in inline_scripts(html)
    ]
    script_src = " ".join(hashes) or "'none'"
    return (
        f"default-src 'self'; script-src {script_src}; "
        "style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; "
        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    )


def _add_common_headers(app: FastAPI) -> None:
    """Общие заголовки для всех ответов.

    * Защита в глубину: браузер не угадывает тип содержимого и не шлёт Referer.
    * Явная кодировка JSON (см. ``JSON_CONTENT_TYPE``). Меняем её здесь, а не через
      ``default_response_class``: ошибки 413/422/502/503 FastAPI и наши middleware отдают своими
      ``JSONResponse``, а через этот слой проходят все ответы.
    """

    @app.middleware("http")
    async def common_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if response.headers.get("content-type") == "application/json":
            response.headers["content-type"] = JSON_CONTENT_TYPE
        return response


def _add_validation_error_handler(app: FastAPI) -> None:
    """Ответ 422 без эха ввода.

    Стандартный ответ FastAPI кладёт в ``detail[].input`` присланное значение целиком: вопрос
    на мегабайт возвращался обратно, а в вопросе могут быть личные данные (в логи мы их
    специально не пишем). Клиенту для исправления хватает места ошибки и её описания.
    """

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
            for e in exc.errors()
        ]
        return JSONResponse({"detail": errors}, status_code=422)


def _configure_service_logging() -> None:
    """Включает логи kbrag, если сервис запущен напрямую через uvicorn (Docker, ``make serve``).

    uvicorn настраивает только свои логгеры, и без этого строки kbrag уровня INFO («Порог отказа
    загружен», «Сервис готов», «ask: …») терялись. Если логирование уже настроено (``kbrag serve``,
    тесты), ничего не меняем. Уровень — ``$KBRAG_LOG_LEVEL`` (по умолчанию INFO). Пустое значение
    (``KBRAG_LOG_LEVEL=`` из .env) — тоже INFO; опечатка в уровне не роняет старт сервиса, а даёт
    предупреждение и INFO.
    """
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    kbrag_logger = logging.getLogger("kbrag")
    level_name = (os.environ.get(LOG_LEVEL_ENV_VAR) or "INFO").strip().upper()
    try:
        kbrag_logger.setLevel(level_name)
    except ValueError:
        kbrag_logger.setLevel(logging.INFO)
        kbrag_logger.warning(
            "Неизвестный уровень логов %s=%r — используется INFO", LOG_LEVEL_ENV_VAR, level_name
        )


def _add_metrics_middleware(app: FastAPI, metrics: ServiceMetrics) -> None:
    """Считает каждый запрос — включая 422, которые отклоняются ещё до обработчика."""

    @app.middleware("http")
    async def collect_metrics(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        endpoint = request.url.path if request.url.path in TRACKED_ENDPOINTS else OTHER_ENDPOINT
        status = 500  # если обработчик упал с исключением, ответа нет: это внутренняя ошибка
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            metrics.observe_request(endpoint, status)
            # Латентность меряем только у /ask: быстрые служебные /health и /metrics, которые
            # опрашиваются каждые несколько секунд, «разбавили» бы гистограмму и скрыли медленные ответы.
            if endpoint == LATENCY_ENDPOINT:
                metrics.observe_latency((time.perf_counter() - started) * 1000.0)


def _add_routes(app: FastAPI, state: ServiceState) -> None:
    """Регистрирует эндпоинты. Обработчики — замыкания над ``state`` этого приложения."""

    def require_pipeline() -> RAGPipeline:
        # Как и 503 в /health: под uvicorn недостижимо (см. комментарий там), это защита без lifespan.
        if state.pipeline is None:
            raise HTTPException(
                status_code=503, detail="Сервис ещё загружает индекс. Повторите запрос позже."
            )
        return state.pipeline

    # Обычный def, а не async def: пайплайн считает синхронно (поиск, модели, HTTP к LLM).
    # Такие обработчики FastAPI выполняет в пуле потоков, и долгий ответ не блокирует остальные запросы.
    @app.post(
        "/ask",
        response_model=AskResponse,
        summary="Ответить на вопрос по базе знаний",
        responses={
            502: {"description": "LLM недоступна или вернула ошибку"},
            503: {"description": "Индекс ещё загружается"},
        },
    )
    def ask(body: AskRequest) -> AskResponse:
        pipeline = require_pipeline()
        try:
            answer = pipeline.answer(body.question, top_k=body.top_k)
        except LLMError as exc:
            # Подробности (адрес LLM, тело ответа) — только в лог: клиенту они не нужны и могут быть чувствительными.
            logger.error("Ошибка LLM при ответе на вопрос: %s", exc)
            raise HTTPException(
                status_code=502,
                detail="Сервис генерации ответов (LLM) недоступен или вернул ошибку. Повторите запрос позже.",
            ) from exc
        state.metrics.observe_answer(answer.abstained, count_injection_flags(answer.flags))
        # Сам вопрос в лог не пишем: в нём могут быть личные данные сотрудника.
        logger.info(
            "ask: %d символов, отказ=%s, %.0f мс",
            len(body.question),
            answer.abstained,
            answer.timings_ms.get("total_ms", 0.0),
        )
        return build_ask_response(answer, getattr(pipeline, "score_source", None))

    @app.get("/health", summary="Готовность сервиса и параметры индекса")
    def health(response: Response) -> dict[str, Any]:
        pipeline = state.pipeline
        if pipeline is None:
            # Под uvicorn сюда не попасть: соединения принимаются только после lifespan, а до тех
            # пор клиент (и Docker HEALTHCHECK) получает «connection refused». 503 — защита на
            # случай приложения без lifespan (тесты, ручной запуск create_app без сервера).
            response.status_code = 503
            return {"status": "starting", "version": __version__}
        stats = dict(getattr(pipeline, "index_stats", None) or {})
        return {
            "status": "ok",
            "version": __version__,
            "n_docs": stats.get("n_docs"),
            "n_chunks": stats.get("n_chunks"),
            "index_stats": stats,
            "retrieval_mode": state.cfg.retrieval.mode if state.cfg is not None else None,
            # top_k по умолчанию: страница показывает его в поле «Фрагментов». Порог отказа
            # откалиброван именно для него (уверенность всегда считается по первым top_k фрагментам).
            "top_k": state.cfg.retrieval.top_k if state.cfg is not None else None,
            "reranker": _component_name(pipeline, "reranker"),
            "generator": _component_name(pipeline, "generator"),
            # Из какого score поиска берётся «уверенность» (rerank / dense / bm25).
            "score_source": getattr(pipeline, "score_source", None),
            # Чем решается отказ сейчас: "threshold" или "classifier:<модель>" (без файла модели —
            # откат на порог, и это должно быть видно здесь, а не только в логах).
            "abstention_method": getattr(pipeline, "abstention_method", None),
            "abstention_threshold": _finite_or_none(getattr(pipeline, "threshold", None)),
        }

    @app.get("/metrics", summary="Метрики в формате Prometheus", response_class=PlainTextResponse)
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(state.metrics.render(), media_type=PROMETHEUS_CONTENT_TYPE)

    @app.get("/", include_in_schema=False)
    def index_page() -> HTMLResponse:
        if not INDEX_HTML.is_file():
            raise HTTPException(
                status_code=404, detail="Веб-страница не найдена: нет файла api/static/index.html"
            )
        # CSP — только для страницы: Swagger UI на /docs грузит скрипты с CDN и под ней бы не открылся.
        return HTMLResponse(
            INDEX_HTML.read_text(encoding="utf-8"),
            headers={"Content-Security-Policy": _index_csp(), "X-Frame-Options": "DENY"},
        )


def _component_name(pipeline: object, attribute: str) -> str | None:
    # Имя генератора/реранкера, если оно есть (у фейкового пайплайна в тестах его может не быть).
    component = getattr(pipeline, attribute, None)
    return getattr(component, "name", None) if component is not None else None


# Точка входа для `uvicorn kbrag.api.app:app`. Создание приложения дешёвое: индекс строится
# позже, в lifespan, поэтому импорт этого модуля (в тестах, в CLI) ничего не загружает.
app = create_app()
