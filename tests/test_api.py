"""Тесты HTTP-сервиса: схема ответа /ask, валидация (422), коды ошибок, /health, /metrics, /.

Почти везде используется фейковый пайплайн: так тест проверяет именно слой API
(валидацию, сериализацию, метрики), а не качество поиска. Один тест в конце гоняет
настоящий RAGPipeline на мини-корпусе (BM25 + extractive, офлайн и за миллисекунды).
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kbrag.api.app import (
    INDEX_HTML,
    LATENCY_BUCKETS_MS,
    MAX_BODY_BYTES,
    AskResponse,
    ServiceMetrics,
    _configure_service_logging,
    build_ask_response,
    config_from_env,
    count_injection_flags,
    create_app,
    inline_scripts,
)
from kbrag.config import AppConfig
from kbrag.generation import LLMError
from kbrag.generation.citations import CITATION_RE
from kbrag.pipeline import RAGPipeline
from kbrag.types import Answer, Chunk, Citation, Document, ScoredChunk

# abstain_score — P(нет ответа) от классификатора отказа (null при пороговом методе).
RESPONSE_FIELDS = {
    "answer",
    "abstained",
    "confidence",
    "abstain_score",
    "citations",
    "contexts",
    "timings_ms",
    "flags",
}
CITATION_FIELDS = {"index", "context_index", "chunk_id", "doc_id", "title"}
CONTEXT_FIELDS = {"index", "chunk_id", "doc_id", "title", "section", "score", "confidence_score", "text"}
ASK_REQUESTS = 'kbrag_requests_total{endpoint="/ask",status="%s"}'


def scored(doc_id: str, n: int, text: str, score: float, section: str = "1. Раздел") -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"{doc_id}::{n:03d}",
        doc_id=doc_id,
        text=text,
        start=0,
        end=len(text),
        title=f"Документ {doc_id}",
        section=section,
    )
    return ScoredChunk(chunk=chunk, score=score, source="bm25", details={"bm25": score})


def make_answer(abstained: bool = False, flags: list[str] | None = None) -> Answer:
    contexts = [
        # numpy.float32 — так score приходит из dense-поиска; API обязан превратить его в обычное число.
        scored("hr-vacation", 0, "Отпуск — 28 календарных дней.", np.float32(6.5)),
        scored("it-vpn", 1, "Клиент VPN — «Шлюз».", 1.25, section=""),
    ]
    return Answer(
        question="Сколько дней отпуска?",
        text="Отпуск — 28 календарных дней [1].",
        citations=[
            Citation(index=1, chunk_id="hr-vacation::000", doc_id="hr-vacation", title="Документ hr-vacation")
        ],
        contexts=contexts,
        abstained=abstained,
        timings_ms={"retrieve_ms": 1.23456, "rerank_ms": 0.0, "generate_ms": 2.0, "total_ms": 3.5},
        flags=list(flags or []),
        confidence=0.42,
    )


class FakePipeline:
    """Заменитель RAGPipeline: ровно то, что нужно API, — метод answer() и атрибут index_stats."""

    def __init__(self, answer: Answer | None = None, error: Exception | None = None) -> None:
        self.index_stats = {"n_docs": 2, "n_chunks": 3, "avg_chunk_words": 12.5, "index_time_ms": 1.0}
        self.calls: list[tuple[str, int | None]] = []
        self._answer = answer or make_answer()
        self._error = error

    def answer(self, question: str, top_k: int | None = None) -> Answer:
        self.calls.append((question, top_k))
        if self._error is not None:
            raise self._error
        return self._answer


def metric_value(metrics_text: str, sample: str) -> float:
    """Значение строки метрики (``имя{метки} значение``); 0, если строки ещё нет."""
    for line in metrics_text.splitlines():
        if line.startswith(sample + " "):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


@pytest.fixture()
def fake() -> FakePipeline:
    return FakePipeline()


@pytest.fixture()
def client(fake: FakePipeline):
    cfg = AppConfig.model_validate({"retrieval": {"mode": "bm25"}})
    with TestClient(create_app(cfg=cfg, pipeline=fake)) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------


def test_ask_returns_exactly_the_fields_the_web_page_uses(client: TestClient, fake: FakePipeline) -> None:
    response = client.post("/ask", json={"question": "  Сколько дней отпуска?  ", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == RESPONSE_FIELDS
    assert body["answer"] == "Отпуск — 28 календарных дней [1]."
    assert body["abstained"] is False
    assert body["confidence"] == pytest.approx(0.42)
    assert body["abstain_score"] is None  # классификатор отказа не используется
    assert body["citations"] == [
        {
            "index": 1,
            "context_index": 1,
            "chunk_id": "hr-vacation::000",
            "doc_id": "hr-vacation",
            "title": "Документ hr-vacation",
        }
    ]
    assert all(set(ctx) == CONTEXT_FIELDS for ctx in body["contexts"])
    # Нумерация с 1 — так же, как ссылки [n] в тексте ответа.
    assert [ctx["index"] for ctx in body["contexts"]] == [1, 2]
    first = body["contexts"][0]
    assert first["chunk_id"] == "hr-vacation::000"
    assert first["section"] == "1. Раздел"
    assert first["text"] == "Отпуск — 28 календарных дней."
    assert isinstance(first["score"], float) and first["score"] == pytest.approx(6.5)
    assert body["timings_ms"]["retrieve_ms"] == 1.235  # округление до микросекунд
    # Пайплайн получил вопрос уже без пробелов по краям и top_k как есть.
    assert fake.calls == [("Сколько дней отпуска?", 2)]


def test_ask_without_top_k_lets_pipeline_use_config_default(client: TestClient, fake: FakePipeline) -> None:
    assert client.post("/ask", json={"question": "Вопрос"}).status_code == 200
    assert client.post("/ask", json={"question": "Вопрос", "top_k": None}).status_code == 200
    assert fake.calls == [("Вопрос", None), ("Вопрос", None)]


def test_question_of_exactly_max_length_is_accepted(client: TestClient, fake: FakePipeline) -> None:
    question = "я" * 1000
    response = client.post("/ask", json={"question": f"   {question}   ", "top_k": 20})
    assert response.status_code == 200
    assert fake.calls == [(question, 20)]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"question": ""}, id="empty-question"),
        pytest.param({"question": "   \n\t "}, id="whitespace-only"),
        pytest.param({"question": "я" * 1001}, id="too-long"),
        pytest.param({"question": "Вопрос", "top_k": 0}, id="top_k-zero"),
        pytest.param({"question": "Вопрос", "top_k": 21}, id="top_k-too-big"),
        pytest.param({"question": "Вопрос", "top_k": "5"}, id="top_k-string"),
        pytest.param({"question": "Вопрос", "top_k": 2.5}, id="top_k-float"),
        pytest.param({"question": "Вопрос", "topk": 3}, id="unknown-field"),
        pytest.param({}, id="no-question"),
    ],
)
def test_invalid_request_returns_422_without_calling_pipeline(
    client: TestClient, fake: FakePipeline, payload: dict
) -> None:
    response = client.post("/ask", json=payload)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)  # FastAPI объясняет, какое поле не так
    assert fake.calls == []


def test_llm_error_becomes_502_without_leaking_internal_details() -> None:
    pipeline = FakePipeline(error=LLMError("HTTP 500 от http://llm-internal:8000/v1: stack trace…"))
    with TestClient(create_app(pipeline=pipeline)) as client:
        response = client.post("/ask", json={"question": "Вопрос"})
        metrics = client.get("/metrics").text

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "LLM" in detail
    assert "llm-internal" not in detail
    assert metric_value(metrics, ASK_REQUESTS % 502) == 1


def test_unexpected_error_is_counted_as_500() -> None:
    pipeline = FakePipeline(error=RuntimeError("баг"))
    with TestClient(create_app(pipeline=pipeline), raise_server_exceptions=False) as client:
        assert client.post("/ask", json={"question": "Вопрос"}).status_code == 500
        assert metric_value(client.get("/metrics").text, ASK_REQUESTS % 500) == 1


def test_returns_503_while_pipeline_is_not_built() -> None:
    # TestClient без `with` не запускает lifespan — это и есть состояние «индекс ещё строится».
    client = TestClient(create_app(cfg=AppConfig()))

    ask = client.post("/ask", json={"question": "Вопрос"})
    health = client.get("/health")

    assert ask.status_code == 503
    assert "индекс" in ask.json()["detail"]
    assert health.status_code == 503
    assert health.json()["status"] == "starting"


# ---------------------------------------------------------------------------
# GET /health, GET /
# ---------------------------------------------------------------------------


def test_health_reports_index_and_components(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["n_docs"] == 2
    assert body["n_chunks"] == 3
    assert body["index_stats"]["avg_chunk_words"] == 12.5
    assert body["retrieval_mode"] == "bm25"
    assert body["top_k"] == AppConfig().retrieval.top_k  # подсказка для поля «Фрагментов» на странице
    # У фейкового пайплайна нет генератора и реранкера — сервис не падает, а честно пишет null.
    assert body["generator"] is None
    assert body["reranker"] is None
    assert body["score_source"] is None


def test_index_page_is_served_as_html(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>kb-rag" in response.text


def test_index_page_leaves_top_k_to_the_server_by_default(client: TestClient) -> None:
    # Порог отказа откалиброван на top_k из конфига: страница не должна по умолчанию слать своё число.
    page = client.get("/").text
    top_k_input = next(line for line in page.splitlines() if 'id="top-k"' in line)
    assert "value=" not in top_k_input


def test_citation_reports_its_position_in_contexts_and_confidence_scores_are_exposed() -> None:
    # После guardrails (drop_chunk) генератор видит не все фрагменты: [1] в ответе — это второй
    # фрагмент выдачи. Клиенту не нужно сверять это самому по chunk_id.
    answer = make_answer()
    answer.citations = [Citation(index=1, chunk_id="it-vpn::001", doc_id="it-vpn", title="Документ it-vpn")]

    response = build_ask_response(answer, score_source="bm25")

    assert response.citations[0].index == 1 and response.citations[0].context_index == 2
    # score — с последнего этапа поиска, confidence_score — тот компонент, по которому считается уверенность.
    assert [c.confidence_score for c in response.contexts] == [pytest.approx(6.5), pytest.approx(1.25)]
    assert build_ask_response(answer).contexts[0].confidence_score is None


def test_page_citation_regex_matches_the_server_one() -> None:
    # Страница сама размечает [n] в тексте ответа; её регулярка обязана понимать те же форматы,
    # что parse_citations на сервере ([3; 1], [ 2 ]), иначе такие ссылки показываются простым текстом.
    page = INDEX_HTML.read_text(encoding="utf-8")
    js_pattern = re.search(r"const CITATION_RE = /(.+)/g;", page).group(1)
    assert js_pattern == CITATION_RE.pattern


# ---------------------------------------------------------------------------
# Защита сервиса: размер тела, Host, заголовки, ответ 422
# ---------------------------------------------------------------------------


def test_too_large_body_is_rejected_with_413_and_counted(client: TestClient, fake: FakePipeline) -> None:
    big = '{"question": "' + "а" * (MAX_BODY_BYTES) + '"}'
    by_length = client.post("/ask", content=big.encode("utf-8"), headers={"Content-Type": "application/json"})

    def chunks():  # без Content-Length (chunked): размер считается по пришедшим байтам
        yield b'{"question": "'
        for _ in range(4):
            yield ("я" * MAX_BODY_BYTES).encode("utf-8")
        yield b'"}'

    streamed = client.post("/ask", content=chunks(), headers={"Content-Type": "application/json"})

    assert by_length.status_code == 413 and streamed.status_code == 413
    assert "detail" in by_length.json() and "detail" in streamed.json()
    assert fake.calls == []
    assert metric_value(client.get("/metrics").text, ASK_REQUESTS % 413) == 2


def test_validation_error_does_not_echo_the_input(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "Вопрос", "extra": "<script>alert(1)</script>"})
    too_long = client.post("/ask", json={"question": "секрет " * 300})

    assert response.status_code == 422 and too_long.status_code == 422
    assert "<script>" not in response.text and "секрет" not in too_long.text
    detail = response.json()["detail"]
    assert detail and set(detail[0]) == {"loc", "msg", "type"}


def test_unknown_host_header_is_rejected(client: TestClient) -> None:
    # Защита от DNS rebinding: страница злоумышленника не сможет читать локальный сервис.
    assert client.get("/health", headers={"host": "attacker.example"}).status_code == 400
    assert client.get("/health", headers={"host": "localhost:8080"}).status_code == 200


def test_allowed_hosts_come_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KBRAG_ALLOWED_HOSTS", "kb.intranet, testserver")
    with TestClient(create_app(pipeline=FakePipeline())) as client:
        assert client.get("/health", headers={"host": "kb.intranet"}).status_code == 200
        assert client.get("/health", headers={"host": "localhost"}).status_code == 400


def test_every_json_response_declares_utf8(client: TestClient) -> None:
    # Регрессия: без charset Windows PowerShell 5.1 показывал кириллицу в ответах как «ÐÑÐ¾…».
    # Проверяем и успешные ответы, и ошибки — их FastAPI и middleware отдают своими JSONResponse.
    responses = {
        "ask": client.post("/ask", json={"question": "Сколько дней отпуска?"}),
        "health": client.get("/health"),
        "422": client.post("/ask", json={"question": ""}),
        "413": client.post(
            "/ask", content=b"x" * (MAX_BODY_BYTES + 1), headers={"Content-Type": "application/json"}
        ),
        # TestClient без `with` не запускает lifespan: пайплайн не построен → 503.
        "503": TestClient(create_app(cfg=AppConfig())).post("/ask", json={"question": "Вопрос"}),
    }
    for name, response in responses.items():
        assert response.headers["content-type"] == "application/json; charset=utf-8", name
    # Не-JSON ответы сохраняют свои типы.
    assert client.get("/metrics").headers["content-type"].startswith("text/plain; version=0.0.4")
    assert client.get("/").headers["content-type"].startswith("text/html")


def test_security_headers_and_page_csp(client: TestClient) -> None:
    import base64
    import hashlib

    health = client.get("/health")
    page = client.get("/")

    for response in (health, page):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
    csp = page.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in csp and "default-src 'self'" in csp
    # Разрешён ровно встроенный скрипт страницы — по хешу, а не 'unsafe-inline'.
    # Скрипт извлекаем НЕ той же функцией, что сервер: сначала вырезаем HTML-комментарии, как это
    # делает браузер. Раньше тест повторял регулярку сервера и вместе с ней ошибался: в комментарии
    # наверху страницы есть слово «<script>», хеш считался не от того текста, и браузер блокировал скрипт.
    without_comments = re.sub(r"<!--.*?-->", "", page.text, flags=re.DOTALL)
    script = re.search(r"<script>(.*?)</script>", without_comments, re.DOTALL).group(1)
    assert "use strict" in script  # это действительно код страницы, а не кусок разметки
    digest = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii")
    assert f"'sha256-{digest}'" in csp and "unsafe-inline" not in csp.split("script-src")[1].split(";")[0]


def test_inline_scripts_skip_comments_and_external_scripts() -> None:
    html = (
        "<!-- пример вредной вставки: <script>alert(1)</script> -->\n"
        '<p>текст</p><script src="lib.js"></script>\n'
        "<script>run('<b>ok</b>');</script>"
    )
    assert inline_scripts(html) == ["run('<b>ok</b>');"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, logging.INFO), ("", logging.INFO), ("warning", logging.WARNING), ("verbose", logging.INFO)],
)
def test_service_logging_is_configured_when_uvicorn_left_it_empty(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: int
) -> None:
    # uvicorn настраивает только свои логгеры: без этого INFO-строки kbrag («порог загружен»,
    # «сервис готов») в Docker терялись. Пустой KBRAG_LOG_LEVEL (так он лежит в .env.example)
    # и опечатка в уровне не должны ронять старт сервиса.
    kbrag_logger = logging.getLogger("kbrag")
    monkeypatch.setattr(logging.root, "handlers", [])
    monkeypatch.setattr(kbrag_logger, "level", kbrag_logger.level)
    if value is not None:
        monkeypatch.setenv("KBRAG_LOG_LEVEL", value)

    _configure_service_logging()

    assert logging.root.handlers
    assert kbrag_logger.getEffectiveLevel() == expected


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


def test_metrics_counters_grow_with_requests() -> None:
    pipeline = FakePipeline(
        answer=make_answer(abstained=True, flags=["prompt_injection:it-vpn::001", "question_injection"])
    )
    with TestClient(create_app(pipeline=pipeline)) as client:
        before = client.get("/metrics").text
        client.post("/ask", json={"question": "Вопрос 1"})
        client.post("/ask", json={"question": "Вопрос 2"})
        client.post("/ask", json={"question": ""})  # 422 тоже считается
        response = client.get("/metrics")

    after = response.text
    assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert metric_value(before, ASK_REQUESTS % 200) == 0
    assert metric_value(after, ASK_REQUESTS % 200) == 2
    assert metric_value(after, ASK_REQUESTS % 422) == 1
    assert metric_value(after, 'kbrag_requests_total{endpoint="/metrics",status="200"}') == 1
    assert metric_value(after, "kbrag_abstained_total") == 2
    assert metric_value(after, "kbrag_injection_flags_total") == 4  # по 2 флага в каждом из двух ответов
    # Гистограмма: все три запроса /ask (включая 422) попали в count и в бакет +Inf.
    assert metric_value(after, "kbrag_request_latency_ms_count") == 3
    assert metric_value(after, 'kbrag_request_latency_ms_bucket{le="+Inf"}') == 3
    assert "# TYPE kbrag_request_latency_ms histogram" in after


def test_unknown_paths_share_one_label_to_avoid_cardinality_explosion(client: TestClient) -> None:
    for path in ("/wp-login.php", "/admin", "/a/b/c"):
        assert client.get(path).status_code == 404
    metrics = client.get("/metrics").text
    assert metric_value(metrics, 'kbrag_requests_total{endpoint="other",status="404"}') == 3
    assert "wp-login" not in metrics


def test_histogram_buckets_are_cumulative_and_inclusive() -> None:
    metrics = ServiceMetrics()
    for latency in (10.0, 50.0, 60.0, 300.0, 6000.0):
        metrics.observe_latency(latency)
    text = metrics.render()

    expected = {"50": 2, "100": 3, "250": 3, "500": 4, "1000": 4, "2500": 4, "5000": 4, "+Inf": 5}
    for le, count in expected.items():
        assert metric_value(text, f'kbrag_request_latency_ms_bucket{{le="{le}"}}') == count, le
    assert metric_value(text, "kbrag_request_latency_ms_sum") == pytest.approx(6420.0)
    assert metric_value(text, "kbrag_request_latency_ms_count") == 5
    # Бакеты в порядке возрастания границ — так их ждёт Prometheus.
    bucket_lines = [line for line in text.splitlines() if line.startswith("kbrag_request_latency_ms_bucket")]
    assert [line.split('"')[1] for line in bucket_lines] == [*(str(b) for b in LATENCY_BUCKETS_MS), "+Inf"]


def test_service_metrics_are_thread_safe() -> None:
    metrics = ServiceMetrics()
    n_threads, n_iterations = 8, 2000

    def hammer() -> None:
        for _ in range(n_iterations):
            metrics.observe_request("/ask", 200)
            metrics.observe_latency(1.0)
            metrics.observe_answer(abstained=True, n_injection_flags=1)

    threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    total = n_threads * n_iterations
    text = metrics.render()
    assert metric_value(text, ASK_REQUESTS % 200) == total
    assert metric_value(text, "kbrag_abstained_total") == total
    assert metric_value(text, "kbrag_injection_flags_total") == total
    assert metric_value(text, "kbrag_request_latency_ms_count") == total


def test_count_injection_flags_ignores_other_flags() -> None:
    flags = ["low_confidence", "prompt_injection:a::000", "prompt_injection:b::001", "question_injection"]
    assert count_injection_flags(flags) == 3
    assert count_injection_flags([]) == 0


# ---------------------------------------------------------------------------
# Конфиг и жизненный цикл
# ---------------------------------------------------------------------------


def test_module_level_app_does_not_build_pipeline_on_import() -> None:
    import kbrag.api.app as app_module

    assert isinstance(app_module.app, FastAPI)
    assert app_module.app.state.kbrag.pipeline is None


def test_lifespan_builds_pipeline_from_kbrag_config(
    tmp_path: Path, mini_corpus_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "service.yaml"
    config = {
        "corpus_dir": str(mini_corpus_dir),
        "retrieval": {"mode": "bm25"},
        "abstention": {"threshold": None},
    }
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("KBRAG_CONFIG", str(config_path))

    app = create_app()
    assert app.state.kbrag.pipeline is None  # до старта сервера ничего не строится
    with TestClient(app) as client:
        health = client.get("/health").json()

    assert health["n_docs"] == 4
    assert health["retrieval_mode"] == "bm25"
    assert health["generator"] == "extractive"


def test_config_from_env_falls_back_to_defaults_without_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # здесь нет configs/default.yaml
    assert config_from_env() == AppConfig()


def test_config_from_env_reads_default_file_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text("retrieval:\n  mode: bm25\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert config_from_env().retrieval.mode == "bm25"


def test_config_from_env_fails_on_missing_explicit_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Опечатка в KBRAG_CONFIG не должна молча запускать сервис с настройками по умолчанию.
    monkeypatch.setenv("KBRAG_CONFIG", str(tmp_path / "missing.yaml"))
    with pytest.raises(FileNotFoundError):
        config_from_env()


# ---------------------------------------------------------------------------
# Настоящий пайплайн
# ---------------------------------------------------------------------------


def test_ask_with_real_offline_pipeline(mini_docs: list[Document]) -> None:
    cfg = AppConfig.model_validate(
        {
            "retrieval": {"mode": "bm25", "top_k": 3},
            "generation": {"backend": "extractive"},
            "abstention": {"threshold": None},
        }
    )
    pipeline = RAGPipeline.from_config(cfg, docs=mini_docs)

    with TestClient(create_app(pipeline=pipeline)) as client:
        vacation = client.post("/ask", json={"question": "Сколько календарных дней длится ежегодный отпуск?"})
        remote = client.post(
            "/ask", json={"question": "Сколько дней в неделю можно работать удалённо?", "top_k": 2}
        )
        health = client.get("/health").json()
        metrics = client.get("/metrics").text

    assert vacation.status_code == 200
    body = AskResponse.model_validate(vacation.json()).model_dump()  # ответ соответствует схеме
    assert "28 календарных дней" in body["answer"]
    assert body["abstained"] is False
    assert body["contexts"][0]["doc_id"] == "hr-vacation"
    assert body["citations"], "extractive-генератор ставит ссылки [n]"
    assert {c["chunk_id"] for c in body["citations"]} <= {c["chunk_id"] for c in body["contexts"]}

    remote_body = remote.json()
    assert len(remote_body["contexts"]) == 2
    # Во фрагменте hr-remote-2025 есть «примечание для ИИ-ассистента» — guardrails его пометили.
    assert "prompt_injection:hr-remote-2025::000" in remote_body["flags"]
    assert metric_value(metrics, "kbrag_injection_flags_total") == 1

    assert health["n_docs"] == 4
    assert health["n_chunks"] == len(pipeline.chunks)
    assert health["retrieval_mode"] == "bm25"
    assert health["generator"] == "extractive"
    assert health["reranker"] is None
    assert health["top_k"] == 3 and health["score_source"] == "bm25"
