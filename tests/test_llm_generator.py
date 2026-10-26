"""Тесты LLM-генератора (с фейковым клиентом) и фабрики make_generator."""

from __future__ import annotations

import re

import httpx
import pytest

from kbrag.config import GenerationConfig
from kbrag.generation import (
    ABSTAIN_TEXT,
    ExtractiveGenerator,
    LLMError,
    LLMGenerator,
    LLMResponse,
    OpenAICompatibleClient,
    build_messages,
    make_generator,
)
from kbrag.types import Chunk, Document, ScoredChunk

_PARAGRAPH_RE = re.compile(r"[^\n]+(?:\n[^\n]+)*")


def make_context(docs: list[Document], doc_id: str, needle: str) -> ScoredChunk:
    """Фрагмент = абзац мини-документа, содержащий ``needle`` (чанкер здесь не нужен)."""
    doc = next(d for d in docs if d.doc_id == doc_id)
    for i, match in enumerate(_PARAGRAPH_RE.finditer(doc.text)):
        if needle in match.group():
            chunk = Chunk(
                chunk_id=f"{doc_id}::{i:03d}", doc_id=doc_id, text=match.group(),
                start=match.start(), end=match.end(), title=doc.title, metadata=dict(doc.metadata),
            )
            return ScoredChunk(chunk=chunk, score=1.0, source="bm25")
    raise AssertionError(f"абзац с {needle!r} не найден в {doc_id}")


class FakeClient:
    """Подменяет OpenAICompatibleClient: возвращает заготовленный текст и запоминает вызовы."""

    def __init__(self, reply: str = "Отпуск длится 28 календарных дней [1].", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls: list[dict] = []

    def chat(self, messages, temperature=0.0, max_tokens=512, response_format=None) -> LLMResponse:
        self.calls.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        if self.error is not None:
            raise self.error
        return LLMResponse(text=self.reply, usage={"prompt_tokens": 50, "completion_tokens": 9}, latency_ms=12.0, model="fake")


@pytest.fixture()
def contexts(mini_docs: list[Document]) -> list[ScoredChunk]:
    return [
        make_context(mini_docs, "hr-vacation", "28 календарных"),
        make_context(mini_docs, "hr-vacation", "Заявление на отпуск"),
    ]


def test_sends_messages_from_build_messages(contexts: list[ScoredChunk]) -> None:
    client = FakeClient()
    generator = LLMGenerator(client, temperature=0.3, max_tokens=128, max_context_chars=400)
    question = "Сколько длится отпуск?"

    generator.generate(question, contexts)

    (call,) = client.calls
    assert call["messages"] == build_messages(question, contexts, max_chars=400)
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 128


def test_regular_answer_is_returned_with_usage(contexts: list[ScoredChunk]) -> None:
    client = FakeClient(reply="  Отпуск длится 28 календарных дней [1].\n")

    result = LLMGenerator(client).generate("Сколько длится отпуск?", contexts)

    assert result.text == "Отпуск длится 28 календарных дней [1]."
    assert not result.abstained
    assert result.usage == {"prompt_tokens": 50, "completion_tokens": 9}
    assert result.raw == "  Отпуск длится 28 календарных дней [1].\n"  # сырой ответ — для отладки


@pytest.mark.parametrize(
    "reply",
    [
        ABSTAIN_TEXT,
        "в базе знаний нет ответа на этот вопрос",
        "В базе знаний нет ответа на этот вопрос. [1]",
        "К сожалению, В БАЗЕ ЗНАНИЙ НЕТ ОТВЕТА НА ЭТОТ ВОПРОС!",
    ],
)
def test_abstention_is_recognized(contexts: list[ScoredChunk], reply: str) -> None:
    result = LLMGenerator(FakeClient(reply=reply)).generate("Сколько стоит обед?", contexts)
    assert result.abstained


def test_similar_but_different_answer_is_not_abstention(contexts: list[ScoredChunk]) -> None:
    result = LLMGenerator(FakeClient(reply="В базе знаний есть ответ: 28 дней [1].")).generate("?", contexts)
    assert not result.abstained


def test_no_contexts_means_abstention_without_calling_llm() -> None:
    client = FakeClient()
    result = LLMGenerator(client).generate("Сколько длится отпуск?", [])

    assert result.abstained and result.text == ABSTAIN_TEXT
    assert client.calls == []


def test_llm_errors_are_propagated(contexts: list[ScoredChunk]) -> None:
    generator = LLMGenerator(FakeClient(error=LLMError("сервер недоступен")))
    with pytest.raises(LLMError):
        generator.generate("Сколько длится отпуск?", contexts)


def test_works_end_to_end_with_real_client_and_mock_transport(contexts: list[ScoredChunk]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": ABSTAIN_TEXT}}], "usage": {}})

    client = OpenAICompatibleClient("http://llm.test/v1", "m", transport=httpx.MockTransport(handler))
    result = LLMGenerator(client).generate("Сколько стоит обед?", contexts)

    assert result.abstained
    assert LLMGenerator.name == "llm"


# --- make_generator --------------------------------------------------------------------------


def test_make_generator_extractive() -> None:
    generator = make_generator(GenerationConfig(backend="extractive", max_sentences=2))
    assert isinstance(generator, ExtractiveGenerator)
    assert generator.max_sentences == 2


def test_make_generator_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KBRAG_TEST_KEY", "k-123")
    cfg = GenerationConfig(
        backend="openai", base_url="http://gpu:8000/v1", model="qwen", api_key_env="KBRAG_TEST_KEY",
        temperature=0.1, max_tokens=256, max_context_chars=3000,
    )

    generator = make_generator(cfg)

    assert isinstance(generator, LLMGenerator)
    assert (generator.temperature, generator.max_tokens, generator.max_context_chars) == (0.1, 256, 3000)
    assert isinstance(generator.client, OpenAICompatibleClient)
    assert generator.client.model == "qwen"
    assert generator.client.api_key == "k-123"
