"""Генерация ответа: промпты, защита от инъекций, LLM-клиент, генераторы и ссылки."""

from __future__ import annotations

from kbrag.config import GenerationConfig
from kbrag.generation.base import GenerationResult, Generator
from kbrag.generation.citations import parse_citations
from kbrag.generation.extractive import ExtractiveGenerator
from kbrag.generation.guardrails import check_question, detect_injection, sanitize_contexts
from kbrag.generation.llm import LLMError, LLMResponse, OpenAICompatibleClient
from kbrag.generation.llm_generator import LLMGenerator
from kbrag.generation.prompts import (
    ABSTAIN_TEXT,
    SYSTEM_PROMPT,
    build_messages,
    format_context,
    is_abstention,
    select_within_budget,
)


def make_generator(cfg: GenerationConfig) -> Generator:
    """Создаёт генератор по ``cfg.backend``: ``extractive`` (офлайн) или ``openai`` (LLM по API)."""
    if cfg.backend == "extractive":
        return ExtractiveGenerator(max_sentences=cfg.max_sentences)
    if cfg.backend == "openai":
        return LLMGenerator(
            OpenAICompatibleClient.from_config(cfg),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            max_context_chars=cfg.max_context_chars,
        )
    raise ValueError(f"Неизвестный backend генерации: {cfg.backend!r}")


__all__ = [
    "ABSTAIN_TEXT",
    "SYSTEM_PROMPT",
    "ExtractiveGenerator",
    "GenerationResult",
    "Generator",
    "LLMError",
    "LLMGenerator",
    "LLMResponse",
    "OpenAICompatibleClient",
    "build_messages",
    "check_question",
    "detect_injection",
    "format_context",
    "is_abstention",
    "make_generator",
    "parse_citations",
    "sanitize_contexts",
    "select_within_budget",
]
