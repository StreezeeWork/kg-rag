"""Генератор на LLM: промпт из ``prompts`` → OpenAI-совместимый API → ``GenerationResult``."""

from __future__ import annotations

from kbrag.generation.base import GenerationResult, Generator
from kbrag.generation.llm import OpenAICompatibleClient
from kbrag.generation.prompts import ABSTAIN_TEXT, build_messages, is_abstention
from kbrag.types import ScoredChunk


class LLMGenerator(Generator):
    """Отвечает с помощью LLM строго по переданным фрагментам.

    Ошибки API (``LLMError``) не глушим: лучше честная ошибка в ответе сервиса,
    чем «отказ», неотличимый от настоящего отсутствия ответа в базе знаний.
    """

    name = "llm"

    def __init__(
        self,
        client: OpenAICompatibleClient,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_context_chars: int = 6000,
    ) -> None:
        self.client = client
        # temperature = 0 — ответы воспроизводимы, а для фактических вопросов «креатив» не нужен.
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_context_chars = max_context_chars

    def generate(self, question: str, contexts: list[ScoredChunk]) -> GenerationResult:
        if not contexts:
            # Без контекста правильный ответ один — отказ; платить за вызов модели незачем.
            return GenerationResult(text=ABSTAIN_TEXT, abstained=True)

        messages = build_messages(question, contexts, max_chars=self.max_context_chars)
        response = self.client.chat(messages, temperature=self.temperature, max_tokens=self.max_tokens)
        text = response.text.strip()
        return GenerationResult(
            text=text,
            abstained=is_abstention(text),
            usage=dict(response.usage),
            raw=response.text,
        )
