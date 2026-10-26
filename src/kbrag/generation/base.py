"""Общий интерфейс генераторов ответа.

Пайплайну всё равно, кто пишет ответ — LLM или простой офлайн-алгоритм: он работает
с абстрактным ``Generator``. Поэтому генераторы легко подменять в конфиге и в тестах.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from kbrag.types import ScoredChunk


@dataclass
class GenerationResult:
    """Результат генерации до разбора ссылок.

    ``raw`` хранит «сырой» ответ модели: если текст как-то постобработан, исходник
    всё равно доступен для отладки. ``flags`` — служебные пометки генератора (например,
    ``answer_truncated``); пайплайн добавляет их к ``Answer.flags``.
    """

    text: str
    abstained: bool
    usage: dict[str, int] = field(default_factory=dict)
    raw: str | None = None
    flags: list[str] = field(default_factory=list)


class Generator(ABC):
    """Пишет ответ на вопрос по найденным фрагментам (контекстам)."""

    name: str = "base"

    @abstractmethod
    def generate(self, question: str, contexts: list[ScoredChunk]) -> GenerationResult:
        """Возвращает ответ со ссылками ``[n]`` на 1-based номера в ``contexts``."""
