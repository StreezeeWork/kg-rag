"""Офлайн-генератор без LLM: собирает ответ из самых подходящих предложений контекста.

Зачем он, если есть LLM: проект должен запускаться и оцениваться без GPU, ключей и сети
(в CI, на ноутбуке стажёра). Extractive-ответ — честный бейзлайн: он ничего не
выдумывает (каждое предложение дословно из документа) и всегда снабжён ссылкой [n].
Если LLM не обгоняет этот бейзлайн по метрикам — вопрос к промпту или модели.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kbrag.generation.base import GenerationResult, Generator
from kbrag.generation.prompts import ABSTAIN_TEXT
from kbrag.text import analyze, normalize, split_sentences, tokenize
from kbrag.types import ScoredChunk

# Предложение короче трёх слов («Срок.», «См. ниже») ответом быть не может.
_MIN_SENTENCE_TOKENS = 3
# Бонус за ранг фрагмента меньше единицы: он лишь разрешает ничьи в пользу
# более релевантного фрагмента и никогда не перевешивает лишнее совпавшее слово.
_RANK_BONUS = 0.1

_LINE_RE = re.compile(r"[^\n]+")
_HEADING_RE = re.compile(r"^\s*#{1,6}(?:\s|$)")
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_EMPHASIS_RE = re.compile(r"\*\*|__|`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_SPACES_RE = re.compile(r"\s+")


def clean_markdown(sentence: str) -> str:
    """Убирает разметку, которая в ответе выглядит мусором: маркеры списков, ``|``, ``**``."""
    text = _LIST_MARKER_RE.sub("", sentence)
    text = _LINK_RE.sub(r"\1", text)
    text = _EMPHASIS_RE.sub("", text)
    if "|" in text:
        # Строка таблицы «| Отпуск | 28 дней |» → «Отпуск — 28 дней».
        cells = [cell.strip() for cell in text.split("|")]
        text = " — ".join(cell for cell in cells if cell)
    return _SPACES_RE.sub(" ", text).strip()


@dataclass(frozen=True)
class _Candidate:
    score: float
    rank: int  # 1-based номер фрагмента в контексте — он же номер ссылки [n]
    position: int  # смещение предложения во фрагменте, для стабильной сортировки
    text: str


class ExtractiveGenerator(Generator):
    """Выбирает до ``max_sentences`` предложений, больше всего пересекающихся с вопросом."""

    name = "extractive"

    def __init__(self, max_sentences: int = 3, min_overlap: int = 1) -> None:
        if max_sentences < 1:
            raise ValueError("max_sentences должен быть ≥ 1")
        if min_overlap < 1:
            raise ValueError("min_overlap должен быть ≥ 1")
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    def generate(self, question: str, contexts: list[ScoredChunk]) -> GenerationResult:
        question_stems = set(analyze(question))
        candidates = self._candidates(question_stems, contexts) if question_stems else []
        if not candidates:
            return GenerationResult(text=ABSTAIN_TEXT, abstained=True)

        # Детерминизм: при равном score раньше идёт более релевантный фрагмент и более раннее предложение.
        candidates.sort(key=lambda c: (-c.score, c.rank, c.position))
        chosen: list[_Candidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = normalize(candidate.text)
            if key in seen:  # одно и то же предложение из перекрывающихся фрагментов
                continue
            seen.add(key)
            chosen.append(candidate)
            if len(chosen) == self.max_sentences:
                break

        answer = " ".join(f"{c.text} [{c.rank}]" for c in chosen)
        return GenerationResult(text=answer, abstained=False, raw=answer)

    def _candidates(self, question_stems: set[str], contexts: list[ScoredChunk]) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for rank, scored in enumerate(contexts, start=1):
            for position, sentence in _sentences_without_headings(scored.chunk.text):
                cleaned = clean_markdown(sentence)
                if len(tokenize(cleaned)) < _MIN_SENTENCE_TOKENS:
                    continue
                overlap = len(question_stems & set(analyze(cleaned)))
                if overlap < self.min_overlap:
                    continue
                candidates.append(_Candidate(overlap + _RANK_BONUS / rank, rank, position, cleaned))
        return candidates


def _sentences_without_headings(text: str) -> list[tuple[int, str]]:
    """Предложения фрагмента (со смещениями), кроме строк-заголовков ``#``.

    Заголовки отсекаем по строкам *до* деления на предложения: иначе «## 2. Порядок
    оформления отпуска» распалось бы на «## 2.» и «Порядок оформления отпуска», и вторая
    часть прошла бы как обычное предложение. Заголовок — не ответ, хотя слова с вопросом
    у него совпадают почти всегда.
    """
    sentences: list[tuple[int, str]] = []
    for line in _LINE_RE.finditer(text):
        if _HEADING_RE.match(line.group()):
            continue
        for start, end in split_sentences(line.group()):
            sentences.append((line.start() + start, line.group()[start:end]))
    return sentences
