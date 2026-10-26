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
# Относительная отсечка: второе и третье предложения нужны, только если они почти так же
# хороши, как лучшее (score ≥ половины лучшего). Без неё ответ всегда добирал max_sentences
# предложений — даже совпавших с вопросом одним словом — и смешивал чужие документы.
# Константа выбрана из общих соображений, а не подбором по test.
_RELATIVE_CUTOFF = 0.5

_LINE_RE = re.compile(r"[^\n]+")
_HEADING_RE = re.compile(r"^\s*#{1,6}(?:\s|$)")
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
# Номер раздела в заголовке («3.», «4.2») смысла не несёт: «Что запрещено», а не «3. Что запрещено».
_HEADING_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s*")
# Разделитель пути разделов в Chunk.section (см. ingest.chunking._SectionIndex).
_SECTION_SEP = " > "
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
    """Выбирает до ``max_sentences`` предложений, больше всего пересекающихся с вопросом.

    Score предложения — число общих с вопросом стемов плюс маленький бонус за ранг фрагмента.
    В ответ идут лучшие предложения, но не слабее половины лучшего (``_RELATIVE_CUTOFF``).

    Известные ограничения бейзлайна (их и закрывает LLM-режим): слова не взвешены по редкости
    («дней» весит столько же, сколько «календарных»), конфликт действующей и архивной редакции
    не разрешается, а предложение с местоимением («Он регистрирует…») выводится без антецедента.
    """

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
        min_score = _RELATIVE_CUTOFF * candidates[0].score
        chosen: list[_Candidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.score < min_score:  # дальше только слабые: список отсортирован
                break
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
            chunk = scored.chunk
            for position, sentence in _answer_sentences(chunk.text, chunk.section, chunk.title):
                if len(tokenize(sentence)) < _MIN_SENTENCE_TOKENS:
                    continue
                overlap = len(question_stems & set(analyze(sentence)))
                if overlap < self.min_overlap:
                    continue
                candidates.append(_Candidate(overlap + _RANK_BONUS / rank, rank, position, sentence))
        return candidates


def _answer_sentences(text: str, section: str = "", title: str = "") -> list[tuple[int, str]]:
    """Предложения фрагмента, пригодные для ответа: пары (смещение, очищенный текст).

    Три правила, и у каждого своя причина.

    1. Строки-заголовки ``#`` пропускаем. Отсекаем их по строкам *до* деления на предложения:
       иначе «## 2. Порядок оформления отпуска» распалось бы на «## 2.» и «Порядок оформления
       отпуска», и вторая часть прошла бы как обычное предложение. Заголовок — не ответ, хотя
       слова с вопросом у него совпадают почти всегда.

    2. Пункт списка — не самостоятельное предложение, а продолжение вводной фразы. Пункт
       «- хранить секреты в репозиториях GitLab;» из раздела «## 3. Что запрещено», показанный
       отдельно, читается как совет, то есть *противоположно* источнику. Поэтому пункт
       склеиваем с его контекстом: с вводной фразой на «:» прямо перед списком, а если её нет —
       с заголовком раздела («Что запрещено: хранить секреты…»). С этим контекстом пункт
       и оценивается (слова вводной — часть его смысла), и попадает в ответ. Заголовок берём
       из последней строки ``#`` внутри фрагмента, а если фрагмент начался посреди раздела —
       из ``Chunk.section``.

    3. Предложение на «:» («Если секрет попал в репозиторий или чат:») — вводная, а не ответ:
       само по себе кандидатом не бывает, зато становится контекстом следующих пунктов.
    """
    sentences: list[tuple[int, str]] = []
    heading = _heading_title(section.split(_SECTION_SEP)[-1], title)
    lead_in: str | None = None  # вводная фраза на «:» перед текущим списком
    for line in _LINE_RE.finditer(text):
        raw = line.group()
        if _HEADING_RE.match(raw):
            heading, lead_in = _heading_title(raw.strip().lstrip("#"), title), None
            continue

        # Маркер пункта («- », «1. ») снимаем ДО деления на предложения: иначе «1. Отзовите
        # секрет» распалось бы на «1.» и «Отзовите секрет», и контекст достался бы не тому куску.
        marker = _LIST_MARKER_RE.match(raw)
        body_start = marker.end() if marker else 0
        body = raw[body_start:]
        context = (lead_in or heading) if marker else ""

        last_sentence = ""
        for i, (start, end) in enumerate(split_sentences(body)):
            last_sentence = clean_markdown(body[start:end])
            if last_sentence.endswith(":"):
                continue
            # Контекст нужен только первому предложению пункта: именно оно начинается с обрывка фразы.
            sentence = f"{context}: {last_sentence}" if context and i == 0 else last_sentence
            sentences.append((line.start() + body_start + start, sentence))

        if not marker:
            # Обычная строка: если она кончается на «:», это вводная к следующему списку.
            lead_in = last_sentence.rstrip(":").strip() if last_sentence.endswith(":") else None
    return sentences


def _heading_title(heading: str, doc_title: str = "") -> str:
    """«## 3. Что запрещено» → «Что запрещено». Пустая строка — контекста нет.

    Название самого документа контекстом не считаем: оно ничего не добавляет к смыслу пункта.
    """
    cleaned = _HEADING_NUMBER_RE.sub("", clean_markdown(heading.strip()))
    if not any(char.isalnum() for char in cleaned) or normalize(cleaned) == normalize(doc_title):
        return ""
    return cleaned
