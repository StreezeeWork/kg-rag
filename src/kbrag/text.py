"""Обработка текста: нормализация, токенизация, стемминг, разбиение на предложения.

Один и тот же анализатор используется в BM25, extractive-генераторе и метриках,
чтобы «слово» везде означало одно и то же.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from functools import lru_cache

import snowballstemmer

_WS_RE = re.compile(r"\s+")
# Слово: буквы/цифры, допускаем внутренние дефисы и точки ("VPN-шлюз", "help.nimbus.local", "2.5").
_WORD_RE = re.compile(r"[0-9a-zа-я]+(?:[-.][0-9a-zа-я]+)*")
_CYRILLIC_RE = re.compile(r"[а-я]")
_LATIN_RE = re.compile(r"[a-z]")
# Сокращения, после которых точка — не конец предложения, хотя дальше заглавная буква или цифра:
# «ул. Облачная, д. 7», «г. Казань», «см. Политику паролей». Без этого extractive-ответ обрывался
# на «…во дворе офиса на ул.». Список короткий намеренно: «руб.» или «мин.» часто и правда завершают фразу.
_ABBREVIATIONS = ("ул", "д", "г", "см", "т", "стр")
_NOT_AFTER_ABBREVIATION = "".join(rf"(?<!\b{abbr}\.)" for abbr in _ABBREVIATIONS)
# Граница предложения: . ! ? … перед пробелом и заглавной буквой/цифрой/кавычкой, либо перевод строки.
_SENT_END_RE = re.compile(rf"(?<=[.!?…]){_NOT_AFTER_ABBREVIATION}\s+(?=[A-ZА-ЯЁ0-9«\"(])|\n+")

# Стоп-слова записаны одной строкой, а не списком литералов: так их проще читать и дополнять
# (поэтому правило ruff SIM905 здесь отключено). Формы с «ё» (её, ещё, неё) срабатывать не могут —
# analyze сравнивает уже нормализованные токены (ё→е), — но их пары «ее», «еще», «нее» тоже в списке.
STOPWORDS: frozenset[str] = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее её мне
    было вот от меня еще ещё нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него
    до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы
    тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого
    какой совсем ним здесь этом один почти мой тем чтобы нее неё сейчас были куда зачем всех никогда
    можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много
    разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда
    конечно всю между это также который которая которые которых которой котором является ли
    the a an of to in on for and or is are be by with as at from that this it
    """.split()  # noqa: SIM905
)

_RU_STEMMER = snowballstemmer.stemmer("russian")
_EN_STEMMER = snowballstemmer.stemmer("english")
# Объект snowballstemmer хранит состояние разбора (current, cursor) в себе, а /ask в API
# выполняется в пуле потоков: два одновременных вызова портят друг другу слово (чужой стем
# или IndexError). lru_cache сам потокобезопасен, но тело функции не защищает. Блокировка
# берётся только при промахе кэша, поэтому после прогрева почти ничего не стоит.
_STEMMER_LOCK = threading.Lock()


def normalize(text: str) -> str:
    """Нижний регистр, ё→е, Unicode NFKC, схлопывание пробелов."""
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Разбивает текст на нормализованные токены (без стемминга)."""
    return _WORD_RE.findall(normalize(text))


@lru_cache(maxsize=200_000)
def stem(token: str) -> str:
    """Стемминг: русские слова — русским стеммером, латиница — английским, числа как есть."""
    if _CYRILLIC_RE.search(token):
        with _STEMMER_LOCK:
            return _RU_STEMMER.stemWord(token)
    if _LATIN_RE.search(token) and token.isalpha():
        with _STEMMER_LOCK:
            return _EN_STEMMER.stemWord(token)
    return token


def analyze(text: str, remove_stopwords: bool = True) -> list[str]:
    """Полный анализатор: токенизация → удаление стоп-слов → стемминг."""
    tokens = tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [stem(t) for t in tokens]


def word_spans(text: str) -> list[tuple[int, int]]:
    """Смещения «слов» (последовательностей непробельных символов) в исходном тексте."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Смещения предложений в исходном тексте (без ведущих/хвостовых пробелов).

    Перевод строки тоже считается границей: в markdown это пункты списков и строки таблиц.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    for match in _SENT_END_RE.finditer(text):
        _append_trimmed(text, pos, match.start(), spans)
        pos = match.end()
    _append_trimmed(text, pos, len(text), spans)
    return spans


def _append_trimmed(text: str, start: int, end: int, spans: list[tuple[int, int]]) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end > start:
        spans.append((start, end))


def find_span(haystack: str, needle: str) -> tuple[int, int] | None:
    """Ищет ``needle`` в ``haystack`` с точностью до пробельных символов.

    Возвращает смещения в исходном ``haystack`` или None. Нужна, чтобы цитата из
    эталонного набора находилась в документе, даже если в ней иначе расставлены переносы строк.
    """
    parts = needle.split()
    if not parts:
        return None
    pattern = r"\s+".join(re.escape(p) for p in parts)
    match = re.search(pattern, haystack)
    return (match.start(), match.end()) if match else None
