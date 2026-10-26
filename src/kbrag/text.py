"""Обработка текста: нормализация, токенизация, стемминг, разбиение на предложения.

Один и тот же анализатор используется в BM25, extractive-генераторе и метриках,
чтобы «слово» везде означало одно и то же.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from functools import lru_cache
from typing import Literal, get_args

import snowballstemmer

_WS_RE = re.compile(r"\s+")
# Слово: буквы/цифры, допускаем внутренние дефисы и точки ("VPN-шлюз", "help.nimbus.local", "2.5").
_WORD_RE = re.compile(r"[0-9a-zа-я]+(?:[-.][0-9a-zа-я]+)*")
_CYRILLIC_RE = re.compile(r"[а-я]")
_LATIN_RE = re.compile(r"[a-z]")
_COMPOUND_SEP_RE = re.compile(r"[-.]")
# Как BM25 обращается с составными токенами («vpn-шлюз»): см. analyze(split_compounds=...).
SplitCompounds = Literal["none", "mixed", "all"]
SPLIT_COMPOUNDS_MODES: tuple[str, ...] = get_args(SplitCompounds)
# Сокращения, после которых точка — не конец предложения, хотя дальше заглавная буква или цифра:
# «ул. Облачная, д. 7», «г. Казань», «см. Политику паролей», «п. 3», «ст. 81», «рис. 2», «табл. 1».
# Без этого extractive-ответ обрывался на «…во дворе офиса на ул.». Список короткий намеренно:
# «руб.», «мин.» или «ч.» часто и правда завершают фразу. Обратная сторона — после «т. д.» и «г.»
# в конце предложения («…и т. д. Далее», «…в 2023 г. Новый абзац») граница не ставится:
# склеить два предложения безопаснее, чем оборвать фразу на сокращении.
_ABBREVIATIONS = ("ул", "д", "г", "см", "т", "стр", "п", "пп", "ст", "рис", "табл")
_NOT_AFTER_ABBREVIATION = "".join(rf"(?<!\b{abbr}\.)" for abbr in _ABBREVIATIONS)
# Граница предложения: . ! ? … перед пробелом и заглавной буквой/цифрой/кавычкой, либо перевод строки.
_SENT_END_RE = re.compile(rf"(?<=[.!?…]){_NOT_AFTER_ABBREVIATION}\s+(?=[A-ZА-ЯЁ0-9«\"(])|\n+")

# Стоп-слова — только служебные слова без собственного смысла: предлоги, союзы, частицы,
# местоимения, вспомогательные глаголы и местоимённые наречия (основа — список NLTK).
# Из списка NLTK намеренно убраны слова, которые в базе знаний и есть факт: числительные
# («один», «два», «три» — остальные числительные стоп-словами и не были), «раз», «нет»,
# слова разрешения и запрета («можно», «нельзя», «надо», «может») и сравнения («более»,
# «больше», «лучше»). Иначе «три дня» и «пять дней» обрабатывались по-разному, а для метрик
# «не более 6 часов» и «не менее 6 часов» были одним и тем же фактом. Составные предлоги
# «из-за» и «из-под» добавлены: токенизатор держит их целиком, и без этого они становились термом.
# Слова записаны одной строкой, а не списком литералов: так их проще читать и дополнять
# (поэтому правило ruff SIM905 здесь отключено). Формы с «ё» (её, ещё, неё) срабатывать не могут —
# analyze сравнивает уже нормализованные токены (ё→е), — но их пары «ее», «еще», «нее» тоже в списке.
STOPWORDS: frozenset[str] = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее её мне
    было вот от меня еще ещё о из ему теперь когда даже ну вдруг ли если уже или ни быть был него
    до вас нибудь опять уж вам ведь там потом себя ничего ей они тут где есть ней для мы
    тебя их чем была сам чтоб без будто чего тоже себе под будет ж тогда кто этот того потому этого
    какой совсем ним здесь этом почти мой тем чтобы нее неё сейчас были куда зачем всех никогда
    при наконец об другой хоть после над тот через эти нас про всего них какая много
    разве эту моя впрочем хорошо свою этой перед иногда чуть том такой им всегда
    конечно всю между это также который которая которые которых которой котором является
    из-за из-под
    the a an of to in on for and or is are be by with as at from that this it
    """.split()  # noqa: SIM905
)

_RU_STEMMER = snowballstemmer.stemmer("russian")
# Слова, которые Snowball режет неправильно, а в корпусе они важны. «Орбита» — название системы
# заявок: «в Орбите» стеммер принимает за глагол на «-ите» и даёт «орб», а «Орбита», «Орбиты»,
# «Орбиту» — «орбит». Без исключения вопрос «заявка в Орбите» не находил бы «систему Орбита».
_STEM_EXCEPTIONS: dict[str, str] = {"орбите": "орбит", "орбит": "орбит"}
_EN_STEMMER = snowballstemmer.stemmer("english")
# Объект snowballstemmer хранит состояние разбора (current, cursor) в себе, а /ask в API
# выполняется в пуле потоков: два одновременных вызова портят друг другу слово (чужой стем
# или IndexError). lru_cache сам потокобезопасен, но тело функции не защищает. Блокировка
# берётся только при промахе кэша, поэтому после прогрева почти ничего не стоит.
_STEMMER_LOCK = threading.Lock()


def normalize(text: str) -> str:
    """Нижний регистр, ё→е, Unicode NFKC, схлопывание пробелов.

    Знак «№» заменяется пробелом до NFKC: иначе NFKC превращает его в латинское «no»,
    и в тексте появляется слово, которого там не было («приказ № 14-П» → «приказ no 14-п»).
    """
    text = unicodedata.normalize("NFKC", text.replace("№", " ")).lower().replace("ё", "е")
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Разбивает текст на нормализованные токены (без стемминга)."""
    return _WORD_RE.findall(normalize(text))


@lru_cache(maxsize=200_000)
def stem(token: str) -> str:
    """Стемминг: русские слова — русским стеммером, латиница — английским, числа как есть.

    Для немногих слов, которые стеммер режет неправильно, основа задана вручную (``_STEM_EXCEPTIONS``).
    """
    if token in _STEM_EXCEPTIONS:
        return _STEM_EXCEPTIONS[token]
    if _CYRILLIC_RE.search(token):
        with _STEMMER_LOCK:
            return _RU_STEMMER.stemWord(token)
    if _LATIN_RE.search(token) and token.isalpha():
        with _STEMMER_LOCK:
            return _EN_STEMMER.stemWord(token)
    return token


def analyze(text: str, remove_stopwords: bool = True, split_compounds: SplitCompounds = "none") -> list[str]:
    """Полный анализатор: токенизация → (части составных токенов) → удаление стоп-слов → стемминг.

    ``split_compounds`` нужен только поиску (BM25); метрики и генератор работают с целыми токенами:

    * ``"none"`` — составной токен остаётся одним термом: «vpn-шлюз», «help.nimbus.local»;
    * ``"mixed"`` — для токенов, где латиница смешана с кириллицей, добавляются и части:
      «vpn-шлюз» → «vpn-шлюз», «vpn», «шлюз». Так вопрос про «VPN-шлюз» находит «шлюз VPN»;
    * ``"all"`` — части добавляются для любого токена с дефисом или точкой («нимбус.люди», «2.5»).

    Целый токен остаётся всегда, поэтому точный код «vpn-01» по-прежнему совпадает с самим собой.
    """
    if split_compounds not in SPLIT_COMPOUNDS_MODES:
        raise ValueError(
            f"split_compounds: ожидается одно из {SPLIT_COMPOUNDS_MODES}, получено {split_compounds!r}"
        )
    tokens = tokenize(text)
    if split_compounds != "none":
        tokens = _with_compound_parts(tokens, split_compounds)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [stem(t) for t in tokens]


def _with_compound_parts(tokens: list[str], mode: SplitCompounds) -> list[str]:
    """Добавляет после составного токена его части (по дефисам и точкам)."""
    result: list[str] = []
    for token in tokens:
        result.append(token)
        if not _COMPOUND_SEP_RE.search(token):
            continue
        if mode == "all" or (_CYRILLIC_RE.search(token) and _LATIN_RE.search(token)):
            result.extend(_COMPOUND_SEP_RE.split(token))
    return result


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
