"""Эмбеддеры: превращают текст в вектор, чтобы искать «по смыслу», а не по точным словам.

Два варианта с одним интерфейсом ``Embedder``:

- ``LSAEmbedder`` — классический латентно-семантический анализ (TF-IDF по символьным
  n-граммам + SVD). Работает офлайн, без скачивания моделей, за секунды; символьные
  n-граммы сглаживают словоформы и опечатки. Это «плотный» baseline для CI и ноутбука.
- ``SentenceTransformerEmbedder`` — нейросетевая модель (например, multilingual-e5).
  Качественнее, но требует ``pip install -e .[neural]`` и скачивания весов.

Все векторы L2-нормированы, поэтому косинусная близость = скалярное произведение.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from kbrag.config import DenseConfig
from kbrag.text import normalize

logger = logging.getLogger(__name__)

NEURAL_INSTALL_HINT = "pip install -e .[neural]"


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Нормирует строки матрицы до единичной длины (нулевые строки остаются нулевыми).

    Нулевой вектор получается у запроса, в котором нет ни одной известной n-граммы;
    делить его на ноль нельзя — оставляем нулём, и он ни на что не будет похож.
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def resolve_device(device: str) -> str:
    """``"auto"`` → ``"cuda"``, если доступна видеокарта, иначе ``"cpu"``; остальное — как есть."""
    if device != "auto":
        return device
    try:
        import torch  # ленивый импорт: torch тяжёлый и нужен только нейросетевым моделям
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def neural_available() -> bool:
    """Установлен ли ``sentence_transformers`` (без его импорта — это занимает секунды)."""
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):
        # ValueError бывает, если модуль подменён в sys.modules объектом без __spec__ (в тестах).
        return sys.modules.get("sentence_transformers") is not None


class Embedder(ABC):
    """Интерфейс эмбеддера. Запросы и пассажи кодируются отдельно: у части моделей
    (e5) для них разные префиксы, и смешивать их нельзя."""

    name: str = "embedder"

    def fit(self, passages: list[str]) -> None:  # noqa: B027 — пустая реализация по умолчанию намеренна
        """Обучение на корпусе. Нейросетевым моделям не нужно, LSA — обязательно."""

    @abstractmethod
    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Матрица (len(texts), dim), float32, строки L2-нормированы."""

    @abstractmethod
    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Матрица (len(texts), dim), float32, строки L2-нормированы."""


class LSAEmbedder(Embedder):
    """TF-IDF по символьным n-граммам (3–5) + TruncatedSVD.

    SVD сжимает десятки тысяч n-грамм в ``dim`` «тем», и тексты с похожими n-граммами
    оказываются рядом, даже если общих слов целиком у них нет. ``seed`` фиксирует
    рандомизированный SVD: одинаковый корпус → одинаковые векторы при каждом запуске.
    """

    name = "lsa"

    def __init__(self, dim: int = 256, seed: int = 0) -> None:
        if dim < 1:
            raise ValueError(f"dim должен быть >= 1, получено {dim}")
        self.dim = dim
        self.seed = seed
        self.n_components: int | None = None  # фактическая размерность после fit
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None

    def fit(self, passages: list[str]) -> None:
        if not passages:
            raise ValueError("LSAEmbedder.fit: нужен хотя бы один текст")
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",  # n-граммы внутри слов (с пробелами по краям слова)
            ngram_range=(3, 5),
            sublinear_tf=True,  # 1 + log(tf): частые n-граммы не доминируют
            preprocessor=normalize,  # тот же нижний регистр и ё→е, что и в остальном проекте
            dtype=np.float32,
        )
        tfidf = vectorizer.fit_transform(passages)
        n_docs, n_features = tfidf.shape
        # SVD не может дать компонент больше, чем (число документов − 1) и (число признаков − 1):
        # на крошечном корпусе честно уменьшаем размерность вместо падения.
        n_components = min(self.dim, n_features - 1, n_docs - 1)
        if n_components >= 1:
            self._svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
            self._svd.fit(tfidf)
            self.n_components = n_components
        else:
            # Один документ: сжимать нечего — используем сам TF-IDF-вектор (это тоже валидный эмбеддинг).
            logger.warning("LSAEmbedder: корпус слишком мал для SVD (%d док.), используется TF-IDF", n_docs)
            self._svd = None
            self.n_components = n_features
        self._vectorizer = vectorizer

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._vectorizer is None:
            raise RuntimeError("LSAEmbedder не обучен: сначала вызовите fit(passages)")
        if not texts:
            return np.zeros((0, self.n_components or 0), dtype=np.float32)
        tfidf = self._vectorizer.transform(texts)
        vectors = self._svd.transform(tfidf) if self._svd is not None else tfidf.toarray()
        return l2_normalize(vectors)


class SentenceTransformerEmbedder(Embedder):
    """Нейросетевой эмбеддер на ``sentence_transformers.SentenceTransformer``.

    Эмбеддинги корпуса кэшируются на диск: пересчёт всех пассажей на CPU — минуты,
    а в серии экспериментов один и тот же корпус кодируется много раз.
    """

    name = "sentence-transformers"

    def __init__(
        self,
        model_name: str,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
        batch_size: int = 32,
        device: str = "auto",
        cache_dir: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                f"Для нейросетевых эмбеддингов нужны дополнительные зависимости: {NEURAL_INSTALL_HINT}"
            ) from exc

        self.model_name = model_name
        # Модели семейства e5 обучались с префиксами "query: " / "passage: "; без них качество падает.
        is_e5 = "e5" in model_name.lower()
        default_query, default_passage = ("query: ", "passage: ") if is_e5 else ("", "")
        self.query_prefix = default_query if query_prefix is None else query_prefix
        self.passage_prefix = default_passage if passage_prefix is None else passage_prefix
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self.cache_dir = cache_dir
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        # Запросы не кэшируем: они каждый раз новые, а кодируются быстро.
        return self._encode([self.query_prefix + t for t in texts])

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        cache_path = self.cache_path(texts)
        if cache_path is not None and cache_path.exists():
            cached = self._load_cache(cache_path, expected_rows=len(texts))
            if cached is not None:
                logger.info("Эмбеддинги пассажей загружены из кэша %s", cache_path)
                return cached

        vectors = self._encode([self.passage_prefix + t for t in texts])
        if cache_path is not None:
            self._save_cache(cache_path, vectors)
        return vectors

    def cache_path(self, texts: list[str]) -> Path | None:
        """Путь к кэшу: ``{cache_dir}/emb/{sha1[:20]}.npy`` (коротко из-за MAX_PATH в Windows).

        В хэш входят модель, префикс и все тексты: изменился хоть один символ — новый файл,
        поэтому устаревший кэш никогда не подхватится.
        """
        if self.cache_dir is None:
            return None
        digest = hashlib.sha1()
        for part in (self.model_name, self.passage_prefix, *texts):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")  # разделитель: ["ab", "c"] и ["a", "bc"] дают разные хэши
        return Path(self.cache_dir) / "emb" / f"{digest.hexdigest()[:20]}.npy"

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Модель уже нормирует, но приводим к float32 и нормируем ещё раз — дёшево и надёжно.
        return l2_normalize(vectors)

    @staticmethod
    def _load_cache(path: Path, expected_rows: int) -> np.ndarray | None:
        try:
            vectors = np.load(path)
        except (OSError, ValueError) as exc:
            logger.warning("Повреждённый кэш эмбеддингов %s (%s) — пересчитываем", path, exc)
            return None
        if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
            logger.warning("Кэш эмбеддингов %s не совпадает по размеру — пересчитываем", path)
            return None
        return vectors.astype(np.float32, copy=False)

    @staticmethod
    def _save_cache(path: Path, vectors: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Пишем во временный файл и атомарно переименовываем: если процесс упадёт посреди
        # записи, на диске не останется «половинки», которую потом примут за кэш.
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "wb") as fh:
            np.save(fh, vectors)
        os.replace(tmp_path, path)


def make_embedder(cfg: DenseConfig, cache_dir: str | None = None) -> Embedder:
    """Создаёт эмбеддер по секции ``retrieval.dense`` конфига."""
    if cfg.embedder == "lsa":
        return LSAEmbedder(dim=cfg.lsa_dim)
    if cfg.embedder == "sentence-transformers":
        return SentenceTransformerEmbedder(
            model_name=cfg.model,
            query_prefix=cfg.query_prefix,
            passage_prefix=cfg.passage_prefix,
            batch_size=cfg.batch_size,
            device=cfg.device,
            cache_dir=cache_dir,
        )
    raise ValueError(f"Неизвестный эмбеддер: {cfg.embedder!r}")
