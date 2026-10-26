"""Подготовка данных: чтение корпуса (``loader``) и нарезка на фрагменты (``chunking``).

Типичное использование::

    docs = load_corpus("data/corpus")
    chunks = chunk_corpus(docs, make_chunker(cfg.chunking))
"""

from kbrag.ingest.chunking import Chunker, FixedSizeChunker, RecursiveChunker, chunk_corpus, make_chunker
from kbrag.ingest.loader import load_corpus, load_document, parse_front_matter

__all__ = [
    "Chunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "chunk_corpus",
    "load_corpus",
    "load_document",
    "make_chunker",
    "parse_front_matter",
]
