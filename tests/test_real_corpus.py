"""Проверки нарезки на настоящем корпусе ``data/corpus`` и эталоне ``data/eval/gold.jsonl``.

Остальные тесты нарезки работают на мини-корпусе из ``conftest.py`` и синтетике: быстро,
но регрессию в самих данных (правку документа, после которой фрагменты ломают инварианты или
цитата из эталона перестаёт попадать во фрагмент) они не поймают. Здесь — те же проверки на
реальных данных для каждой нарезки, которая встречается в ``configs/`` (боевые конфиги и сетка
экспериментов). Это несколько секунд; без папки ``data`` тесты пропускаются.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from chunk_helpers import assert_chunk_invariants

from kbrag.config import ChunkingConfig, load_config
from kbrag.eval.dataset import GoldItem, evidence_spans, is_relevant, load_gold, validate_gold
from kbrag.eval.runner import load_experiments
from kbrag.ingest import chunk_corpus, load_corpus, make_chunker
from kbrag.types import Document

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus"
GOLD_PATH = ROOT / "data" / "eval" / "gold.jsonl"
CONFIGS_DIR = ROOT / "configs"

pytestmark = pytest.mark.skipif(
    not CORPUS_DIR.is_dir() or not GOLD_PATH.is_file(), reason="нет data/corpus или data/eval/gold.jsonl"
)


def _chunking_setups() -> list[ChunkingConfig]:
    """Все различающиеся настройки нарезки из боевых конфигов и сетки экспериментов."""
    configs = [
        load_config(path) for path in sorted(CONFIGS_DIR.glob("*.yaml")) if path.name != "experiments.yaml"
    ]
    experiments_path = CONFIGS_DIR / "experiments.yaml"
    if experiments_path.is_file():
        base_path, specs = load_experiments(experiments_path)
        if base_path is not None and not base_path.is_absolute():
            base_path = ROOT / base_path  # относительный base считается от корня проекта
        configs += [load_config(base_path, spec.overrides) for spec in specs]
    # with_header на нарезку не влияет: фрагменты одинаковые, отличается только текст для индекса.
    unique = {
        (c.chunking.strategy, c.chunking.chunk_size, c.chunking.chunk_overlap): c.chunking for c in configs
    }
    return list(unique.values())


def _setup_id(cfg: ChunkingConfig) -> str:
    return f"{cfg.strategy}-{cfg.chunk_size}" + (
        f"-overlap{cfg.chunk_overlap}" if cfg.strategy == "fixed" else ""
    )


@pytest.fixture(scope="module")
def docs() -> list[Document]:
    return load_corpus(CORPUS_DIR)


@pytest.fixture(scope="module")
def gold() -> list[GoldItem]:
    return load_gold(GOLD_PATH)


def test_gold_quotes_are_found_in_corpus(docs: list[Document], gold: list[GoldItem]) -> None:
    assert validate_gold(gold, docs) == []


@pytest.mark.parametrize("chunking", _chunking_setups() if CONFIGS_DIR.is_dir() else [], ids=_setup_id)
def test_real_corpus_chunking(chunking: ChunkingConfig, docs: list[Document], gold: list[GoldItem]) -> None:
    chunker = make_chunker(chunking)
    for doc in docs:
        assert_chunk_invariants(doc, chunker.split(doc), chunking.chunk_size)

    # Каждое доказательство из эталона должно быть «найдено» хотя бы одним фрагментом по тому же
    # правилу, что и в метриках (is_relevant). Иначе Recall для такого вопроса не может быть 1
    # при любом поиске — значит, сломана нарезка или разметка, а не поиск. Требовать, чтобы цитата
    # целиком лежала в одном фрагменте, нельзя: при мелкой нарезке она законно режется границей.
    chunks = chunk_corpus(docs, chunker)
    docs_by_id = {doc.doc_id: doc for doc in docs}
    uncovered = [
        item.qid
        for item in gold
        for span in evidence_spans(item, docs_by_id)
        if not any(is_relevant(chunk, span) for chunk in chunks if chunk.doc_id == span[0])
    ]
    assert uncovered == []
