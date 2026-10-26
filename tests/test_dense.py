"""Тесты плотного поиска и эмбеддеров (LSA, SentenceTransformer с фейковой моделью, кэш)."""

from __future__ import annotations

import logging
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from chunk_helpers import paragraph_chunks, plain_chunk
from sklearn.feature_extraction.text import TfidfVectorizer

from kbrag.config import DenseConfig
from kbrag.retrieval import (
    DenseRetriever,
    Embedder,
    LSAEmbedder,
    SentenceTransformerEmbedder,
    make_embedder,
    neural_available,
)
from kbrag.retrieval.embeddings import l2_normalize, resolve_device
from kbrag.types import Chunk, Document


class KeywordEmbedder(Embedder):
    """Прозрачный эмбеддер для тестов: ось вектора = число вхождений ключевого слова."""

    name = "keywords"
    AXES = ("отпуск", "vpn", "пароль")

    def __init__(self) -> None:
        self.fit_calls: list[list[str]] = []
        self.query_calls: list[list[str]] = []

    def fit(self, passages: list[str]) -> None:
        self.fit_calls.append(list(passages))

    def _vectors(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(
            np.array([[t.lower().count(axis) for axis in self.AXES] for t in texts], dtype=float)
        )

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        self.query_calls.append(list(texts))
        return self._vectors(texts)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._vectors(texts)


@pytest.fixture()
def mini_chunks(mini_docs: list[Document]) -> list[Chunk]:
    return paragraph_chunks(mini_docs)


@pytest.fixture()
def keyword_chunks() -> list[Chunk]:
    return [
        plain_chunk("a::000", "отпуск отпуск"),
        plain_chunk("b::000", "vpn"),
        plain_chunk("c::000", "отпуск vpn"),
    ]


# --- DenseRetriever с прозрачным эмбеддером ---


def test_dense_scores_are_cosine_similarities(keyword_chunks: list[Chunk]) -> None:
    retriever = DenseRetriever(KeywordEmbedder())
    retriever.index(keyword_chunks)
    results = retriever.search("отпуск", k=3)
    assert [r.chunk.chunk_id for r in results] == ["a::000", "c::000", "b::000"]
    assert [r.score for r in results] == pytest.approx([1.0, 1 / np.sqrt(2), 0.0], abs=1e-6)
    for item in results:
        assert item.source == "dense"
        assert item.details == {"dense": item.score}


def test_dense_respects_k(keyword_chunks: list[Chunk]) -> None:
    retriever = DenseRetriever(KeywordEmbedder())
    retriever.index(keyword_chunks)
    assert [r.chunk.chunk_id for r in retriever.search("vpn", k=1)] == ["b::000"]
    assert retriever.search("vpn", k=0) == []


def test_zero_query_vector_returns_empty(keyword_chunks: list[Chunk]) -> None:
    retriever = DenseRetriever(KeywordEmbedder())
    retriever.index(keyword_chunks)
    assert retriever.search("погода в городе", k=3) == []


def test_index_passes_header_texts_to_fit() -> None:
    chunk = plain_chunk("a::000", "отпуск", title="Отпуска")
    embedder = KeywordEmbedder()
    DenseRetriever(embedder, with_header=True).index([chunk])
    assert embedder.fit_calls == [["Отпуска\nотпуск"]]

    embedder = KeywordEmbedder()
    DenseRetriever(embedder, with_header=False).index([chunk])
    assert embedder.fit_calls == [["отпуск"]]


def test_search_batch_encodes_queries_once(keyword_chunks: list[Chunk]) -> None:
    embedder = KeywordEmbedder()
    retriever = DenseRetriever(embedder)
    retriever.index(keyword_chunks)
    queries = ["отпуск", "vpn", "ничего"]
    batch = retriever.search_batch(queries, k=2)
    assert embedder.query_calls == [queries]  # одно пакетное кодирование, а не цикл
    assert batch == [retriever.search(q, k=2) for q in queries]
    assert retriever.search_batch([], k=2) == []


def test_empty_index_skips_fit_and_returns_empty() -> None:
    embedder = KeywordEmbedder()
    retriever = DenseRetriever(embedder)
    retriever.index([])
    assert embedder.fit_calls == []
    assert retriever.search("отпуск", 3) == []


def test_search_before_index_raises() -> None:
    with pytest.raises(RuntimeError, match="index"):
        DenseRetriever(KeywordEmbedder()).search("отпуск", 3)


def test_wrong_embedding_shape_is_reported(keyword_chunks: list[Chunk]) -> None:
    class BrokenEmbedder(KeywordEmbedder):
        def encode_passages(self, texts: list[str]) -> np.ndarray:
            return np.ones((1, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="форм"):
        DenseRetriever(BrokenEmbedder()).index(keyword_chunks)


# --- LSAEmbedder ---


def test_lsa_is_deterministic(mini_chunks: list[Chunk]) -> None:
    texts = [c.index_text() for c in mini_chunks]
    first, second = LSAEmbedder(seed=0), LSAEmbedder(seed=0)
    first.fit(texts)
    second.fit(texts)
    np.testing.assert_array_equal(first.encode_passages(texts), second.encode_passages(texts))
    np.testing.assert_array_equal(first.encode_queries(["отпуск"]), second.encode_queries(["отпуск"]))


def test_lsa_rows_are_normalized_float32(mini_chunks: list[Chunk]) -> None:
    texts = [c.index_text() for c in mini_chunks]
    embedder = LSAEmbedder()
    embedder.fit(texts)
    vectors = embedder.encode_passages(texts)
    assert vectors.dtype == np.float32
    # 8 документов → не больше 7 компонент, сколько бы ни просили (dim=256).
    assert embedder.n_components == 7
    assert vectors.shape == (8, 7)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_lsa_respects_requested_dim(mini_chunks: list[Chunk]) -> None:
    embedder = LSAEmbedder(dim=3)
    embedder.fit([c.text for c in mini_chunks])
    assert embedder.encode_queries(["отпуск", "VPN"]).shape == (2, 3)


def test_lsa_two_documents_gives_one_component() -> None:
    embedder = LSAEmbedder(dim=256)
    embedder.fit(["отпуск двадцать восемь дней", "клиент VPN Шлюз"])
    assert embedder.n_components == 1
    assert embedder.encode_passages(["отпуск"]).shape == (1, 1)


def test_lsa_single_document_falls_back_to_tfidf() -> None:
    embedder = LSAEmbedder()
    embedder.fit(["отпуск двадцать восемь дней"])
    query = embedder.encode_queries(["отпуск"])
    passage = embedder.encode_passages(["отпуск двадцать восемь дней"])
    assert query.shape[1] == passage.shape[1] == embedder.n_components
    assert float((query @ passage.T)[0, 0]) > 0


@pytest.mark.parametrize("method", ["encode_queries", "encode_passages"])
def test_lsa_encode_before_fit_raises(method: str) -> None:
    with pytest.raises(RuntimeError, match="fit"):
        getattr(LSAEmbedder(), method)(["отпуск"])


def test_lsa_rejects_empty_fit_and_bad_dim() -> None:
    with pytest.raises(ValueError):
        LSAEmbedder().fit([])
    with pytest.raises(ValueError):
        LSAEmbedder(dim=0)


def test_lsa_empty_input_and_unknown_ngrams(mini_chunks: list[Chunk]) -> None:
    embedder = LSAEmbedder()
    embedder.fit([c.text for c in mini_chunks])
    assert embedder.encode_queries([]).shape == (0, embedder.n_components)
    # Ни одной известной n-граммы → нулевой вектор (а не NaN после деления на ноль).
    assert not np.any(embedder.encode_queries(["zzzz qqqq"]))


def test_lsa_uses_project_normalization(mini_chunks: list[Chunk]) -> None:
    embedder = LSAEmbedder()
    embedder.fit([c.text for c in mini_chunks])
    np.testing.assert_allclose(embedder.encode_queries(["ПОДАЁТСЯ"]), embedder.encode_queries(["подается"]))


def test_lsa_warns_when_svd_is_near_full_rank(
    mini_chunks: list[Chunk], caplog: pytest.LogCaptureFixture
) -> None:
    texts = [c.index_text() for c in mini_chunks]
    with caplog.at_level(logging.WARNING, logger="kbrag.retrieval.embeddings"):
        LSAEmbedder(dim=256).fit(texts)  # 8 фрагментов → 7 компонент из 7 возможных
    assert "почти полного ранга" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="kbrag.retrieval.embeddings"):
        embedder = LSAEmbedder(dim=3)
        embedder.fit(texts)  # 3 из 7: SVD действительно сжимает
    assert "почти полного ранга" not in caplog.text
    assert embedder.explained_variance is not None and 0.0 < embedder.explained_variance < 1.0


def test_lsa_with_full_rank_ranks_exactly_like_plain_tfidf() -> None:
    # Фиксируем вырождение, о котором предупреждает fit: если компонент не меньше ранга
    # матрицы TF-IDF, SVD сохраняет скалярные произведения запроса с фрагментами,
    # и LSA ранжирует ровно как TF-IDF без сжатия. Повторы текстов делают ранг (6) меньше
    # числа компонент (8 из 9 фрагментов), поэтому равенство здесь точное.
    unique = [
        "отпуск составляет 28 календарных дней",
        "заявление на отпуск подаётся за 14 дней",
        "клиент VPN устанавливают через сервис-деск",
        "пароль меняется каждые 90 дней",
        "компенсация интернета 1500 рублей в месяц",
        "удалённая работа до 3 дней в неделю",
    ]
    texts = unique + unique[:3]
    queries = ["сколько дней отпуска", "как поменять пароль", "VPN клиент", "работа из дома"]

    lsa = LSAEmbedder(dim=256)
    lsa_passages = lsa.fit_encode_passages(texts)
    lsa_scores = lsa.encode_queries(queries) @ lsa_passages.T

    tfidf = lsa._vectorizer  # тот же векторизатор, но без SVD
    tfidf_passages = l2_normalize(tfidf.transform(texts).toarray())
    tfidf_scores = l2_normalize(tfidf.transform(queries).toarray()) @ tfidf_passages.T

    for lsa_row, tfidf_row in zip(lsa_scores, tfidf_scores, strict=True):
        # У LSA другая норма запроса (часть его n-грамм вне пространства корпуса), поэтому
        # сравниваем с точностью до множителя: порядок фрагментов обязан совпасть.
        np.testing.assert_allclose(lsa_row / lsa_row.max(), tfidf_row / tfidf_row.max(), atol=1e-4)


def test_lsa_fit_encode_passages_equals_fit_then_encode(mini_chunks: list[Chunk]) -> None:
    texts = [c.index_text() for c in mini_chunks]
    separate = LSAEmbedder(dim=5)
    separate.fit(texts)
    np.testing.assert_allclose(
        LSAEmbedder(dim=5).fit_encode_passages(texts), separate.encode_passages(texts), atol=1e-5
    )  # допуск — на округление float32


def test_dense_index_builds_lsa_tfidf_once(mini_chunks: list[Chunk], monkeypatch: pytest.MonkeyPatch) -> None:
    # Регрессия: раньше index вызывал fit (fit_transform TF-IDF), а затем encode_passages
    # заново прогонял те же тексты через transform — двойная работа на каждом построении индекса.
    calls: list[int] = []
    original = TfidfVectorizer.transform

    def counting_transform(self, raw_documents):
        calls.append(len(raw_documents))
        return original(self, raw_documents)

    monkeypatch.setattr(TfidfVectorizer, "transform", counting_transform)
    retriever = DenseRetriever(LSAEmbedder(dim=5))
    retriever.index(mini_chunks)
    assert calls == []  # при индексации — только fit_transform
    retriever.search("отпуск", k=1)
    assert calls == [1]  # запрос кодируется через transform, как и положено


def test_dense_lsa_on_mini_corpus(mini_chunks: list[Chunk], mini_gold_records: list[dict]) -> None:
    retriever = DenseRetriever(LSAEmbedder())
    retriever.index(mini_chunks)
    by_id = {item["id"]: item for item in mini_gold_records}
    for qid in ("t-001", "t-002", "t-004"):
        top = retriever.search(by_id[qid]["question"], k=1)
        assert top[0].chunk.doc_id == by_id[qid]["evidence"][0]["doc_id"], qid
    # Перефразировка без общих слов с BM25-точностью: символьные n-граммы всё равно находят тему.
    top2 = retriever.search("работать из дома", k=2)
    assert {r.chunk.doc_id for r in top2} == {"hr-remote-2023", "hr-remote-2025"}
    assert retriever.search("zzzz qqqq", k=3) == []


# --- SentenceTransformerEmbedder с фейковой моделью (без скачивания) ---


class FakeSentenceTransformer:
    """Подмена sentence_transformers.SentenceTransformer: детерминированные векторы, учёт вызовов."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.encode_calls: list[list[str]] = []

    def encode(
        self, texts, batch_size=32, normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=None
    ):
        self.encode_calls.append(list(texts))
        vectors = np.array([[len(t), t.count("а") + 1.0, 1.0] for t in texts], dtype=np.float64)
        if normalize_embeddings:
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


@pytest.fixture()
def fake_st(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


def test_e5_models_get_default_prefixes(fake_st: types.ModuleType) -> None:
    embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-small", device="cpu")
    assert (embedder.query_prefix, embedder.passage_prefix) == ("query: ", "passage: ")
    embedder.encode_queries(["вопрос"])
    embedder.encode_passages(["текст"])
    assert embedder.model.encode_calls == [["query: вопрос"], ["passage: текст"]]
    assert embedder.model.device == "cpu"


def test_other_models_have_no_prefix_and_explicit_prefix_wins(fake_st: types.ModuleType) -> None:
    plain = SentenceTransformerEmbedder("sentence-transformers/LaBSE", device="cpu")
    assert (plain.query_prefix, plain.passage_prefix) == ("", "")
    # Явная пустая строка — тоже явный выбор, а не «определить автоматически».
    custom = SentenceTransformerEmbedder(
        "intfloat/multilingual-e5-small", query_prefix="", passage_prefix="doc: ", device="cpu"
    )
    assert (custom.query_prefix, custom.passage_prefix) == ("", "doc: ")


def test_sentence_transformer_output_is_normalized_float32(fake_st: types.ModuleType) -> None:
    embedder = SentenceTransformerEmbedder("some-model", device="cpu")
    vectors = embedder.encode_passages(["один", "два текста"])
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    assert embedder.encode_queries([]).shape[0] == 0


def test_embedding_cache_roundtrip(fake_st: types.ModuleType, tmp_path: Path) -> None:
    texts = ["отпуск 28 дней", "клиент Шлюз"]
    first = SentenceTransformerEmbedder(
        "intfloat/multilingual-e5-small", device="cpu", cache_dir=str(tmp_path)
    )
    path = first.cache_path(texts)
    assert path is not None
    assert path.parent == tmp_path / "emb"
    assert re.fullmatch(r"[0-9a-f]{20}\.npy", path.name)

    vectors = first.encode_passages(texts)
    assert path.exists()
    assert len(first.model.encode_calls) == 1

    # Новый экземпляр (как при следующем запуске) берёт векторы из кэша, не вызывая модель.
    second = SentenceTransformerEmbedder(
        "intfloat/multilingual-e5-small", device="cpu", cache_dir=str(tmp_path)
    )
    np.testing.assert_array_equal(second.encode_passages(texts), vectors)
    assert second.model.encode_calls == []


def test_cache_key_depends_on_model_prefix_and_texts(fake_st: types.ModuleType, tmp_path: Path) -> None:
    def path_for(model: str, texts: list[str], prefix: str | None = None) -> Path | None:
        embedder = SentenceTransformerEmbedder(
            model, passage_prefix=prefix, device="cpu", cache_dir=str(tmp_path)
        )
        return embedder.cache_path(texts)

    base = path_for("model-a", ["ab", "c"])
    assert base == path_for("model-a", ["ab", "c"])
    assert base != path_for("model-b", ["ab", "c"])
    assert base != path_for("model-a", ["ab", "c"], prefix="passage: ")
    assert base != path_for("model-a", ["a", "bc"])  # границы между текстами учитываются
    assert base != path_for("model-a", ["ab", "c", "d"])


def test_cache_key_depends_on_model_revision_and_library_version(
    fake_st: types.ModuleType, tmp_path: Path
) -> None:
    # Регрессия: ключ не учитывал ревизию весов и версию библиотеки, и после обновления
    # модели под тем же именем молча подхватывались старые векторы.
    texts = ["отпуск 28 дней"]
    base = SentenceTransformerEmbedder("model-a", device="cpu", cache_dir=str(tmp_path)).cache_path(texts)

    class RevisionedModel(FakeSentenceTransformer):
        model_card_data = types.SimpleNamespace(base_model_revision="614241f6")

    fake_st.SentenceTransformer = RevisionedModel
    revisioned = SentenceTransformerEmbedder("model-a", device="cpu", cache_dir=str(tmp_path))
    assert revisioned.model_revision == "614241f6"
    assert revisioned.cache_path(texts) != base

    fake_st.SentenceTransformer = FakeSentenceTransformer
    fake_st.__version__ = "99.0"
    assert (
        SentenceTransformerEmbedder("model-a", device="cpu", cache_dir=str(tmp_path)).cache_path(texts)
        != base
    )


def test_cache_write_failure_is_not_fatal(
    fake_st: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Регрессия: на Windows os.replace бросает PermissionError, если кэш-файл открыт другим
    # процессом, и это роняло всю оценку, хотя кэш — лишь ускорение.
    def locked_replace(src, dst):
        raise PermissionError(32, "файл занят другим процессом")

    monkeypatch.setattr("kbrag.retrieval.embeddings.os.replace", locked_replace)
    embedder = SentenceTransformerEmbedder("model-a", device="cpu", cache_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="kbrag.retrieval.embeddings"):
        vectors = embedder.encode_passages(["первый", "второй"])
    assert vectors.shape == (2, 3)
    assert "Не удалось сохранить кэш" in caplog.text
    assert list((tmp_path / "emb").iterdir()) == []  # временный файл убран, «половинок» нет


def test_cache_temp_files_are_unique(
    fake_st: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Регрессия: временный файл назывался path.with_suffix(".tmp") — один на всех, и два процесса
    # с одним ключом писали в него одновременно.
    sources: list[str] = []
    original_replace = os.replace

    def recording_replace(src, dst):
        sources.append(str(src))
        original_replace(src, dst)

    monkeypatch.setattr("kbrag.retrieval.embeddings.os.replace", recording_replace)
    path = tmp_path / "emb" / "0123456789abcdef0123.npy"
    vectors = np.ones((2, 3), dtype=np.float32)
    SentenceTransformerEmbedder._save_cache(path, vectors)
    SentenceTransformerEmbedder._save_cache(path, vectors)
    assert len(set(sources)) == 2
    assert str(path.with_suffix(".tmp")) not in sources
    np.testing.assert_array_equal(np.load(path), vectors)
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_corrupted_or_mismatched_cache_is_recomputed(fake_st: types.ModuleType, tmp_path: Path) -> None:
    texts = ["первый", "второй"]
    embedder = SentenceTransformerEmbedder("model-a", device="cpu", cache_dir=str(tmp_path))
    path = embedder.cache_path(texts)
    path.parent.mkdir(parents=True)

    path.write_bytes(b"not a numpy file")
    vectors = embedder.encode_passages(texts)
    assert vectors.shape == (2, 3)
    assert len(embedder.model.encode_calls) == 1
    np.testing.assert_array_equal(np.load(path), vectors)  # кэш перезаписан корректным файлом

    np.save(path, np.zeros((5, 3), dtype=np.float32))  # «чужой» размер
    np.testing.assert_array_equal(embedder.encode_passages(texts), vectors)
    assert len(embedder.model.encode_calls) == 2


def test_no_cache_dir_means_no_files_and_queries_never_cached(
    fake_st: types.ModuleType, tmp_path: Path
) -> None:
    embedder = SentenceTransformerEmbedder("model-a", device="cpu")
    assert embedder.cache_path(["текст"]) is None
    embedder.encode_passages(["текст"])
    embedder.encode_passages(["текст"])
    assert len(embedder.model.encode_calls) == 2

    cached = SentenceTransformerEmbedder("model-a", device="cpu", cache_dir=str(tmp_path))
    cached.encode_queries(["вопрос"])
    cached.encode_queries(["вопрос"])
    assert len(cached.model.encode_calls) == 2
    assert not (tmp_path / "emb").exists()


def test_missing_sentence_transformers_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # import → ImportError
    with pytest.raises(ImportError, match=r"pip install -e \.\[neural\]"):
        SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
    assert neural_available() is False


def test_neural_available_with_fake_module(fake_st: types.ModuleType) -> None:
    assert neural_available() is True


# --- resolve_device и make_embedder ---


@pytest.mark.parametrize("cuda", [True, False])
def test_resolve_device_auto(monkeypatch: pytest.MonkeyPatch, cuda: bool) -> None:
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert resolve_device("auto") == ("cuda" if cuda else "cpu")
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    assert resolve_device("auto") == "cpu"


def test_make_embedder_lsa() -> None:
    embedder = make_embedder(DenseConfig(embedder="lsa", lsa_dim=64))
    assert isinstance(embedder, LSAEmbedder)
    assert embedder.dim == 64


def test_make_embedder_sentence_transformers(fake_st: types.ModuleType, tmp_path: Path) -> None:
    cfg = DenseConfig(embedder="sentence-transformers", model="my-e5-model", batch_size=8, device="cpu")
    embedder = make_embedder(cfg, cache_dir=str(tmp_path))
    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert (embedder.model_name, embedder.batch_size, embedder.cache_dir) == ("my-e5-model", 8, str(tmp_path))
    assert embedder.query_prefix == "query: "
