"""Тесты классификатора отказа: признаки выдачи, модели и их файлы (без пайплайна и без данных проекта).

Признаки проверяются на фрагментах, собранных вручную, — значения легко посчитать в уме.
Модели обучаются на синтетике за доли секунды; catboost-тесты пропускаются, если пакета нет.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from kbrag.abstention import (
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    AbstentionModel,
    extract_features,
    features_to_matrix,
    load_abstention_model,
    make_estimator,
)
from kbrag.abstention.models import import_catboost
from kbrag.config import AppConfig
from kbrag.pipeline import calibration_fingerprint
from kbrag.types import Chunk, ScoredChunk


def scored(
    chunk_id: str, text: str, details: dict[str, float], *, status: str = "active", title: str = ""
) -> ScoredChunk:
    """Фрагмент выдачи с нужными score; ``doc_id`` — часть ``chunk_id`` до ``::``."""
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("::")[0],
        text=text,
        start=0,
        end=len(text),
        title=title,
        metadata={"status": status},
    )
    return ScoredChunk(chunk=chunk, score=0.0, source="hybrid", details=dict(details))


QUESTION = "Пароль VPN клиент"  # три значимых стема: парол, vpn, клиент

# Гибридная выдача: лучший по BM25 — второй фрагмент, лучший по dense — первый (поисковики спорят).
HYBRID_CONTEXTS = [
    scored("it-vpn::001", "Пароль VPN", {"bm25": 4.0, "dense": 0.9}),
    scored("it-vpn::002", "Клиент установлен", {"bm25": 6.0, "dense": 0.5}),
    scored("hr-old::000", "Архивная редакция", {"dense": 0.4}, status="archived"),  # BM25 его не нашёл
]


# ---------------------------------------------------------------------------
# Признаки
# ---------------------------------------------------------------------------


def test_feature_list_is_fixed_small_and_documented() -> None:
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) < 25
    assert list(FEATURE_DESCRIPTIONS) == list(FEATURE_NAMES)
    assert all(description.strip() for description in FEATURE_DESCRIPTIONS.values())
    assert list(extract_features(QUESTION, HYBRID_CONTEXTS)) == list(FEATURE_NAMES)


def test_source_features_sort_scores_instead_of_trusting_positions() -> None:
    f = extract_features(QUESTION, HYBRID_CONTEXTS)

    # BM25 есть у двух фрагментов из трёх: top1 — максимум, а не score первого места.
    assert (f["bm25_top1"], f["bm25_top2"], f["bm25_gap"], f["bm25_mean"]) == (6.0, 4.0, 2.0, 5.0)
    assert f["dense_top1"] == 0.9 and f["dense_top2"] == 0.5
    assert f["dense_gap"] == pytest.approx(0.4) and f["dense_mean"] == pytest.approx(0.6)
    assert f["bm25_top1_per_term"] == pytest.approx(6.0 / 3)


def test_missing_source_gives_nan_not_zero() -> None:
    f = extract_features(QUESTION, HYBRID_CONTEXTS)
    # Реранкера нет: «нет значения» и «реранкер поставил 0» — разные вещи.
    for name in ("rerank_top1", "rerank_top2", "rerank_gap", "rerank_mean"):
        assert math.isnan(f[name])

    bm25_only = [scored("a::000", "Пароль", {"bm25": 3.0})]
    f = extract_features(QUESTION, bm25_only)
    assert f["bm25_top1"] == 3.0
    assert math.isnan(f["bm25_top2"]) and math.isnan(f["bm25_gap"])  # второго фрагмента нет
    assert math.isnan(f["dense_top1"]) and math.isnan(f["bm25_dense_same_top1"])


def test_agreement_documents_and_archived_share() -> None:
    f = extract_features(QUESTION, HYBRID_CONTEXTS)
    assert f["bm25_dense_same_top1"] == 0.0  # BM25 → it-vpn::002, dense → it-vpn::001
    assert f["n_docs"] == 2.0
    assert f["archived_share"] == pytest.approx(1 / 3)

    agree = [
        scored("a::000", "x", {"bm25": 5.0, "dense": 0.8}),
        scored("b::000", "y", {"bm25": 1.0, "dense": 0.1}),
    ]
    assert extract_features(QUESTION, agree)["bm25_dense_same_top1"] == 1.0


def test_question_coverage_uses_first_fragment_and_whole_list() -> None:
    f = extract_features(QUESTION, HYBRID_CONTEXTS)
    assert f["q_len"] == 3.0
    assert f["q_cov_top1"] == pytest.approx(2 / 3)  # «Пароль VPN»: нет только «клиент»
    assert f["q_cov_topk"] == 1.0  # «клиент» нашёлся во втором фрагменте
    assert f["q_has_digits"] == 0.0
    assert extract_features("Пароль из 12 символов", HYBRID_CONTEXTS)["q_has_digits"] == 1.0


def test_coverage_counts_the_header_the_generator_also_sees() -> None:
    titled = [scored("a::000", "Текст раздела", {"bm25": 1.0}, title="Пароль VPN клиент")]
    assert extract_features(QUESTION, titled)["q_cov_top1"] == 1.0


def test_empty_retrieval() -> None:
    f = extract_features(QUESTION, [])
    assert f["n_docs"] == 0.0
    assert f["q_cov_topk"] == 0.0  # ни одно слово вопроса не найдено
    assert math.isnan(f["q_cov_top1"])  # «лучшего фрагмента» нет
    assert math.isnan(f["archived_share"]) and math.isnan(f["bm25_top1"])


def test_question_of_stopwords_only_has_undefined_coverage() -> None:
    f = extract_features("и в на", HYBRID_CONTEXTS)
    assert f["q_len"] == 0.0
    assert math.isnan(f["q_cov_top1"]) and math.isnan(f["q_cov_topk"]) and math.isnan(f["bm25_top1_per_term"])


def test_features_to_matrix_keeps_column_order_and_fills_nan() -> None:
    matrix = features_to_matrix([{"bm25_top1": 1.5}, {"q_len": 3.0, "unknown": 9.0}])
    assert matrix.shape == (2, len(FEATURE_NAMES))
    assert matrix[0, FEATURE_NAMES.index("bm25_top1")] == 1.5
    assert matrix[1, FEATURE_NAMES.index("q_len")] == 3.0
    assert np.isnan(matrix).sum() == 2 * len(FEATURE_NAMES) - 2


# ---------------------------------------------------------------------------
# Модели и их файлы
# ---------------------------------------------------------------------------


def synthetic_data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Матрица в формате FEATURE_NAMES: класс зависит от двух признаков, есть NaN и пустой столбец."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    matrix = rng.normal(size=(n, len(FEATURE_NAMES)))
    matrix[:, FEATURE_NAMES.index("dense_top1")] -= 1.5 * y
    matrix[:, FEATURE_NAMES.index("q_cov_top1")] -= 1.0 * y
    matrix[rng.random(matrix.shape) < 0.1] = np.nan  # пропуски
    matrix[:, FEATURE_NAMES.index("rerank_top1")] = np.nan  # источника нет вовсе (офлайн-конфиг)
    return matrix, y


def test_logreg_numpy_prediction_matches_sklearn_and_survives_json(tmp_path: Path) -> None:
    matrix, y = synthetic_data()
    estimator = make_estimator("logreg", {"C": 1.0}).fit(matrix, y)
    model = AbstentionModel.from_estimator(
        "logreg", estimator, 0.5, params={"C": 1.0}, metadata={"note": "x"}
    )

    expected = estimator.predict_proba(matrix)[:, 1]
    np.testing.assert_allclose(model.predict_proba(matrix), expected, rtol=0, atol=1e-12)

    path = model.save(tmp_path / "model.json")
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text  # строгий JSON
    restored = AbstentionModel.load(path)
    np.testing.assert_allclose(restored.predict_proba(matrix), expected, rtol=0, atol=1e-12)
    assert restored.decision_threshold == 0.5 and restored.params == {"C": 1.0}
    assert restored.metadata == {"note": "x"} and restored.feature_names == FEATURE_NAMES


def test_predict_accepts_feature_dicts_like_the_service() -> None:
    matrix, y = synthetic_data()
    model = AbstentionModel.from_estimator("logreg", make_estimator("logreg").fit(matrix, y), 0.5)
    row = extract_features(QUESTION, HYBRID_CONTEXTS)
    assert model.predict_one(row) == pytest.approx(float(model.predict_proba(features_to_matrix([row]))[0]))
    assert 0.0 <= model.predict_one(row) <= 1.0


def test_decision_rule_is_p_greater_or_equal_threshold() -> None:
    matrix, y = synthetic_data()
    model = AbstentionModel.from_estimator("logreg", make_estimator("logreg").fit(matrix, y), 0.4)
    assert model.abstains(0.4) and model.abstains(0.9)
    assert not model.abstains(0.39)


def test_catboost_round_trip_through_cbm(tmp_path: Path) -> None:
    pytest.importorskip("catboost")
    matrix, y = synthetic_data()
    estimator = make_estimator("catboost", {"depth": 2, "iterations": 50}).fit(matrix, y)
    model = AbstentionModel.from_estimator("catboost", estimator, 0.6, params={"depth": 2})

    path = model.save(tmp_path / "abstention.json")
    assert (tmp_path / "abstention.cbm").is_file()
    restored = AbstentionModel.load(path)
    np.testing.assert_allclose(
        restored.predict_proba(matrix), estimator.predict_proba(matrix)[:, 1], atol=1e-12
    )

    # JSON и деревья из разных обучений не должны молча склеиться.
    (tmp_path / "abstention.cbm").write_bytes(b"not a model")
    with pytest.raises(ValueError, match="sha256"):
        AbstentionModel.load(path)


def test_catboost_without_package_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "catboost", None)  # так Python ведёт себя, когда пакета нет
    with pytest.raises(ImportError, match=r"pip install -e \.\[classic\]"):
        import_catboost()
    with pytest.raises(ImportError, match="classic"):
        make_estimator("catboost")


def test_unknown_model_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="неизвестный тип"):
        make_estimator("svm")


# ---------------------------------------------------------------------------
# Загрузка моделью сервиса
# ---------------------------------------------------------------------------


def make_cfg(tmp_path: Path, **retrieval) -> AppConfig:
    return AppConfig.model_validate(
        {
            "retrieval": {"mode": "bm25", **retrieval},
            "abstention": {"method": "classifier", "model_path": (tmp_path / "abstention.json").as_posix()},
        }
    )


def saved_model(cfg: AppConfig, corpus_sha256: str = "corpus-1") -> AbstentionModel:
    matrix, y = synthetic_data()
    model = AbstentionModel.from_estimator(
        "logreg",
        make_estimator("logreg").fit(matrix, y),
        0.5,
        metadata={"fingerprint": calibration_fingerprint(cfg), "corpus_sha256": corpus_sha256},
    )
    model.save(cfg.abstention.model_path)
    return model


def test_service_loads_matching_model(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    saved_model(cfg)
    model = load_abstention_model(cfg, corpus_sha256="corpus-1")
    assert model is not None and model.model_type == "logreg"


def test_missing_model_file_is_a_warning_not_a_crash(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        assert load_abstention_model(make_cfg(tmp_path)) is None
    assert "kbrag train-abstention" in caplog.text


def test_model_for_other_search_or_corpus_is_not_applied(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    saved_model(make_cfg(tmp_path))
    with caplog.at_level(logging.WARNING):
        # Другой top_k — другая выдача, другие признаки: модель к ним не подходит.
        assert load_abstention_model(make_cfg(tmp_path, top_k=7), corpus_sha256="corpus-1") is None
        assert "fingerprint" in caplog.text
        assert load_abstention_model(make_cfg(tmp_path), corpus_sha256="corpus-2") is None
        assert "корпусе" in caplog.text


def test_model_with_other_feature_list_is_not_applied(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    saved_model(cfg)
    path = Path(cfg.abstention.model_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["feature_names"] = list(reversed(data["feature_names"]))  # код признаков поменялся после обучения
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_abstention_model(cfg, corpus_sha256="corpus-1") is None


def test_broken_model_file_is_a_warning(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    Path(cfg.abstention.model_path).write_text("{not json", encoding="utf-8")
    assert load_abstention_model(cfg) is None
