"""Тесты конфигов из configs/: все YAML валидны, а в сетке экспериментов нет «пустых» шагов.

Сетка устроена «лесенкой»: соседние эксперименты отличаются одной настройкой, и разницу в
метриках приписывают именно ей. Если настройка на этих данных ничего не меняет (например,
chunk_size больше самого длинного раздела), в leaderboard появится строка-дубль, а вывод
«размер не важен» будет ложным. Такие ошибки ловим здесь, дёшево и до запуска сетки.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from kbrag.config import AppConfig, load_config
from kbrag.eval.runner import load_experiments
from kbrag.ingest import load_corpus
from kbrag.ingest.chunking import chunk_corpus, make_chunker

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


@pytest.fixture(autouse=True)
def _project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    # Пути в конфигах (base, corpus_dir) записаны от корня проекта, как при запуске `kbrag`.
    # Переменные KBRAG_* убирает общая фикстура conftest._isolate_kbrag_env.
    monkeypatch.chdir(ROOT)


def experiment_configs() -> dict[str, AppConfig]:
    """Итоговый конфиг каждого эксперимента: base + overrides (как в run_grid)."""
    base, specs = load_experiments(CONFIGS / "experiments.yaml")
    return {spec.name: load_config(base, overrides=spec.overrides) for spec in specs}


@pytest.mark.parametrize("name", ["default.yaml", "offline.yaml"])
def test_service_configs_are_valid(name: str) -> None:
    load_config(CONFIGS / name)


def test_every_experiment_config_is_valid() -> None:
    assert experiment_configs()


def _chunk_signature(cfg: AppConfig, docs: list) -> list[tuple[str, int, int, str]]:
    """Что реально попадает в индекс: границы фрагментов и текст с заголовком или без."""
    chunks = chunk_corpus(docs, make_chunker(cfg.chunking))
    return [(c.chunk_id, c.start, c.end, c.index_text(cfg.chunking.with_header)) for c in chunks]


def test_chunking_steps_of_the_grid_really_change_the_chunks() -> None:
    """Регрессия: hybrid_bm25_e5_rerank_chunk300 давал ту же нарезку, что chunk_size 180.

    Самый длинный раздел корпуса — 166 слов, а RecursiveChunker режет только разделы длиннее
    лимита, поэтому любое значение ≥ 166 даёт «один раздел = один фрагмент».
    """
    configs = experiment_configs()
    docs = load_corpus(next(iter(configs.values())).corpus_dir)

    pairs = []
    for (name_a, cfg_a), (name_b, cfg_b) in itertools.combinations(configs.items(), 2):
        rest_a = cfg_a.model_dump(mode="json", exclude={"chunking"})
        rest_b = cfg_b.model_dump(mode="json", exclude={"chunking"})
        if rest_a == rest_b and cfg_a.chunking != cfg_b.chunking:
            pairs.append((name_a, name_b))
    assert pairs, "в сетке нет пар, отличающихся только нарезкой — тест ничего не проверил"

    signatures: dict[str, list] = {}
    for name_a, name_b in pairs:
        for name in (name_a, name_b):
            if name not in signatures:
                signatures[name] = _chunk_signature(configs[name], docs)
        assert signatures[name_a] != signatures[name_b], (
            f"{name_a} и {name_b} отличаются только секцией chunking, но индексируют одинаковые "
            "фрагменты: такой шаг абляции ничего не проверяет"
        )


@pytest.mark.parametrize("name", ["default.yaml", "offline.yaml"])
def test_calibration_hint_in_config_mentions_save_flag(name: str) -> None:
    """Регрессия: комментарий советовал `kbrag eval --config …` без `--save-calibration`.

    Без флага файл калибровки не создаётся, и сервис молча работает без порога отказа.
    """
    hints = [
        line
        for line in (CONFIGS / name).read_text(encoding="utf-8").splitlines()
        if "kbrag eval --config" in line
    ]
    assert hints, "в конфиге должна быть подсказка, как откалибровать порог для сервиса"
    assert all("--save-calibration" in line for line in hints)


@pytest.mark.parametrize("section", ["generation", "judge"])
@pytest.mark.parametrize("url", ["localhost:8000/v1", "ftp://llm.test/v1", ""])
def test_llm_base_url_without_http_scheme_fails_when_config_is_read(section: str, url: str) -> None:
    # Ошибка в адресе LLM видна при загрузке конфига, а не при первом вопросе к сервису.
    data = (
        {"generation": {"base_url": url}}
        if section == "generation"
        else {"eval": {"judge": {"base_url": url}}}
    )
    with pytest.raises(ValueError, match="http:// или https://"):
        AppConfig.model_validate(data)


def test_llm_base_url_from_environment_is_validated_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KBRAG_LLM_BASE_URL", "llm:8000/v1")
    with pytest.raises(ValueError, match="KBRAG|base_url"):
        load_config(CONFIGS / "offline.yaml")
    monkeypatch.setenv("KBRAG_LLM_BASE_URL", "https://llm.example/v1")
    assert load_config(CONFIGS / "offline.yaml").generation.base_url == "https://llm.example/v1"
