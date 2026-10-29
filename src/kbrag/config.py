"""Конфигурация проекта.

Все «ручки» системы (размер фрагментов, тип поиска, модель, пороги) живут в YAML,
а не в коде. Это позволяет запускать серию экспериментов, меняя только конфиг,
и всегда знать, с какими параметрами получен результат.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    # Опечатка в имени параметра в YAML должна падать с ошибкой, а не молча игнорироваться.
    model_config = ConfigDict(extra="forbid")


def _check_http_url(value: str) -> str:
    """``base_url`` без схемы («localhost:8000/v1») httpx не поймёт — падаем при чтении конфига, а не на первом /ask."""
    if not value.strip().lower().startswith(("http://", "https://")):
        raise ValueError(f"base_url должен начинаться с http:// или https://, получено {value!r}")
    return value


class ChunkingConfig(_Strict):
    strategy: Literal["fixed", "recursive"] = "recursive"
    chunk_size: int = Field(180, gt=10, description="Максимальный размер фрагмента в словах")
    chunk_overlap: int = Field(30, ge=0, description="Перекрытие соседних фрагментов в словах (для fixed)")
    with_header: bool = Field(
        True, description="Приписывать к фрагменту заголовок документа/раздела при индексации"
    )


class BM25Config(_Strict):
    k1: float = 1.5
    b: float = 0.75
    # Части составных токенов («vpn-шлюз» → + «vpn», «шлюз»), см. kbrag.text.analyze:
    # none — не добавлять, mixed — только у смеси латиницы и кириллицы, all — у любых.
    split_compounds: Literal["none", "mixed", "all"] = "none"


class DenseConfig(_Strict):
    embedder: Literal["lsa", "sentence-transformers"] = "lsa"
    model: str = "intfloat/multilingual-e5-small"
    # Commit снапшота модели на Hugging Face Hub: веса под тем же именем могут обновиться,
    # и метрики «того же» конфига сдвинутся. None — последняя версия (или та, что уже в кэше).
    revision: str | None = None
    query_prefix: str | None = None  # None → определить по имени модели (e5: "query: ")
    passage_prefix: str | None = None  # None → определить по имени модели (e5: "passage: ")
    # Подобрано на dev (configs/offline.yaml): при ~260 фрагментах 256 — SVD почти полного ранга ≈ TF-IDF.
    lsa_dim: int = 128
    batch_size: int = 32
    device: str = "auto"


class HybridConfig(_Strict):
    fusion: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    weights: dict[str, float] = Field(default_factory=lambda: {"bm25": 0.5, "dense": 0.5})


class RetrievalConfig(_Strict):
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    top_k: int = Field(5, gt=0, description="Сколько фрагментов отдаём генератору")
    candidates_k: int = Field(30, gt=0, description="Сколько кандидатов достаём до реранкинга")
    bm25: BM25Config = Field(default_factory=BM25Config)
    dense: DenseConfig = Field(default_factory=DenseConfig)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)


class RerankConfig(_Strict):
    enabled: bool = False
    model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    revision: str | None = None  # commit снапшота на HF Hub, как у retrieval.dense.revision
    batch_size: int = 16
    max_length: int = 512
    device: str = "auto"


class GenerationConfig(_Strict):
    backend: Literal["extractive", "openai"] = "extractive"
    # OpenAI-совместимый API: vLLM, Ollama, LM Studio, OpenAI и т.п.
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    api_key_env: str = "LLM_API_KEY"  # имя переменной окружения, сам ключ в конфиг не пишем
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: float = 60.0  # общий лимит на один ответ LLM вместе с повторами и паузами между ними
    max_retries: int = 3
    max_sentences: int = 3  # для extractive-генератора
    max_context_chars: int = 6000

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        return _check_http_url(value)


class AbstentionConfig(_Strict):
    enabled: bool = True
    # Как решаем об отказе: threshold — порог по одной «уверенности» поиска (ниже);
    # classifier — обученная модель на признаках выдачи (kbrag.abstention, файл model_path).
    # Нет файла модели или он от другого поиска — сервис откатывается на threshold.
    method: Literal["threshold", "classifier"] = "threshold"
    # число → фиксированный порог; "auto" → калибруется на dev-выборке при оценке
    # (а в сервисе берётся из calibration_path, если файл есть); null → не отказываться по порогу
    threshold: float | Literal["auto"] | None = "auto"
    # откуда брать «уверенность»: auto = rerank, если включён, иначе dense, иначе bm25
    score_source: Literal["auto", "rerank", "dense", "bm25"] = "auto"
    calibration_path: str | None = "reports/calibration.json"
    # Файл модели отказа (пишет `kbrag train-abstention`); свой для каждого конфига поиска.
    model_path: str | None = None


class GuardrailsConfig(_Strict):
    enabled: bool = True
    action: Literal["drop_line", "drop_chunk", "flag_only"] = "drop_line"


class JudgeConfig(_Strict):
    enabled: bool = False
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    api_key_env: str = "LLM_API_KEY"
    timeout_s: float = 60.0  # общий лимит на одну оценку вместе с повторами

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        return _check_http_url(value)


class EvalConfig(_Strict):
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    report_split: Literal["test", "dev", "all"] = "test"
    judge: JudgeConfig = Field(default_factory=JudgeConfig)


class AppConfig(_Strict):
    corpus_dir: str = "data/corpus"
    gold_path: str = "data/eval/gold.jsonl"
    # Обучающие вопросы для классификатора отказа (`kbrag build-train`). Отдельно от gold:
    # на gold оценивают, и обучение на нём было бы утечкой.
    train_path: str = "data/eval/train.jsonl"
    cache_dir: str = ".cache"
    reports_dir: str = "reports"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    abstention: AbstentionConfig = Field(default_factory=AbstentionConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


# Переменные окружения, которые перекрывают конфиг (удобно в Docker/CI).
_ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "KBRAG_GENERATION_BACKEND": ("generation", "backend"),
    "KBRAG_LLM_BASE_URL": ("generation", "base_url"),
    "KBRAG_LLM_MODEL": ("generation", "model"),
    # Медленной локальной модели (4 бита на слабой видеокарте) нужны другие лимиты, чем vLLM на A100;
    # переменные позволяют поднять их, не копируя YAML.
    "KBRAG_LLM_TIMEOUT_S": ("generation", "timeout_s"),
    "KBRAG_LLM_MAX_TOKENS": ("generation", "max_tokens"),
    "KBRAG_CORPUS_DIR": ("corpus_dir",),
    # Запасной порог конфига с классификатором отказа пересчитывают тем же конфигом, но с порогом:
    # KBRAG_ABSTENTION_METHOD=threshold kbrag eval --config configs/default.yaml --save-calibration.
    # Отпечаток калибровки от способа отказа не зависит, поэтому сервис с классификатором этот порог примет.
    "KBRAG_ABSTENTION_METHOD": ("abstention", "method"),
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Рекурсивно сливает ``override`` поверх ``base`` (не изменяя аргументы)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_env(data: dict[str, Any]) -> dict[str, Any]:
    for env_name, path in _ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if not value:
            continue
        node = data
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return data


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Читает YAML (если указан), накладывает ``overrides`` и переменные окружения.

    Порядок приоритета: значения по умолчанию < YAML < overrides < переменные окружения.
    """
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    if overrides:
        data = deep_merge(data, overrides)
    data = _apply_env(data)
    return AppConfig.model_validate(data)


def config_fingerprint(cfg: AppConfig, *sections: str) -> str:
    """Короткий хэш выбранных секций конфига — ключ для кэширования индексов."""
    dumped = cfg.model_dump(mode="json")
    payload = {s: dumped[s] for s in sections} if sections else dumped
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
