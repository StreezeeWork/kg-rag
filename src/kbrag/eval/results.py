"""Результат одного эксперимента и единые имена метрик.

Имена метрик — это контракт между тем, кто считает (``runner``), и тем, кто
показывает (``report``, CLI с ``--fail-under``). Поэтому они собраны здесь,
в одном месте, а не разбросаны строковыми литералами по модулям.

Структура ``ExperimentResult.metrics``::

    {
      "dev":  {"hit@1": 0.8, ..., "n_questions": 60},
      "test": {...},
      "all":  {...},
      "by_type": {"numeric": {...}, "multi_hop": {...}, ...},  # для сплита отчёта
    }
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_K_VALUES = (1, 3, 5, 10)
METRIC_SPLITS = ("dev", "test", "all")
BY_TYPE_KEY = "by_type"

RETRIEVAL_PREFIXES = ("hit", "recall", "precision", "ndcg")
GENERATION_METRICS = (
    "fact_recall",
    "fact_recall_answered",  # fact_recall только по вопросам, где система не отказалась
    "token_f1",
    "citation_precision",
    "citation_recall",
    "invalid_citation_rate",
)
ABSTENTION_METRICS = ("abstain_precision", "abstain_recall", "abstain_f1", "false_abstention_rate")
LATENCY_METRICS = ("latency_p50_ms", "latency_p95_ms")

# Поля строки per_question (помимо метрик вопроса под теми же именами, что и в metrics).
PER_QUESTION_FIELDS = (
    "qid",
    "type",
    "split",
    "question",
    "gold_answer",
    "answer",
    "abstained",
    "confidence",
    "retrieved",
    "flags",
    "latency_ms",
)


def retrieval_metric_names(k_values: Sequence[int] = DEFAULT_K_VALUES) -> list[str]:
    """``hit@1 … hit@10, recall@…, precision@…, ndcg@…, mrr`` для заданных k."""
    return [f"{prefix}@{k}" for prefix in RETRIEVAL_PREFIXES for k in k_values] + ["mrr"]


def metric_names(k_values: Sequence[int] = DEFAULT_K_VALUES) -> list[str]:
    """Полный список метрик сплита в каноническом порядке."""
    return [
        *retrieval_metric_names(k_values),
        *GENERATION_METRICS,
        *ABSTENTION_METRICS,
        *LATENCY_METRICS,
        "n_questions",
    ]


@dataclass
class ExperimentResult:
    """Всё, что нужно знать об одном прогоне: конфиг, метрики, ответы по каждому вопросу.

    ``skipped`` — причина, по которой эксперимент не запускался (например, не установлены
    нейросетевые зависимости). Такой результат попадает в отчёт отдельным списком,
    чтобы было видно, что эксперимент не забыт, а пропущен осознанно.
    """

    name: str
    config: dict
    metrics: dict[str, dict]
    per_question: list[dict]
    threshold: float | None
    index_stats: dict
    skipped: str | None = None

    @property
    def is_skipped(self) -> bool:
        return self.skipped is not None

    def metric(self, name: str, split: str = "test") -> float | None:
        """Значение метрики на сплите (None, если её нет)."""
        return self.metrics.get(split, {}).get(name)

    def to_dict(self) -> dict[str, Any]:
        """Глубокая копия в виде словаря — готова к ``json.dumps``."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentResult:
        """Обратное к ``to_dict``. Неизвестные ключи игнорируются (совместимость со старыми отчётами)."""
        if "name" not in data:
            raise ValueError("в результате эксперимента нет поля name")
        return cls(
            name=str(data["name"]),
            config=dict(data.get("config") or {}),
            metrics=dict(data.get("metrics") or {}),
            per_question=list(data.get("per_question") or []),
            threshold=data.get("threshold"),
            index_stats=dict(data.get("index_stats") or {}),
            skipped=data.get("skipped"),
        )

    @classmethod
    def for_skipped(cls, name: str, reason: str, config: dict | None = None) -> ExperimentResult:
        """Результат-заглушка для эксперимента, который не запускался."""
        return cls(
            name=name,
            config=dict(config or {}),
            metrics={},
            per_question=[],
            threshold=None,
            index_stats={},
            skipped=reason,
        )
