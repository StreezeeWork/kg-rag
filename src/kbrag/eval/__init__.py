"""Оценка качества: эталонный набор, метрики, LLM-судья, результаты и отчёт.

``runner`` (прогон экспериментов) тянет за собой весь пайплайн (поиск, sklearn, генерацию),
а метрики и датасет должны импортироваться быстро и без лишних зависимостей. Поэтому
``run_experiment`` и ``run_grid`` экспортируются **лениво**: ``from kbrag.eval import run_grid``
работает, но модуль ``runner`` загружается только в момент первого обращения к этим именам.
"""

from typing import Any

from kbrag.eval.dataset import (
    QUESTION_TYPES,
    SPLITS,
    Evidence,
    GoldItem,
    Span,
    assign_splits,
    evidence_spans,
    is_relevant,
    load_gold,
    save_gold,
    validate_gold,
)
from kbrag.eval.judge import JudgeError, JudgeVerdict, LLMJudge, cohen_kappa, parse_verdict
from kbrag.eval.metrics import (
    abstention_report,
    choose_threshold,
    citation_precision,
    citation_recall,
    coverage_matrix,
    fact_recall,
    hit_at_k,
    mean_ignore_none,
    mrr,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    retrieval_metrics,
    token_f1,
)
from kbrag.eval.report import load_results, render_leaderboard, write_report
from kbrag.eval.results import ExperimentResult, metric_names, retrieval_metric_names

__all__ = [
    "QUESTION_TYPES",
    "SPLITS",
    "Evidence",
    "ExperimentResult",
    "GoldItem",
    "JudgeError",
    "JudgeVerdict",
    "LLMJudge",
    "Span",
    "abstention_report",
    "assign_splits",
    "choose_threshold",
    "citation_precision",
    "citation_recall",
    "cohen_kappa",
    "coverage_matrix",
    "evidence_spans",
    "fact_recall",
    "hit_at_k",
    "is_relevant",
    "load_gold",
    "load_results",
    "mean_ignore_none",
    "metric_names",
    "mrr",
    "ndcg_at_k",
    "parse_verdict",
    "percentile",
    "precision_at_k",
    "recall_at_k",
    "render_leaderboard",
    "retrieval_metric_names",
    "retrieval_metrics",
    "run_experiment",
    "run_grid",
    "save_gold",
    "token_f1",
    "validate_gold",
    "write_report",
]

# Имена, которые отдаются лениво из kbrag.eval.runner (см. docstring модуля).
_LAZY_RUNNER_EXPORTS = ("run_experiment", "run_grid")


def __getattr__(name: str) -> Any:
    # PEP 562: Python вызывает эту функцию, только если обычного атрибута с таким именем нет.
    if name in _LAZY_RUNNER_EXPORTS:
        from kbrag.eval import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
