"""Классификатор отказа: решает, отвечать ли на вопрос, по признакам выдачи поиска.

Пороговый метод (``abstention.method: threshold``) отказывается, если лучший score одного
источника ниже порога. Классификатор (``abstention.method: classifier``) смотрит на два десятка
признаков сразу (``features``), а решение принимает обученная модель (``models``): логистическая
регрессия или CatBoost. Обучение, кросс-валидация и подбор порога — ``train`` (``kbrag train-abstention``).

``train`` здесь не реэкспортируется: он тянет за собой пайплайн и sklearn.model_selection,
а сервису нужны только признаки и загрузка модели.
"""

from kbrag.abstention.features import (
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    extract_features,
    features_to_matrix,
)
from kbrag.abstention.models import (
    MODEL_TYPES,
    AbstentionModel,
    LogRegParams,
    catboost_available,
    load_abstention_model,
    make_estimator,
)

__all__ = [
    "FEATURE_DESCRIPTIONS",
    "FEATURE_NAMES",
    "MODEL_TYPES",
    "AbstentionModel",
    "LogRegParams",
    "catboost_available",
    "extract_features",
    "features_to_matrix",
    "load_abstention_model",
    "make_estimator",
]
