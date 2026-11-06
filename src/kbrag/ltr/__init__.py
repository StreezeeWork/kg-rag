"""Learning-to-rank: обученная модель задаёт финальный порядок кандидатов вместо одного score реранкера.

Без LTR порядок выдачи определяет cross-encoder. С LTR (``ltr.enabled: true``) модель CatBoost
смотрит на два десятка признаков кандидата сразу (``features``): score и ранги BM25, dense и реранкера,
сколько кусков того же документа уже выше, покрытие слов вопроса. Файл модели и правила её
применения — ``models``; обучение, кросс-валидация по вопросам и сравнение на dev — ``train``
(``kbrag train-ranker``).

``train`` здесь не реэкспортируется: он тянет за собой пайплайн, а сервису нужны только признаки
и загрузка модели.
"""

from kbrag.ltr.features import FEATURE_DESCRIPTIONS, FEATURE_NAMES, candidate_features
from kbrag.ltr.models import (
    LOSSES,
    RankerModel,
    candidates_fingerprint,
    load_ranker,
    make_ranker,
    ranker_identity,
)

__all__ = [
    "FEATURE_DESCRIPTIONS",
    "FEATURE_NAMES",
    "LOSSES",
    "RankerModel",
    "candidate_features",
    "candidates_fingerprint",
    "load_ranker",
    "make_ranker",
    "ranker_identity",
]
