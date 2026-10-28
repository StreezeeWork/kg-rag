"""Обучение классификатора отказа: признаки → кросс-валидация → финальная модель → порог на dev.

Схема (``train_abstention``, команда ``kbrag train-abstention``)::

    train.jsonl ──► признаки ──► CV на train: бейзлайн, logreg (C), catboost (depth) ──► лучший кандидат
                                                                                        │
    gold.jsonl ──► только dev ──► признаки ──► порог решения по balanced accuracy ◄── обучение на всём train
                                                  │
                                                  └──► отчёт на dev + файл модели (reports/abstention_*.json)

Почему данные разделены именно так — это главное, что нужно понять в модуле:

* **train** (``data/eval/train.jsonl``) — только для обучения и выбора модели (кросс-валидация).
  Это отдельный файл, а не часть gold: на gold систему оценивают.
* **dev** (часть gold) — только для порога решения. Порог — тоже параметр, подобранный по данным;
  подбирать его на тех же вопросах, на которых училась модель, значит получить оптимистичный порог
  (на train модель увереннее, чем на новых вопросах). На dev же модель сравнивается с бейзлайном
  (нынешним порогом): train может быть устроен иначе, чем эталон, и выигрыш на train-CV ничего
  не доказывает (``TrainReport.beats_baseline_dev``).
* **test** (часть gold) — сюда не попадает вообще. Его честно меряет ``kbrag eval`` уже готовой
  модели с готовым порогом. ``fit_abstention`` получает train и dev явными списками и проверяет их
  ``split`` — test туда передать нельзя даже по ошибке.

Почему кросс-валидация, а не одно разбиение: обучающих вопросов сотни, а без ответа из них
десятки. Одна валидационная часть из ~20 «неотвечаемых» даёт очень шумную оценку; 5 фолдов × 3
повтора усредняют этот шум, а разброс по фолдам (± std) показывает, насколько ему верить.
Все кандидаты проверяются на **одних и тех же** фолдах — сравнение парное и честное.
"""

from __future__ import annotations

import importlib.metadata
import logging
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from kbrag.abstention.features import FEATURE_NAMES, extract_features, features_to_matrix
from kbrag.abstention.models import AbstentionModel, catboost_available, import_catboost, make_estimator
from kbrag.config import AppConfig
from kbrag.eval.dataset import (
    TRAIN_SPLIT,
    GoldItem,
    items_fingerprint,
    load_gold,
    train_gold_conflicts,
    validate_gold,
)
from kbrag.eval.metrics import abstention_report, auroc, auroc_ci, auroc_diff_ci, choose_threshold
from kbrag.ingest import corpus_fingerprint, load_corpus
from kbrag.pipeline import RAGPipeline, calibration_fingerprint
from kbrag.types import Document

logger = logging.getLogger(__name__)

MODEL_CHOICES = ("auto", "logreg", "catboost")
CV_SPLITS = 5
CV_REPEATS = 3
# Сетки гиперпараметров намеренно крошечные: на сотнях примеров широкий перебор сам становится
# подгонкой под фолды. C — сила регуляризации logreg (меньше — проще модель), depth — глубина деревьев.
LOGREG_C_VALUES = (0.1, 1.0, 10.0)
CATBOOST_DEPTHS = (2, 3, 4)
# Средние ROC-AUC, отличающиеся меньше этого, считаем ничьей — и берём более простую модель.
TIE_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Признаки
# ---------------------------------------------------------------------------


def feature_table(
    pipeline: RAGPipeline, items: Sequence[GoldItem]
) -> tuple[list[dict[str, float]], np.ndarray, list[str]]:
    """Признаки, метки и id вопросов: ``(rows, y, qids)``, ``y = 1`` — в базе нет ответа.

    Почему класс 1 — «нет ответа»: модель предсказывает вероятность события, ради которого
    существует (отказаться), а PR-AUC считается по редкому, интересному классу.
    Фрагменты — ``pipeline.retrieve(вопрос, top_k)``: ровно те, что сервис отдаёт генератору.
    """
    top_k = pipeline.cfg.retrieval.top_k
    rows: list[dict[str, float]] = []
    for item in items:
        contexts, _ = pipeline.retrieve(item.question, top_k=top_k)
        rows.append(extract_features(item.question, contexts))
    y = np.asarray([0 if item.answerable else 1 for item in items], dtype=int)
    return rows, y, [item.qid for item in items]


def baseline_scores(rows: Sequence[Mapping[str, float]], feature: str) -> np.ndarray:
    """Бейзлайн «один признак» в шкале классификатора: больше — вероятнее «нет ответа».

    Это нынешний пороговый метод: уверенность одного источника (например, ``dense_top1``) со знаком
    минус. NaN (у поиска нет оценки) — выше всех: пороговый метод в этом случае тоже отказывается
    («ниже любого порога», см. ``metrics.choose_threshold``). Вместо бесконечности — максимум + 1:
    функции sklearn не принимают inf, а на ранговые метрики конкретное число не влияет.
    """
    values = -features_to_matrix(rows, [feature])[:, 0]
    finite = values[np.isfinite(values)]
    fill = float(finite.max()) + 1.0 if finite.size else 0.0
    return np.where(np.isnan(values), fill, values)


# ---------------------------------------------------------------------------
# Кросс-валидация и выбор модели
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """Кандидат на роль модели отказа.

    ``complexity`` — место в списке ``make_candidates`` (меньше — проще). ``choose_candidate``
    перебирает кандидатов по возрастанию ``complexity`` и при ничьей оставляет более простого.
    """

    name: str
    model_type: str  # "baseline" | "logreg" | "catboost"
    params: dict[str, Any] = field(default_factory=dict)
    complexity: int = 0


def make_candidates(model: str = "auto", baseline_feature: str | None = None) -> list[Candidate]:
    """Список кандидатов от простого к сложному.

    Бейзлайн обучать не нужно, он только точка отсчёта: модель, которая его не обгоняет, не стоит
    своей сложности. Без catboost режим ``auto`` сравнивает только logreg (с предупреждением),
    а явный ``--model catboost`` падает с подсказкой, как его установить.
    """
    if model not in MODEL_CHOICES:
        raise ValueError(f"model должен быть одним из {MODEL_CHOICES}, получено {model!r}")
    candidates: list[Candidate] = []
    if baseline_feature is not None:
        candidates.append(
            Candidate(f"baseline: {baseline_feature}", "baseline", {"feature": baseline_feature})
        )
    if model in ("auto", "logreg"):
        candidates += [Candidate(f"logreg C={c:g}", "logreg", {"C": c}) for c in LOGREG_C_VALUES]
    if model == "catboost":
        import_catboost()  # нет пакета → ImportError с «pip install -e .[classic]»
    if model == "catboost" or (model == "auto" and catboost_available()):
        candidates += [Candidate(f"catboost depth={d}", "catboost", {"depth": d}) for d in CATBOOST_DEPTHS]
    elif model == "auto":
        logger.warning("catboost не установлен: сравниваю только logreg (pip install -e .[classic])")
    # Порядок списка и есть порядок сложности: бейзлайн < logreg (C по возрастанию) < catboost (depth).
    return [Candidate(c.name, c.model_type, c.params, complexity=i) for i, c in enumerate(candidates)]


def cross_validate(
    rows: Sequence[Mapping[str, float]],
    y: Sequence[int],
    seed: int = 0,
    model: str = "auto",
    baseline_feature: str | None = None,
) -> list[dict[str, Any]]:
    """ROC-AUC и PR-AUC (mean ± std по фолдам) для каждого кандидата: RepeatedStratifiedKFold 5 × 3.

    * Стратификация — в каждом фолде та же доля вопросов без ответа, что во всём train
      (иначе в маленьком фолде их могло бы не оказаться вовсе).
    * ROC-AUC не зависит от порога и от доли классов — им выбираем модель.
    * PR-AUC (average precision) по классу «нет ответа» показывает, насколько чисты отказы;
      его «случайный» уровень — доля этого класса, а не 0.5.
    * ``std`` по фолдам занижает настоящую неопределённость (фолды повторов пересекаются),
      поэтому разницу меньше одного std между кандидатами всерьёз не воспринимаем.

    Если вопросов без ответа меньше 5, фолдов столько, сколько их есть (минимум 2).
    ``seed`` задаёт и разбиение на фолды, и случайность CatBoost (``_seeded``).
    """
    matrix = features_to_matrix(rows)
    labels = np.asarray(y, dtype=int)
    folds = list(_folds(labels, seed))
    baseline = baseline_scores(rows, baseline_feature) if baseline_feature is not None else None
    results = []
    for candidate in make_candidates(model, baseline_feature):
        roc, pr = [], []
        for train_idx, valid_idx in folds:
            if candidate.model_type == "baseline":
                scores = baseline[valid_idx]
            else:
                estimator = make_estimator(
                    candidate.model_type, _seeded(candidate.model_type, candidate.params, seed)
                )
                estimator.fit(matrix[train_idx], labels[train_idx])
                scores = estimator.predict_proba(matrix[valid_idx])[:, 1]
            roc.append(roc_auc_score(labels[valid_idx], scores))
            pr.append(average_precision_score(labels[valid_idx], scores))
        results.append(
            {
                "name": candidate.name,
                "model_type": candidate.model_type,
                "params": dict(candidate.params),
                "complexity": candidate.complexity,
                "roc_auc_mean": float(np.mean(roc)),
                "roc_auc_std": float(np.std(roc, ddof=1)),
                "pr_auc_mean": float(np.mean(pr)),
                "pr_auc_std": float(np.std(pr, ddof=1)),
                "n_folds": len(folds),
            }
        )
        logger.info(
            "CV %s: ROC-AUC %.3f ± %.3f",
            candidate.name,
            results[-1]["roc_auc_mean"],
            results[-1]["roc_auc_std"],
        )
    return results


def _seeded(model_type: str, params: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Параметры для ``make_estimator`` с тем же ``seed``, что у фолдов.

    Зачем: ``seed`` обещает воспроизводимость всего обучения, а не только разбиения на фолды.
    У CatBoost своя случайность (подвыборки, порядок объектов), и без проброса она была бы
    всегда ``random_seed = 0`` из ``CATBOOST_DEFAULTS``. У logreg случайности нет — параметры как есть.
    """
    if model_type == "catboost":
        return {**params, "random_seed": seed}
    return dict(params)


def _folds(labels: np.ndarray, seed: int):
    n_minority = int(min(labels.sum(), len(labels) - labels.sum()))
    if n_minority < 2:
        raise ValueError(
            "для кросс-валидации нужно хотя бы по 2 вопроса каждого класса (с ответом и без) "
            f"в обучающем наборе, а меньший класс — {n_minority}"
        )
    n_splits = min(CV_SPLITS, n_minority)
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=CV_REPEATS, random_state=seed).split(
        np.zeros(len(labels)), labels
    )


def choose_candidate(cv_results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Лучший обучаемый кандидат по средней CV ROC-AUC; ничья → более простой.

    Бейзлайн в выборе не участвует: он и есть нынешний пороговый метод. Сравнение с ним —
    отдельный вопрос, и решается он на dev (``TrainReport.beats_baseline_dev``): если модель там
    не доказала преимущества, правильный вывод — оставить ``abstention.method: threshold``,
    а не «выбрать бейзлайн» внутри классификатора. Train-CV служит только выбору среди моделей.
    """
    # Сортировка по complexity, а не доверие порядку списка: правило «ничья → проще» не должно
    # зависеть от того, в каком порядке строки таблицы CV пришли в функцию.
    trainable = sorted(
        (row for row in cv_results if row["model_type"] != "baseline"), key=lambda row: row["complexity"]
    )
    if not trainable:
        raise ValueError("нет ни одного обучаемого кандидата")
    best = trainable[0]
    for row in trainable[1:]:  # по возрастанию сложности: заменяем, только если строго лучше
        if row["roc_auc_mean"] > best["roc_auc_mean"] + TIE_TOLERANCE:
            best = row
    return best


# ---------------------------------------------------------------------------
# Порог решения и метрики на dev
# ---------------------------------------------------------------------------


def tune_decision_threshold(probabilities: Sequence[float], y: Sequence[int]) -> float | None:
    """Порог для правила «отказ, если p ≥ порога», подобранный по balanced accuracy.

    Переиспользуем ``metrics.choose_threshold`` — тот же критерий, что у порогового метода, чтобы
    сравнение было честным. Но у него обратная шкала: он отвечает, если «уверенность» ≥ t. Поэтому
    уверенность классификатора = ``1 − p`` (больше — вероятнее, что ответ есть), и::

        ответить ⇔ 1 − p ≥ t ⇔ p ≤ 1 − t,     отказ ⇔ p > 1 − t.

    Возвращаем ``1 − t``. Строгое «>» против нашего «≥» различается только в точке ровно на
    пороге, а порог — середина между соседними значениями dev, поэтому ни один dev-вопрос
    на нём не лежит. Ничья у ``choose_threshold`` → меньший t → больший порог по p → реже отказ.
    None — если ни один порог не лучше «никогда не отказываться» (или на dev один класс).
    """
    answerable = [int(label) == 0 for label in y]
    threshold = choose_threshold([1.0 - float(p) for p in probabilities], answerable)
    return None if threshold is None else 1.0 - threshold


def decision_metrics(abstained: Sequence[bool], y: Sequence[int]) -> dict[str, float | None]:
    """Abstention precision/recall/F1, FAR (``metrics.abstention_report``) и balanced accuracy.

    Balanced accuracy = (доля пойманных вопросов без ответа + доля отвеченных с ответом) / 2 —
    тот самый критерий, по которому подбирается порог.
    """
    answerable = [int(label) == 0 for label in y]
    report = abstention_report(list(abstained), answerable)
    recall, far = report["abstain_recall"], report["false_abstention_rate"]
    balanced = None if recall is None or far is None else (recall + (1.0 - far)) / 2
    return {"balanced_accuracy": balanced, **report}


def evaluate_on_dev(
    probabilities: Sequence[float],
    baseline_confidence: Sequence[float],
    y: Sequence[int],
    decision_threshold: float | None,
) -> dict[str, dict[str, Any]]:
    """Классификатор против бейзлайна на dev: AUROC с 95% ДИ и метрики решения при своём пороге.

    У бейзлайна порог подбирается тем же ``choose_threshold`` на том же dev — иначе сравнение
    было бы нечестным. Помните: метрики решения тут **оптимистичны для обоих** (порог подобран
    на этих же вопросах); честная проверка — test в ``kbrag eval``. AUROC от порога не зависит.

    ``auroc_diff`` — разница AUROC «классификатор − бейзлайн» на одних и тех же dev-вопросах
    с парным 95% ДИ (``metrics.auroc_diff_ci``). По ней ``TrainReport.beats_baseline_dev``
    решает, стоит ли вообще переключаться с порога на классификатор.
    """
    answerable = [int(label) == 0 for label in y]
    p = [float(value) for value in probabilities]
    clf_confidence = [1.0 - value for value in p]
    classifier: dict[str, Any] = {
        "auroc": auroc(clf_confidence, answerable),
        "auroc_ci95": _as_list(auroc_ci(clf_confidence, answerable)),
        "threshold": decision_threshold,
    }
    if decision_threshold is not None:
        classifier.update(decision_metrics([value >= decision_threshold for value in p], y))

    confidence = [None if np.isnan(value) else float(value) for value in baseline_confidence]
    baseline_threshold = choose_threshold(confidence, answerable)
    baseline: dict[str, Any] = {
        "auroc": auroc(confidence, answerable),
        "auroc_ci95": _as_list(auroc_ci(confidence, answerable)),
        "threshold": baseline_threshold,
    }
    if baseline_threshold is not None:
        abstained = [value is None or value < baseline_threshold for value in confidence]
        baseline.update(decision_metrics(abstained, y))
    diff = auroc_diff_ci(clf_confidence, confidence, answerable)
    auroc_diff = {"value": diff[0], "ci95": [diff[1], diff[2]]} if diff is not None else {}
    return {"classifier": classifier, "baseline": baseline, "auroc_diff": auroc_diff}


def _as_list(interval: tuple[float, float] | None) -> list[float] | None:
    return list(interval) if interval is not None else None


# ---------------------------------------------------------------------------
# Интерпретация
# ---------------------------------------------------------------------------


def feature_importance(
    model_type: str, estimator: Any, matrix: np.ndarray
) -> tuple[str, list[tuple[str, float]]]:
    """Что модель считает важным: ``(вид оценки, [(признак, значение), …])`` по убыванию |значения|.

    * logreg — коэффициенты на **стандартизованных** признаках: их можно сравнивать между собой,
      знак показывает направление (плюс — признак толкает к отказу).
    * catboost — средний |SHAP| по обучающим вопросам: на сколько в среднем признак сдвигает
      предсказание (в логитах). Встроенная «важность» CatBoost считает частоту расщеплений
      и путает «часто использую» с «сильно влияю»; SHAP измеряет именно влияние.
    """
    if model_type == "logreg":
        kind, values = "coef", np.asarray(estimator.named_steps["clf"].coef_[0], dtype=np.float64)
    else:
        catboost = import_catboost()
        shap = estimator.get_feature_importance(data=catboost.Pool(matrix), type="ShapValues")
        # Последний столбец SHAP — базовое значение (ожидание модели), а не признак.
        kind, values = "mean_abs_shap", np.abs(np.asarray(shap)[:, :-1]).mean(axis=0)
    ranked = sorted(zip(FEATURE_NAMES, values.tolist(), strict=True), key=lambda pair: -abs(pair[1]))
    return kind, ranked


# ---------------------------------------------------------------------------
# Обучение целиком
# ---------------------------------------------------------------------------


@dataclass
class TrainReport:
    """Итог ``kbrag train-abstention``: таблица CV, выбор, порог, метрики dev, важность признаков."""

    model_type: str
    params: dict[str, Any]
    chosen: str
    baseline_feature: str
    cv: list[dict[str, Any]]
    decision_threshold: float | None
    dev: dict[str, dict[str, Any]]
    importance_kind: str
    importance: list[tuple[str, float]]
    n_train: int
    n_train_unanswerable: int
    n_dev: int
    n_dev_unanswerable: int
    model_path: str | None = None

    @property
    def beats_baseline_cv(self) -> bool:
        """Обгоняет ли выбранная модель бейзлайн по средней CV ROC-AUC **на train** (только справка).

        Решения по этому флагу не принимают: если train устроен иначе, чем эталон, бейзлайн на
        train может быть почти случайным (так и вышло в docs/results/abstention.md), и выигрыш
        у него ничего не говорит о качестве на эталоне. Для решения — ``beats_baseline_dev``.
        """
        scores = {row["name"]: row["roc_auc_mean"] for row in self.cv}
        baseline = scores.get(f"baseline: {self.baseline_feature}")
        return baseline is None or scores[self.chosen] > baseline + TIE_TOLERANCE

    @property
    def beats_baseline_dev(self) -> bool:
        """Доказала ли модель на dev, что разделяет классы лучше бейзлайна (нынешнего порога).

        True, только если нижняя граница 95% ДИ разницы AUROC «классификатор − бейзлайн» на dev
        больше 0. Бремя доказательства — на новом методе: он сложнее (≈20 признаков, файл модели,
        проверки свежести), поэтому «не хуже в пределах шума» — это ещё не повод переключаться.
        Сравнение именно на dev, а не на train-CV: dev — часть эталона, на котором систему
        оценивают, а train может быть устроен иначе.
        """
        interval = (self.dev.get("auroc_diff") or {}).get("ci95")
        return interval is not None and interval[0] > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "beats_baseline_cv": self.beats_baseline_cv,
            "beats_baseline_dev": self.beats_baseline_dev,
        }


def fit_abstention(
    pipeline: RAGPipeline,
    train: Sequence[GoldItem],
    dev: Sequence[GoldItem],
    model: str = "auto",
    seed: int = 0,
) -> tuple[AbstentionModel | None, TrainReport]:
    """Ядро обучения на готовом пайплайне: CV на train, финальная модель на всём train, порог на dev.

    train и dev передаются явно и проверяются по ``split``: test сюда попасть не может.
    Модель — None, если на dev не нашлось порога лучше «никогда не отказываться»: такую модель
    сервису отдавать незачем (отчёт при этом всё равно возвращается — его стоит прочитать).
    """
    _require_split(train, TRAIN_SPLIT, "обучающий набор")
    _require_split(dev, "dev", "набор для порога")

    rows_train, y_train, _ = feature_table(pipeline, train)
    rows_dev, y_dev, _ = feature_table(pipeline, dev)
    baseline_feature = f"{pipeline.score_source}_top1"

    cv = cross_validate(rows_train, y_train, seed=seed, model=model, baseline_feature=baseline_feature)
    best = choose_candidate(cv)
    matrix_train = features_to_matrix(rows_train)
    estimator = make_estimator(best["model_type"], _seeded(best["model_type"], best["params"], seed)).fit(
        matrix_train, y_train
    )

    # Предсказания dev — через ту же обёртку AbstentionModel, что в сервисе (для logreg — numpy-код из JSON).
    probe = AbstentionModel.from_estimator(best["model_type"], estimator, 0.5, params=best["params"])
    p_dev = probe.predict_proba(rows_dev)
    threshold = tune_decision_threshold(p_dev, y_dev)
    baseline_dev = features_to_matrix(rows_dev, [baseline_feature])[:, 0]
    kind, importance = feature_importance(best["model_type"], estimator, matrix_train)

    report = TrainReport(
        model_type=best["model_type"],
        params=dict(best["params"]),
        chosen=best["name"],
        baseline_feature=baseline_feature,
        cv=cv,
        decision_threshold=threshold,
        dev=evaluate_on_dev(p_dev, baseline_dev, y_dev, threshold),
        importance_kind=kind,
        importance=importance,
        n_train=len(train),
        n_train_unanswerable=int(y_train.sum()),
        n_dev=len(dev),
        n_dev_unanswerable=int(y_dev.sum()),
    )
    if threshold is None:
        logger.warning("Порог решения на dev не подобран: модель не лучше «никогда не отказываться»")
        return None, report
    model_obj = AbstentionModel.from_estimator(
        best["model_type"], estimator, threshold, params=best["params"]
    )
    return model_obj, report


def train_abstention(
    cfg: AppConfig,
    model: str = "auto",
    out: str | Path | None = None,
    *,
    docs: Sequence[Document] | None = None,
    train: Sequence[GoldItem] | None = None,
    gold: Sequence[GoldItem] | None = None,
    seed: int = 0,
) -> TrainReport:
    """``kbrag train-abstention``: обучает модель для конфига ``cfg`` и сохраняет её (если порог подобран).

    ``docs``, ``train``, ``gold`` можно передать готовыми (тесты); иначе они читаются из
    ``cfg.corpus_dir``, ``cfg.train_path``, ``cfg.gold_path``. Из gold в обучение идёт только dev;
    весь gold (вместе с test) нужен ещё для одной проверки — что обучающие вопросы его не повторяют.
    Путь сохранения — ``out`` или ``abstention.model_path``.
    """
    target = out if out is not None else cfg.abstention.model_path
    if not target:
        raise ValueError("не задан путь к модели: abstention.model_path в конфиге или --out")
    docs = list(docs) if docs is not None else load_corpus(cfg.corpus_dir)
    train = list(train) if train is not None else load_gold(cfg.train_path)
    gold = list(gold) if gold is not None else load_gold(cfg.gold_path)

    problems = validate_gold(train, docs, splits=(TRAIN_SPLIT,)) + train_gold_conflicts(train, gold)
    if problems:
        shown = "\n  ".join(problems[:10])
        more = f"\n  … и ещё {len(problems) - 10}" if len(problems) > 10 else ""
        raise ValueError(f"обучающий набор не годится ({len(problems)} проблем):\n  {shown}{more}")
    # Test отбрасываем сразу после чтения: дальше в обучение уходит только dev.
    dev = [item for item in gold if item.split == "dev"]

    # Признакам нужен только поиск. Порог и старую модель не грузим: они не влияют на выдачу,
    # а пайплайн с method: classifier попытался бы прочитать файл, который мы как раз обучаем.
    feature_cfg = cfg.model_copy(
        update={"abstention": cfg.abstention.model_copy(update={"method": "threshold", "threshold": None})}
    )
    pipeline = RAGPipeline.from_config(feature_cfg, docs=docs)
    model_obj, report = fit_abstention(pipeline, train, dev, model=model, seed=seed)
    if model_obj is None:
        return report

    model_obj.metadata.update(
        {
            "fingerprint": calibration_fingerprint(cfg),
            "corpus_sha256": corpus_fingerprint(list(docs)),
            "train_sha256": items_fingerprint(train),
            "dev_sha256": items_fingerprint(dev),
            "baseline_feature": report.baseline_feature,
            "cv_report": {
                "chosen": report.chosen,
                "beats_baseline_cv": report.beats_baseline_cv,
                "candidates": report.cv,
            },
            "dev_metrics": {**report.dev, "beats_baseline_dev": report.beats_baseline_dev},
            "importance": {"kind": report.importance_kind, "values": report.importance},
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "versions": _library_versions(report.model_type),
        }
    )
    report.model_path = str(model_obj.save(target))
    return report


def _require_split(items: Sequence[GoldItem], split: str, what: str) -> None:
    wrong = sorted({item.split for item in items} - {split})
    if wrong:
        raise ValueError(f"{what}: ожидаются только вопросы split={split!r}, а есть {wrong}")
    if not items:
        raise ValueError(f"{what} пуст")


def _library_versions(model_type: str) -> dict[str, str]:
    """Версии, с которыми обучена модель: при расхождении предсказаний первым делом смотрят сюда."""
    packages = ["kbrag", "numpy", "scikit-learn"] + (["catboost"] if model_type == "catboost" else [])
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions
