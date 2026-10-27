"""Модель отказа: обученный классификатор «в базе нет ответа», его порог и «паспорт».

Два вида моделей — намеренно разной природы, чтобы сравнение было содержательным:

* ``logreg`` — логистическая регрессия: медиана вместо NaN → стандартизация → линейная модель
  с ``class_weight="balanced"``. Простая, хорошо калибрует вероятности, а её веса читаются как
  «какой признак сдвигает решение и куда». Это сильный бейзлайн для маленьких табличных данных.
* ``catboost`` — градиентный бустинг на неглубоких деревьях. Ловит нелинейности и взаимодействия
  («низкий score важен, только если вопрос короткий»), сам обрабатывает NaN. Деревья мелкие
  (depth 2–4) и регуляризация сильная: обучающих вопросов сотни, глубокая модель их бы запомнила.

Сериализация **без pickle**. Pickle исполняет код при загрузке и ломается от смены версии
sklearn; файл модели же лежит в git, и его читает сервис. Поэтому:

* logreg хранится в JSON как числа (медианы, среднее/масштаб, коэффициенты, свободный член),
  а предсказывает собственная функция на numpy (``LogRegParams.predict_proba``) — тест сверяет
  её со sklearn до 1e-12;
* catboost хранится в родном формате ``.cbm`` рядом с JSON-метаданными (и с его sha256 в JSON:
  иначе можно подменить один файл, забыв про второй).

Сервис загружает модель через ``load_abstention_model`` — по образцу ``pipeline.load_calibration``:
модель, обученная для другого поиска, другого корпуса или другого набора признаков, не
применяется (предупреждение в лог, сервис откатывается на пороговый метод).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from kbrag.abstention.features import FEATURE_NAMES, features_to_matrix

if TYPE_CHECKING:
    from kbrag.config import AppConfig

logger = logging.getLogger(__name__)

MODEL_TYPES = ("logreg", "catboost")
# Версия формата файла модели: поменяли структуру JSON — увеличьте, и старые файлы не загрузятся молча.
FORMAT_VERSION = 1
CATBOOST_INSTALL_HINT = "pip install -e .[classic]"

# Гиперпараметры CatBoost, общие для всех кандидатов (перебирается только глубина, см. train.py).
# Почему такие: данных мало (сотни вопросов), поэтому много маленьких шагов (learning_rate 0.05)
# и сильная L2-регуляризация листьев; auto_class_weights — вопросов без ответа меньше, и без весов
# модель выгодно «никогда не отказываться»; allow_writing_files=False — не мусорить catboost_info/.
CATBOOST_DEFAULTS: dict[str, Any] = {
    "iterations": 400,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5.0,
    "auto_class_weights": "Balanced",
    "random_seed": 0,
    "verbose": False,
    "allow_writing_files": False,
}
# Ключи JSON, которые описывают саму модель; всё остальное в файле — метаданные («паспорт»).
_STRUCTURE_KEYS = frozenset(
    {"format_version", "model_type", "feature_names", "decision_threshold", "params", "logreg", "catboost"}
)


# ---------------------------------------------------------------------------
# Зависимости и создание моделей
# ---------------------------------------------------------------------------


def catboost_available() -> bool:
    """Установлен ли catboost (без импорта: сам импорт занимает заметное время)."""
    return importlib.util.find_spec("catboost") is not None


def import_catboost() -> Any:
    """Ленивый импорт catboost с понятной подсказкой: ядро проекта работает и без него."""
    try:
        import catboost
    except ImportError as exc:
        raise ImportError(
            f"для модели отказа catboost нужен пакет catboost: {CATBOOST_INSTALL_HINT}"
        ) from exc
    return catboost


def make_estimator(model_type: str, params: Mapping[str, Any] | None = None) -> Any:
    """Необученная модель с интерфейсом sklearn (``fit`` / ``predict_proba``) — для CV и финального обучения.

    ``params``: для ``logreg`` — ``{"C": 1.0}`` (сила регуляризации: меньше C — проще модель),
    для ``catboost`` — ``{"depth": 3}`` (и при желании любые параметры CatBoost поверх ``CATBOOST_DEFAULTS``).
    """
    params = dict(params or {})
    if model_type == "logreg":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                # Медиана, а не среднее: score с длинным хвостом (BM25) сдвигают среднее.
                # keep_empty_features: признак, который в этом конфиге всегда NaN (rerank_* без
                # реранкера), остаётся столбцом из нулей — иначе sklearn выбросил бы его, и номера
                # столбцов разъехались бы с FEATURE_NAMES.
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                # Стандартизация нужна регуляризации (иначе штраф зависит от единиц признака)
                # и делает коэффициенты сравнимыми между собой.
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(C=float(params.get("C", 1.0)), class_weight="balanced", max_iter=1000),
                ),
            ]
        )
    if model_type == "catboost":
        catboost = import_catboost()
        return catboost.CatBoostClassifier(**{**CATBOOST_DEFAULTS, **params})
    raise ValueError(f"неизвестный тип модели отказа {model_type!r}; допустимы {MODEL_TYPES}")


# ---------------------------------------------------------------------------
# Логистическая регрессия без sklearn в рантайме
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogRegParams:
    """Всё, что нужно для предсказания обученного sklearn-пайплайна logreg, — пять массивов чисел.

    Предсказание повторяет sklearn шаг в шаг: NaN → медиана, ``(x − mean) / scale``,
    ``σ(x · coef + intercept)``. Так файл модели — обычный JSON, который можно прочитать глазами
    и загрузить без sklearn и без pickle.
    """

    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    coef: np.ndarray
    intercept: float

    @classmethod
    def from_pipeline(cls, pipeline: Any) -> LogRegParams:
        """Достаёт числа из обученного ``make_estimator("logreg")``."""
        imputer, scaler, clf = (pipeline.named_steps[name] for name in ("impute", "scale", "clf"))
        return cls(
            medians=np.asarray(imputer.statistics_, dtype=np.float64),
            means=np.asarray(scaler.mean_, dtype=np.float64),
            scales=np.asarray(scaler.scale_, dtype=np.float64),
            coef=np.asarray(clf.coef_[0], dtype=np.float64),
            intercept=float(clf.intercept_[0]),
        )

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        """P(класс 1 = «в базе нет ответа») для каждой строки матрицы признаков."""
        filled = np.where(np.isnan(matrix), self.medians, matrix)
        logits = ((filled - self.means) / self.scales) @ self.coef + self.intercept
        # Сигмоида через tanh численно устойчива при больших |logit| (exp не переполняется).
        return 0.5 * (1.0 + np.tanh(0.5 * logits))

    def to_dict(self) -> dict[str, Any]:
        return {
            "medians": self.medians.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], n_features: int) -> LogRegParams:
        arrays = {
            key: np.asarray(data[key], dtype=np.float64) for key in ("medians", "means", "scales", "coef")
        }
        if any(array.shape != (n_features,) for array in arrays.values()):
            raise ValueError(f"в параметрах logreg ожидается по {n_features} чисел на признак")
        return cls(**arrays, intercept=float(data["intercept"]))


# ---------------------------------------------------------------------------
# Модель отказа
# ---------------------------------------------------------------------------


class AbstentionModel:
    """Обученный классификатор + порог решения + метаданные (откуда модель и насколько она хороша).

    Решение: **отказ, если** ``p >= decision_threshold``, где ``p`` — вероятность того, что
    в базе нет ответа (``predict_proba``). Порог подбирается на dev (``train.tune_decision_threshold``)
    и хранится в файле модели: оценка и сервис применяют один и тот же порог, не подбирая его заново.

    ``metadata`` — «паспорт»: ``fingerprint`` (настройки поиска, ``pipeline.calibration_fingerprint``),
    ``corpus_sha256``, ``train_sha256``, ``cv_report``, ``dev_metrics``, ``created_at``, ``versions``.
    Сервис по ним решает, можно ли применять модель; человек — откуда она взялась.
    """

    def __init__(
        self,
        model_type: str,
        decision_threshold: float,
        *,
        params: Mapping[str, Any] | None = None,
        logreg: LogRegParams | None = None,
        catboost_model: Any = None,
        feature_names: Sequence[str] = FEATURE_NAMES,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if model_type not in MODEL_TYPES:
            raise ValueError(f"неизвестный тип модели отказа {model_type!r}; допустимы {MODEL_TYPES}")
        if (model_type == "logreg") != (logreg is not None) or (model_type == "catboost") != (
            catboost_model is not None
        ):
            raise ValueError(f"для model_type={model_type!r} нужны параметры именно этой модели")
        self.model_type = model_type
        self.decision_threshold = float(decision_threshold)
        self.params = dict(params or {})
        self.feature_names = tuple(feature_names)
        self.metadata = dict(metadata or {})
        self._logreg = logreg
        self._catboost = catboost_model

    @classmethod
    def from_estimator(
        cls,
        model_type: str,
        estimator: Any,
        decision_threshold: float,
        params: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AbstentionModel:
        """Обёртка над обученной ``make_estimator(...)``: logreg сразу превращается в числа."""
        if model_type == "logreg":
            return cls(
                model_type,
                decision_threshold,
                params=params,
                logreg=LogRegParams.from_pipeline(estimator),
                metadata=metadata,
            )
        return cls(model_type, decision_threshold, params=params, catboost_model=estimator, metadata=metadata)

    # --- предсказание ---

    def predict_proba(self, rows: Sequence[Mapping[str, float]] | np.ndarray) -> np.ndarray:
        """P(в базе нет ответа) для каждого вопроса; ``rows`` — словари из ``extract_features`` или матрица."""
        matrix = rows if isinstance(rows, np.ndarray) else features_to_matrix(rows, self.feature_names)
        if matrix.shape[0] == 0:
            return np.zeros(0)
        if self._logreg is not None:
            return self._logreg.predict_proba(matrix)
        return np.asarray(self._catboost.predict_proba(matrix)[:, 1], dtype=np.float64)

    def predict_one(self, features: Mapping[str, float]) -> float:
        """То же для одного вопроса — так зовёт модель пайплайн сервиса."""
        return float(self.predict_proba([features])[0])

    def abstains(self, probability: float) -> bool:
        """Правило решения: отказ, если ``p >= decision_threshold``."""
        return probability >= self.decision_threshold

    @property
    def fingerprint(self) -> str | None:
        return self.metadata.get("fingerprint")

    @property
    def corpus_sha256(self) -> str | None:
        return self.metadata.get("corpus_sha256")

    # --- файл ---

    def save(self, path: str | Path) -> Path:
        """Пишет JSON (и ``.cbm`` рядом для catboost). Формат файла задаётся только здесь и в ``load``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "model_type": self.model_type,
            "feature_names": list(self.feature_names),
            "decision_threshold": self.decision_threshold,
            "params": self.params,
            **{key: value for key, value in self.metadata.items() if key not in _STRUCTURE_KEYS},
        }
        if self._logreg is not None:
            payload["logreg"] = self._logreg.to_dict()
        else:
            cbm_path = target.with_suffix(".cbm")
            self._catboost.save_model(str(cbm_path), format="cbm")
            payload["catboost"] = {"file": cbm_path.name, "sha256": _file_sha256(cbm_path)}
        # allow_nan=False: NaN/Infinity — не JSON, другой парсер такой файл не прочитает. Лучше упасть здесь.
        text = json.dumps(_plain(payload), ensure_ascii=False, indent=2, allow_nan=False)
        target.write_text(text + "\n", encoding="utf-8", newline="\n")
        logger.info("Модель отказа %s сохранена в %s", self.model_type, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> AbstentionModel:
        """Читает модель из файла; битый или чужой файл → ``ValueError`` (нет catboost → ``ImportError``)."""
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("файл модели отказа должен содержать JSON-объект")
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"версия формата {data.get('format_version')!r}, а этот код читает {FORMAT_VERSION}"
            )
        model_type = data.get("model_type")
        feature_names = [str(name) for name in data.get("feature_names") or []]
        threshold = data.get("decision_threshold")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
        ):
            raise ValueError(f"decision_threshold должен быть числом, а не {threshold!r}")
        metadata = {key: value for key, value in data.items() if key not in _STRUCTURE_KEYS}
        common = {"params": data.get("params") or {}, "feature_names": feature_names, "metadata": metadata}

        if model_type == "logreg":
            logreg = LogRegParams.from_dict(data.get("logreg") or {}, n_features=len(feature_names))
            return cls("logreg", threshold, logreg=logreg, **common)
        if model_type == "catboost":
            info = data.get("catboost") or {}
            # Только имя файла, без каталогов: JSON не должен указывать на произвольный путь на диске.
            cbm_path = source.parent / Path(str(info.get("file", ""))).name
            if not cbm_path.is_file():
                raise ValueError(f"нет файла {cbm_path} с деревьями CatBoost")
            if info.get("sha256") != _file_sha256(cbm_path):
                raise ValueError(
                    f"{cbm_path} не совпадает с JSON-метаданными (sha256): файлы из разных обучений"
                )
            model = import_catboost().CatBoostClassifier()
            model.load_model(str(cbm_path), format="cbm")
            return cls("catboost", threshold, catboost_model=model, **common)
        raise ValueError(f"неизвестный тип модели {model_type!r}")


def load_abstention_model(cfg: AppConfig, corpus_sha256: str | None = None) -> AbstentionModel | None:
    """Модель отказа для сервиса; None (с предупреждением в лог), если её нельзя применять.

    Почему не исключение: так же устроена ``pipeline.load_calibration`` — без модели сервис
    продолжает работать на пороговом методе, а в логе видно, почему. Причины отказа от модели:
    нет файла; файл битый; модель обучена для других настроек поиска (``fingerprint``), на другом
    корпусе (``corpus_sha256``) или для другого списка признаков (код признаков поменялся).
    Исключение одно: модель catboost без установленного catboost — это ошибка установки,
    и о ней лучше узнать при старте (``ImportError`` с подсказкой), а не по логам.
    """
    # Импорт здесь, а не наверху модуля: kbrag.pipeline сам импортирует этот модуль, и импорт
    # на уровне модуля замкнул бы цикл pipeline → models → pipeline.
    from kbrag.pipeline import calibration_fingerprint

    raw_path = cfg.abstention.model_path
    path = Path(raw_path) if raw_path else None
    hint = "обучите её: kbrag train-abstention --config <конфиг>"
    if path is None or not path.is_file():
        logger.warning("Файл модели отказа %s не найден — %s", path, hint)
        return None
    try:
        model = AbstentionModel.load(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Не удалось прочитать модель отказа %s (%s) — %s", path, exc, hint)
        return None

    expected = calibration_fingerprint(cfg)
    if model.fingerprint != expected:
        logger.warning(
            "Модель отказа %s обучена для других настроек поиска (fingerprint %s, нужен %s) — %s",
            path,
            model.fingerprint,
            expected,
            hint,
        )
        return None
    if model.feature_names != FEATURE_NAMES:
        logger.warning("Модель отказа %s обучена на другом наборе признаков — %s", path, hint)
        return None
    if corpus_sha256 is not None and model.corpus_sha256 is not None and model.corpus_sha256 != corpus_sha256:
        logger.warning(
            "Модель отказа %s обучена на другом корпусе (corpus_sha256 %s, сейчас %s) — %s",
            path,
            model.corpus_sha256,
            corpus_sha256,
            hint,
        )
        return None
    logger.info(
        "Модель отказа %s загружена из %s (порог p ≥ %.4f)", model.model_type, path, model.decision_threshold
    )
    return model


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plain(value: Any) -> Any:
    """numpy-числа и кортежи → обычные типы JSON (json.dumps не знает np.float64 и np.int64)."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
