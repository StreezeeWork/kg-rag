"""Модель ранжирования: CatBoost-ранкер, его файл и правила, когда сервис может им пользоваться.

Файл модели — пара ``reports/ltr_*.json`` + ``.cbm`` рядом (деревья CatBoost в его собственном формате).
Pickle не используем: распаковка pickle исполняет произвольный код, а JSON можно прочитать глазами.
В JSON лежит «паспорт»: для каких настроек поиска (``fingerprint``) и какого корпуса модель обучена,
на каких вопросах, с какими результатами CV и dev.

Чем это отличается от модели отказа (``abstention.models``): та при неподходящем файле откатывается
на порог, а здесь неподходящий файл — ошибка при старте (``load_ranker`` бросает ``ValueError``).
Порядок кандидатов — это вход для всего, что дальше: признаков модели отказа, порога по уверенности,
генератора. Тихо подменить порядок значило бы применять их к выдаче, на которой их не обучали.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kbrag.abstention.models import import_catboost
from kbrag.config import AppConfig, config_fingerprint
from kbrag.ltr.features import FEATURE_NAMES, candidate_features
from kbrag.types import ScoredChunk

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
# Функции потерь, которые перебирает обучение (``ltr.train``). Обе попарные: модель учится тому,
# что релевантный фрагмент должен стоять выше нерелевантного в том же вопросе, а не угадывать
# «абсолютную релевантность» (у разных вопросов она в разной шкале).
LOSSES = ("YetiRank", "PairLogit")
# Ключ score модели в ScoredChunk.details (и source итоговой выдачи).
LTR_SCORE_KEY = "ltr"
LTR_HINT = "обучите её: kbrag train-ranker --config <конфиг>"

# Общие гиперпараметры (перебираются только функция потерь и глубина, см. train.py). Как у модели
# отказа: данных мало (сотни вопросов), поэтому много маленьких шагов и не мусорить catboost_info/.
CATBOOST_RANKER_DEFAULTS: dict[str, Any] = {
    "iterations": 400,
    "learning_rate": 0.05,
    "random_seed": 0,
    "verbose": False,
    "allow_writing_files": False,
}
_STRUCTURE_KEYS = frozenset({"format_version", "feature_names", "params", "catboost"})


def candidates_fingerprint(cfg: AppConfig) -> str:
    """Отпечаток настроек, от которых зависят кандидаты и их признаки: нарезка, поиск, реранкер."""
    return config_fingerprint(cfg, "chunking", "retrieval", "rerank")


def make_ranker(params: Mapping[str, Any] | None = None) -> Any:
    """Необученный ``CatBoostRanker``: ``params`` — например ``{"loss_function": "PairLogit", "depth": 3}``."""
    return import_catboost().CatBoostRanker(**{**CATBOOST_RANKER_DEFAULTS, **dict(params or {})})


class RankerModel:
    """Обученная модель ранжирования вместе с «паспортом» (``metadata``)."""

    def __init__(
        self,
        catboost_model: Any,
        *,
        params: Mapping[str, Any] | None = None,
        feature_names: Sequence[str] = FEATURE_NAMES,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.params = dict(params or {})
        self.feature_names = tuple(feature_names)
        self.metadata = dict(metadata or {})
        self._model = catboost_model

    @property
    def name(self) -> str:
        """Короткое имя для логов и ``/health``: ``catboost:PairLogit-d3``."""
        return f"catboost:{self.params.get('loss_function', '?')}-d{self.params.get('depth', '?')}"

    @property
    def fingerprint(self) -> str | None:
        return self.metadata.get("fingerprint")

    @property
    def corpus_sha256(self) -> str | None:
        return self.metadata.get("corpus_sha256")

    # --- предсказание ---

    def score(self, matrix: np.ndarray) -> np.ndarray:
        """Score модели для строк матрицы признаков: больше — выше в выдаче. Шкала имеет смысл только внутри вопроса."""
        if matrix.shape[0] == 0:
            return np.zeros(0)
        return np.asarray(self._model.predict(matrix), dtype=np.float64)

    def rank(
        self, question: str, candidates: Sequence[ScoredChunk], *, with_header: bool = True
    ) -> list[ScoredChunk]:
        """Новый порядок всех кандидатов по score модели (при равенстве — в порядке, в котором пришли).

        Tie-break тот же, что при обучении и выборе модели (``ltr.train.order_by_scores``): стабильная
        сортировка сохраняет порядок реранкера среди равных score.

        Прежние детали (bm25, dense, rerank, ранги) переносятся, score модели кладётся в ``details["ltr"]``,
        а прежний score — в ``details["pre_ltr"]``: так видно, как модель изменила порядок.
        """
        if not candidates:
            return []
        scores = self.score(candidate_features(question, candidates, with_header=with_header))
        ranked = [
            ScoredChunk(
                chunk=item.chunk,
                score=float(score),
                source=LTR_SCORE_KEY,
                details={**item.details, LTR_SCORE_KEY: float(score), "pre_ltr": float(item.score)},
            )
            for item, score in zip(candidates, scores, strict=True)
        ]
        ranked.sort(key=lambda item: -item.score)  # sort в Python стабильная
        return ranked

    # --- файл ---

    def save(self, path: str | Path) -> Path:
        """Пишет JSON и ``.cbm`` рядом; ``sha256`` деревьев в JSON связывает два файла одного обучения."""
        target = Path(path)
        if target.suffix.lower() != ".json":
            # Иначе with_suffix(".cbm") совпал бы с самим JSON (model.cbm), и деревья затёрли бы паспорт.
            raise ValueError(f"файл модели должен оканчиваться на .json, а не {target.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        cbm_path = target.with_suffix(".cbm")
        self._model.save_model(str(cbm_path), format="cbm")
        payload = {
            "format_version": FORMAT_VERSION,
            "feature_names": list(self.feature_names),
            "params": self.params,
            "catboost": {"file": cbm_path.name, "sha256": file_sha256(cbm_path)},
            **{key: value for key, value in self.metadata.items() if key not in _STRUCTURE_KEYS},
        }
        text = json.dumps(_plain(payload), ensure_ascii=False, indent=2, allow_nan=False)
        target.write_text(text + "\n", encoding="utf-8", newline="\n")
        logger.info("Модель ранжирования %s сохранена в %s", self.name, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> RankerModel:
        """Читает модель; битый или чужой файл → ``ValueError`` (нет catboost → ``ImportError``)."""
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("файл модели ранжирования должен содержать JSON-объект")
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"версия формата {data.get('format_version')!r}, а этот код читает {FORMAT_VERSION}"
            )
        info = data.get("catboost") or {}
        params = data.get("params") or {}
        names = data.get("feature_names") or []
        if not isinstance(info, dict) or not isinstance(params, dict) or not isinstance(names, list):
            raise ValueError("поля catboost и params должны быть объектами, а feature_names — списком")
        # Только имя файла, без каталогов: JSON не должен указывать на произвольный путь на диске.
        cbm_path = source.parent / Path(str(info.get("file", ""))).name
        if not cbm_path.is_file():
            raise ValueError(f"нет файла {cbm_path} с деревьями CatBoost")
        if info.get("sha256") != file_sha256(cbm_path):
            raise ValueError(f"{cbm_path} не совпадает с JSON-метаданными (sha256): файлы из разных обучений")
        model = import_catboost().CatBoostRanker()
        try:
            model.load_model(str(cbm_path), format="cbm")
        except Exception as exc:  # CatBoostError не наследует ValueError, а битый файл — та же «ошибка файла»
            raise ValueError(f"не удалось прочитать деревья CatBoost из {cbm_path}: {exc}") from exc
        return cls(
            model,
            params=params,
            feature_names=[str(name) for name in names],
            metadata={key: value for key, value in data.items() if key not in _STRUCTURE_KEYS},
        )


def load_ranker(cfg: AppConfig, corpus_sha256: str | None = None) -> RankerModel:
    """Модель ранжирования для ``ltr.enabled: true``; любая причина не применять её — ``ValueError``.

    Причины: нет файла; файл битый; модель обучена для других настроек поиска (``fingerprint``),
    на другом корпусе (``corpus_sha256``) или для другого списка признаков (код признаков поменялся).
    """
    raw_path = cfg.ltr.model_path
    path = Path(raw_path) if raw_path else None
    if path is None or not path.is_file():
        raise ValueError(f"ltr.enabled: true, но файла модели ранжирования {path} нет — {LTR_HINT}")
    try:
        model = RankerModel.load(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"не удалось прочитать модель ранжирования {path} ({exc}) — {LTR_HINT}") from exc
    expected = candidates_fingerprint(cfg)
    if model.fingerprint != expected:
        raise ValueError(
            f"модель ранжирования {path} обучена для других настроек поиска "
            f"(fingerprint {model.fingerprint}, нужен {expected}) — {LTR_HINT}"
        )
    if model.feature_names != FEATURE_NAMES:
        raise ValueError(f"модель ранжирования {path} обучена на другом наборе признаков — {LTR_HINT}")
    if corpus_sha256 is not None and model.corpus_sha256 not in (None, corpus_sha256):
        raise ValueError(
            f"модель ранжирования {path} обучена на другом корпусе "
            f"(corpus_sha256 {model.corpus_sha256}, сейчас {corpus_sha256}) — {LTR_HINT}"
        )
    logger.info("Модель ранжирования %s загружена из %s", model.name, path)
    return model


def ranker_identity(cfg: AppConfig) -> str:
    """sha256 JSON-файла модели ранжирования (``"missing"``, если файла нет).

    Входит в ``pipeline.calibration_fingerprint`` при ``ltr.enabled``: порог и модель отказа,
    подобранные на выдаче одной модели ранжирования, не применяются к выдаче переобученной.
    JSON хранит sha256 деревьев, поэтому его хэш меняется при любом переобучении.
    """
    raw_path = cfg.ltr.model_path
    path = Path(raw_path) if raw_path else None
    return file_sha256(path) if path is not None and path.is_file() else "missing"


def file_sha256(path: Path) -> str:
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
