"""Прогон экспериментов: пайплайн отвечает на эталонные вопросы, мы считаем метрики.

Как устроен один эксперимент (``run_experiment``):

1. Собираем пайплайн по конфигу — **один индекс на эксперимент**, все вопросы идут через него.
2. Каждый вопрос прогоняем **без порога отказа** (``apply_threshold=False``) и запоминаем
   уверенность поиска. Так одного прогона хватает и для подбора порога, и для метрик.
3. Порог ``"auto"`` подбираем ``metrics.choose_threshold`` **только на dev**. Если бы в подборе
   участвовал test, метрики test были бы завышены: порог «подогнан» под те же вопросы,
   на которых его потом проверяют (утечка данных).
   С классификатором отказа (``abstention.method: classifier``) порог **не подбирается**: он уже
   подобран на dev при обучении (``kbrag train-abstention``) и лежит в файле модели. Без файла
   модели прогон падает с подсказкой — молча оценить вместо классификатора порог было бы обманом.
   Падает он и тогда, когда dev с момента обучения изменился (``dev_sha256`` модели): иначе
   вопрос, на котором подбирали порог, мог оказаться в test.
4. Порог применяем постфактум ко всем вопросам: ответ считается отказом, если отказался
   генератор ИЛИ уверенность ниже порога (у классификатора — P(нет ответа) ≥ порога модели).
   Для метрик такой ответ заменяется на ``ABSTAIN_TEXT``, ссылки обнуляются — ровно то,
   что вернул бы сервис.
5. Считаем метрики по каждому вопросу и усредняем по сплитам ``dev`` / ``test`` / ``all``
   и по типам вопросов (для сплита отчёта).

Сетка экспериментов (``run_grid``) читается из YAML::

    base: default.yaml            # базовый конфиг (см. правило поиска пути в load_experiments)
    experiments:
      - name: bm25
        overrides: {retrieval: {mode: bm25}}
      - name: hybrid-e5
        requires_neural: true     # без sentence-transformers → пропуск с причиной
        overrides: {retrieval: {mode: hybrid, dense: {embedder: sentence-transformers}}}
      - name: hybrid-e5-rerank
        requires_neural: true
        heavy: true               # долгий: запускается только с include_heavy=True
        overrides: {rerank: {enabled: true}}
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kbrag.config import AppConfig, load_config
from kbrag.eval.dataset import GoldItem, evidence_spans, items_fingerprint, load_gold, validate_gold
from kbrag.eval.metrics import (
    abstention_report,
    auroc,
    auroc_ci,
    choose_threshold,
    citation_precision,
    citation_recall,
    cites_stale_document,
    fact_recall,
    mean_ignore_none,
    percentile,
    retrieval_metrics,
    token_f1,
)
from kbrag.eval.results import (
    AUROC_SPLITS,
    BY_TYPE_KEY,
    CLASSIFIER_SOURCE,
    GENERATION_METRICS,
    METRIC_SPLITS,
    SCORE_SOURCES,
    ExperimentResult,
    metric_names,
    retrieval_metric_names,
)
from kbrag.generation import ABSTAIN_TEXT, LLMError
from kbrag.ingest import corpus_fingerprint, load_corpus
from kbrag.pipeline import ABSTAIN_CLASSIFIER_FLAG, LOW_CONFIDENCE_FLAG, RAGPipeline
from kbrag.retrieval import neural_available
from kbrag.types import Answer, Citation, Document, ScoredChunk

logger = logging.getLogger(__name__)

__all__ = ["ExperimentResult", "ExperimentSpec", "load_experiments", "run_experiment", "run_grid"]

# Ключи эксперимента в experiments.yaml. Опечатка («requires_nueral») должна падать, а не
# молча запускать нейросетевой эксперимент там, где его нет.
_EXPERIMENT_KEYS = frozenset({"name", "overrides", "requires_neural", "heavy", "description"})
_GRID_KEYS = frozenset({"base", "experiments", "description"})


# ---------------------------------------------------------------------------
# Один эксперимент
# ---------------------------------------------------------------------------


def run_experiment(
    cfg: AppConfig,
    name: str = "default",
    gold: list[GoldItem] | None = None,
    docs: list[Document] | None = None,
) -> ExperimentResult:
    """Прогоняет эталонный набор через пайплайн и возвращает метрики (см. описание модуля).

    ``gold`` и ``docs`` можно передать готовыми (тесты, сетка экспериментов на одном корпусе);
    иначе они читаются из ``cfg.gold_path`` и ``cfg.corpus_dir``.
    """
    _reject_unsupported_judge(cfg)
    docs = list(docs) if docs is not None else load_corpus(cfg.corpus_dir)
    gold = list(gold) if gold is not None else load_gold(cfg.gold_path)
    _check_gold(gold, docs)
    k_values = list(cfg.eval.k_values)

    # Пайплайну отдаём конфиг без порога: порог в оценке считаем сами (шаг 3), а иначе
    # он попытался бы прочитать файл калибровки, оставшийся от прошлого запуска.
    pipeline_cfg = cfg.model_copy(
        update={"abstention": cfg.abstention.model_copy(update={"threshold": None})}
    )
    # Классификатор отказа (если он выбран в конфиге) пайплайн загружает сам — тем же кодом, что сервис.
    pipeline = RAGPipeline.from_config(pipeline_cfg, docs=docs)
    classifier = _uses_classifier(cfg)
    if classifier and pipeline.abstention_model is None:
        raise ValueError(
            f"abstention.method: classifier, но модель отказа {cfg.abstention.model_path} не загрузилась "
            "(нет файла или она обучена для другого поиска/корпуса — причина в предупреждении выше): "
            "сначала kbrag train-abstention --config <этот конфиг>"
        )
    if classifier:
        _require_same_dev(pipeline.abstention_model.metadata.get("dev_sha256"), gold)
    logger.info("Эксперимент %r: %d вопросов, индекс %s", name, len(gold), pipeline.index_stats)

    # Шаг 2: все вопросы без порога. Поиск достаём на max(k) фрагментов, чтобы честно
    # посчитать Recall@10, а генератору отдаём только первые top_k — как в сервисе.
    top_k = cfg.retrieval.top_k
    retrieve_k = max([*k_values, top_k])
    runs: list[_QuestionRun] = []
    llm_errors = 0
    for item in gold:
        retrieved, timings = pipeline.retrieve(item.question, top_k=retrieve_k)
        try:
            answer = pipeline.answer_with_contexts(
                item.question, retrieved[:top_k], timings=timings, apply_threshold=False
            )
        except LLMError as exc:
            # Один зависший запрос к LLM не должен обнулять часовой прогон. Вопрос засчитываем
            # как неверный ответ (пессимистично и честно) и помечаем флагом, а итоговое число
            # ошибок печатаем — тихо продолжать, скрыв сбой, было бы хуже, чем упасть.
            llm_errors += 1
            logger.warning("Вопрос %s: LLM не ответила (%s) — засчитан как неверный ответ", item.qid, exc)
            answer = _llm_error_answer(item.question, retrieved[:top_k], timings, pipeline)
        runs.append(
            _QuestionRun(item=item, answer=answer, retrieved_ids=[s.chunk.chunk_id for s in retrieved])
        )
    if llm_errors:
        logger.warning("Эксперимент %r: %d из %d вопросов без ответа LLM", name, llm_errors, len(gold))

    # Шаг 3: порог. У классификатора — из файла модели, без подбора.
    if classifier:
        threshold = pipeline.abstention_model.decision_threshold
    else:
        threshold = _resolve_eval_threshold(cfg, runs)

    # Шаги 4–5: применяем порог и считаем метрики.
    docs_by_id = {doc.doc_id: doc for doc in docs}
    sources = _available_sources(runs)
    per_question = [
        _score_question(run, threshold, pipeline, docs_by_id, k_values, sources, classifier=classifier)
        for run in runs
    ]
    auroc_sources = [*sources, CLASSIFIER_SOURCE] if classifier else sources
    metrics = _split_metrics(per_question, k_values, cfg.eval.report_split)
    logger.info(
        "Эксперимент %r готов: порог %s, recall@5(test)=%s",
        name,
        threshold,
        metrics["test"].get("recall@5"),
    )
    return ExperimentResult(
        name=name,
        config=cfg.model_dump(mode="json"),
        metrics=metrics,
        per_question=per_question,
        threshold=threshold,
        index_stats=dict(pipeline.index_stats),
        # Шкала порога: у классификатора это P(нет ответа), а не score поиска.
        score_source=pipeline.abstention_method if classifier else pipeline.score_source,
        provenance=data_fingerprint(docs, gold),
        confidence_auroc=confidence_auroc(per_question, auroc_sources),
    )


def _uses_classifier(cfg: AppConfig) -> bool:
    """Решает ли отказ классификатор (а не порог по уверенности) в этом конфиге."""
    return cfg.abstention.enabled and cfg.abstention.method == "classifier"


def _require_same_dev(model_dev_sha256: str | None, gold: Sequence[GoldItem]) -> None:
    """Порог классификатора подбирался на том же dev, что сейчас в эталоне? Нет — ``ValueError``.

    Зачем. Порог модели отказа подобран на dev при обучении и лежит в файле модели; оценка
    применяет его как есть. Но ``assign_splits`` раскладывает вопросы внутри типа по рангу хэша,
    и один новый вопрос в эталоне может перенести соседний dev-вопрос в test. Тогда test молча
    посчитался бы с порогом, подобранным в том числе на test-вопросе, — это утечка. У порогового
    метода такой беды нет: ``"auto"`` подбирается заново на текущем dev при каждом прогоне.

    Почему ошибка, а не предупреждение: так же устроена проверка устаревшей модели (другой поиск
    или корпус) — цифры, посчитанные с утечкой, хуже, чем упавший прогон с понятной подсказкой.
    Нет ``dev_sha256`` в файле (модель из теста, собранная вручную) — сверять не с чем, пропускаем.
    """
    if not model_dev_sha256:
        return
    current = items_fingerprint([item for item in gold if item.split == "dev"])
    if model_dev_sha256 != current:
        raise ValueError(
            f"модель отказа подбирала порог на другом dev (dev_sha256 {model_dev_sha256}, сейчас {current}; "
            "после kbrag build-gold вопросы могли перейти между dev и test) — переобучите: "
            "kbrag train-abstention --config <конфиг>"
        )


def data_fingerprint(docs: Sequence[Document], gold: Sequence[GoldItem]) -> dict[str, Any]:
    """Отпечаток данных прогона: на каком именно корпусе и эталоне получены метрики.

    В ``results.json`` уже лежит конфиг, но в нём только пути (``data/corpus``), а не содержимое:
    правка одного документа или другая ``KBRAG_CORPUS_DIR`` меняют метрики при том же конфиге.
    По хэшам сразу видно, сравнимы ли два отчёта. Хэш не зависит от порядка чтения файлов.
    """
    return {
        "corpus_sha256": corpus_fingerprint(list(docs)),
        "gold_sha256": items_fingerprint(gold),
        "n_docs": len(docs),
        "n_gold": len(gold),
    }


LLM_ERROR_FLAG = "llm_error"  # флаг вопроса, на котором LLM не ответила (таймаут, 5xx и т.п.)


def _llm_error_answer(
    question: str, contexts: list[ScoredChunk], timings: dict[str, float], pipeline: RAGPipeline
) -> Answer:
    """Ответ-заглушка для вопроса, на котором LLM упала: пустой текст, флаг ``llm_error``.

    Уверенность считаем как обычно — по ней на dev подбирается порог, и выбрасывать вопрос
    из калибровки из-за сбоя генерации нельзя.
    """
    search_ms = timings.get("retrieve_ms", 0.0) + timings.get("rerank_ms", 0.0)
    return Answer(
        question=question,
        text="",
        citations=[],
        contexts=contexts,
        abstained=False,
        timings_ms={**timings, "generate_ms": 0.0, "total_ms": search_ms},
        flags=[LLM_ERROR_FLAG],
        confidence=pipeline.confidence(contexts),
        abstain_score=pipeline.abstain_score(question, contexts),
    )


@dataclass
class _QuestionRun:
    """Сырой результат одного вопроса до применения порога."""

    item: GoldItem
    answer: Answer
    retrieved_ids: list[str] = field(default_factory=list)


def _reject_unsupported_judge(cfg: AppConfig) -> None:
    """``eval.judge.enabled: true`` → понятная ошибка вместо тихо проигнорированного флага.

    LLM-судья (``kbrag.eval.judge``) пока библиотечный модуль: прогон его не вызывает.
    Если молча продолжить, в отчёте не будет оценок судьи, а в ``results.json`` при этом
    останется ``judge.enabled: true`` — читатель решит, что судья отработал. Это та же
    логика, что у строгих конфигов: настройка либо работает, либо падает (см. ``config._Strict``).
    """
    if cfg.eval.judge.enabled:
        raise ValueError(
            "eval.judge.enabled=true, но LLM-судья пока не подключён к `kbrag eval` / `kbrag grid`: "
            "оценки faithfulness/correctness в отчёт не попали бы. Выключите флаг и вызывайте "
            "kbrag.eval.judge.LLMJudge отдельно, на сохранённых ответах из per_question."
        )


def _check_gold(gold: Sequence[GoldItem], docs: Sequence[Document]) -> None:
    # Битая разметка должна падать сразу, а не через полчаса прогона на середине набора.
    errors = validate_gold(gold, docs)
    if errors:
        shown = "\n  ".join(errors[:10])
        more = f"\n  … и ещё {len(errors) - 10}" if len(errors) > 10 else ""
        raise ValueError(f"Ошибки в эталонном наборе ({len(errors)}):\n  {shown}{more}")


def _resolve_eval_threshold(cfg: AppConfig, runs: Sequence[_QuestionRun]) -> float | None:
    """Порог для оценки: число из конфига как есть, ``"auto"`` — подбор на dev, иначе None."""
    abstention = cfg.abstention
    if not abstention.enabled or abstention.threshold is None:
        return None
    if abstention.threshold != "auto":
        return float(abstention.threshold)

    dev = [run for run in runs if run.item.split == "dev"]
    if not dev:
        logger.warning("Порог 'auto': в наборе нет dev-вопросов, отказ по порогу выключен")
        return None
    # Только dev! Test-вопросы в подборе порога не участвуют (см. описание модуля).
    threshold = choose_threshold(
        [run.answer.confidence for run in dev],
        [run.item.answerable for run in dev],
    )
    logger.info("Порог отказа подобран на %d dev-вопросах: %s", len(dev), threshold)
    return threshold


def _available_sources(runs: Sequence[_QuestionRun]) -> list[str]:
    """Источники уверенности, которые есть у этого поиска (bm25 — только в bm25/гибриде и т. д.)."""
    present = {key for run in runs for item in run.answer.contexts for key in item.details}
    return [source for source in SCORE_SOURCES if source in present]


def _confidence_by_source(
    contexts: Sequence[ScoredChunk], sources: Sequence[str], abstain_score: float | None = None
) -> dict[str, float | None]:
    """Уверенность по каждому источнику — по тому же правилу, что ``RAGPipeline.confidence`` (максимум).

    Если есть оценка классификатора отказа, добавляется ``"classifier": 1 − p``: так у всех
    источников «больше — вероятнее, что ответ есть», и AUROC считается одной функцией.
    """
    result: dict[str, float | None] = {}
    for source in sources:
        values = [item.details[source] for item in contexts if source in item.details]
        result[source] = max(values) if values else None
    if abstain_score is not None:
        result[CLASSIFIER_SOURCE] = 1.0 - abstain_score
    return result


def confidence_auroc(rows: Sequence[dict[str, Any]], sources: Sequence[str]) -> dict[str, dict]:
    """AUROC «отвечаемый vs неотвечаемый» для каждого источника уверенности на dev и test (+ 95% ДИ).

    Зачем: порог отказа берёт уверенность из одного источника (``abstention.score_source``),
    и выбирать его стоит по данным — по dev, — а не по общему соображению «cross-encoder точнее».
    """
    result: dict[str, dict] = {}
    for split in AUROC_SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        answerable = [row["type"] != "unanswerable" for row in split_rows]
        by_source: dict[str, dict] = {}
        for source in sources:
            scores = [row["confidence_by_source"].get(source) for row in split_rows]
            value = auroc(scores, answerable)
            if value is None:
                continue
            interval = auroc_ci(scores, answerable)
            by_source[source] = {"auroc": value, "ci95": list(interval) if interval else None}
        if by_source:
            result[split] = by_source
    return result


def _question_metric_names(k_values: Sequence[int]) -> list[str]:
    """Метрики, которые имеют смысл для одного вопроса (остальные — только для сплита)."""
    return [*retrieval_metric_names(k_values), *GENERATION_METRICS]


def _score_question(
    run: _QuestionRun,
    threshold: float | None,
    pipeline: RAGPipeline,
    docs_by_id: dict[str, Document],
    k_values: Sequence[int],
    sources: Sequence[str] = (),
    classifier: bool = False,
) -> dict[str, Any]:
    """Строка ``per_question``: что спросили, что ответили и метрики этого вопроса.

    ``classifier=True`` — ``threshold`` относится к P(нет ответа) классификатора (отказ, если
    ``abstain_score >= threshold``, как ``AbstentionModel.abstains``), иначе — к уверенности поиска.
    """
    item, answer = run.item, run.answer
    if classifier:
        rule_flag = ABSTAIN_CLASSIFIER_FLAG
        below_threshold = (
            threshold is not None and answer.abstain_score is not None and answer.abstain_score >= threshold
        )
    else:
        rule_flag = LOW_CONFIDENCE_FLAG
        below_threshold = threshold is not None and (
            answer.confidence is None or answer.confidence < threshold
        )
    abstained = answer.abstained or below_threshold

    # Так ответил бы сервис с этим порогом: фиксированная фраза отказа и никаких ссылок.
    text = ABSTAIN_TEXT if abstained else answer.text
    citations: list[Citation] = [] if abstained else list(answer.citations)
    n_invalid = 0 if abstained else answer.invalid_citations
    flags = list(answer.flags) + ([rule_flag] if below_threshold else [])
    # При отказе по порогу сервис не вызывает генератор — не учитываем его время в латентности.
    latency = answer.timings_ms.get("total_ms", 0.0)
    if below_threshold:
        latency -= answer.timings_ms.get("generate_ms", 0.0)

    scores: dict[str, float | None] = dict.fromkeys(_question_metric_names(k_values))
    if item.answerable:
        spans = evidence_spans(item, docs_by_id)
        retrieved_chunks = [pipeline.chunks_by_id[chunk_id] for chunk_id in run.retrieved_ids]
        scores.update(retrieval_metrics(retrieved_chunks, spans, k_values))

        # Отказ на отвечаемый вопрос = 0 (а не «пропуск»): иначе система, отказывающаяся
        # на все трудные вопросы, выглядела бы точнее честно отвечающей.
        facts = fact_recall(text, item.key_facts)
        scores["fact_recall"] = 0.0 if abstained and facts is not None else facts
        scores["fact_recall_answered"] = None if abstained else facts
        scores["token_f1"] = 0.0 if abstained else token_f1(text, item.answer)
        precision = citation_precision(citations, pipeline.chunks_by_id, spans)
        # Ответ без единой ссылки — не «неприменимо», а ноль: иначе такой ответ (обычное дело
        # для LLM) выпадал бы из среднего и завышал CitationPrecision. Отказ остаётся None.
        if precision is None and not abstained:
            precision = 0.0
        scores["citation_precision"] = precision
        scores["citation_recall"] = citation_recall(citations, pipeline.chunks_by_id, spans)
        if not abstained:
            evidence_docs = {evidence.doc_id for evidence in item.evidence}
            stale = cites_stale_document(citations, pipeline.chunks_by_id, evidence_docs)
            scores["stale_citation_rate"] = float(stale)
    n_markers = len(citations) + n_invalid
    scores["invalid_citation_rate"] = n_invalid / n_markers if n_markers else None

    return {
        "qid": item.qid,
        "type": item.qtype,
        "split": item.split,
        "question": item.question,
        "gold_answer": item.answer,
        "answer": text,
        "abstained": abstained,
        "confidence": answer.confidence,
        "abstain_score": answer.abstain_score,
        "confidence_by_source": _confidence_by_source(answer.contexts, sources, answer.abstain_score),
        "retrieved": list(run.retrieved_ids),
        "flags": flags,
        "latency_ms": round(latency, 3),
        **scores,
    }


def aggregate_metrics(rows: Sequence[dict[str, Any]], k_values: Sequence[int]) -> dict[str, float | None]:
    """Метрики группы вопросов (сплита или типа) в каноническом порядке ``metric_names``.

    Метрики вопросов усредняются без None: «неприменимо» (например, recall у
    неотвечаемого вопроса) не тянет среднее вниз. Поэтому retrieval-метрики сами
    собой считаются только по answerable, а ``fact_recall_answered`` — только по отвеченным.
    """
    question_names = set(_question_metric_names(k_values))
    abstention = abstention_report(
        [bool(row["abstained"]) for row in rows],
        [row["type"] != "unanswerable" for row in rows],
    )
    latencies = [row["latency_ms"] for row in rows]
    result: dict[str, float | None] = {}
    for name in metric_names(k_values):
        if name in question_names:
            result[name] = mean_ignore_none([row.get(name) for row in rows])
        elif name in abstention:
            result[name] = abstention[name]
        elif name == "latency_p50_ms":
            result[name] = _round_ms(percentile(latencies, 50))
        elif name == "latency_p95_ms":
            result[name] = _round_ms(percentile(latencies, 95))
        elif name == "n_answerable":
            result[name] = sum(1 for row in rows if row["type"] != "unanswerable")
        elif name == "n_questions":
            result[name] = len(rows)
    return result


def _round_ms(value: float | None) -> float | None:
    """Как ``latency_ms`` у вопроса — 3 знака: без хвостов float вида 73.54679999999999."""
    return None if value is None else round(value, 3)


def _split_metrics(
    rows: Sequence[dict[str, Any]], k_values: Sequence[int], report_split: str
) -> dict[str, dict]:
    """``{"dev": {...}, "test": {...}, "all": {...}, "by_type": {qtype: {...}}}``."""
    metrics: dict[str, dict] = {
        split: aggregate_metrics(
            rows if split == "all" else [row for row in rows if row["split"] == split], k_values
        )
        for split in METRIC_SPLITS
    }
    report_rows = rows if report_split == "all" else [row for row in rows if row["split"] == report_split]
    by_type: dict[str, dict] = {}
    for qtype in sorted({row["type"] for row in report_rows}):
        by_type[qtype] = aggregate_metrics([row for row in report_rows if row["type"] == qtype], k_values)
    metrics[BY_TYPE_KEY] = by_type
    return metrics


# ---------------------------------------------------------------------------
# Сетка экспериментов
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentSpec:
    """Одна строка из ``experiments.yaml``."""

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    requires_neural: bool = False
    heavy: bool = False
    description: str = ""


def load_experiments(experiments_path: str | Path) -> tuple[Path | None, list[ExperimentSpec]]:
    """Читает ``experiments.yaml``: возвращает путь к базовому конфигу и список экспериментов.

    Правило для ``base`` (относительный путь): сначала ищем **рядом с файлом экспериментов**
    (``configs/experiments.yaml`` + ``base: default.yaml`` → ``configs/default.yaml``), а если
    там файла нет — относительно текущего каталога, то есть корня проекта, откуда запускают
    ``kbrag`` (``base: configs/default.yaml``). Так работают обе привычные записи.
    Нет ``base`` → значения по умолчанию из ``AppConfig``.
    """
    path = Path(experiments_path)
    with path.open(encoding="utf-8-sig") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: ожидается YAML-словарь с ключами base и experiments")
    _reject_unknown_keys(data, _GRID_KEYS, where=str(path))

    raw_experiments = data.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ValueError(f"{path}: 'experiments' должен быть непустым списком")

    specs: list[ExperimentSpec] = []
    seen: set[str] = set()
    for position, raw in enumerate(raw_experiments, start=1):
        where = f"{path}: эксперимент №{position}"
        if not isinstance(raw, dict) or not str(raw.get("name") or "").strip():
            raise ValueError(f"{where}: нужен словарь с непустым полем name")
        _reject_unknown_keys(raw, _EXPERIMENT_KEYS, where=where)
        name = str(raw["name"]).strip()
        if name in seen:
            raise ValueError(f"{path}: имя эксперимента {name!r} повторяется")
        seen.add(name)
        overrides = raw.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"{where} ({name}): overrides должен быть словарём")
        specs.append(
            ExperimentSpec(
                name=name,
                overrides=overrides,
                requires_neural=bool(raw.get("requires_neural", False)),
                heavy=bool(raw.get("heavy", False)),
                description=str(raw.get("description") or ""),
            )
        )
    return _resolve_base(path, data.get("base")), specs


def run_grid(
    experiments_path: str | Path,
    only: list[str] | None = None,
    include_heavy: bool = False,
) -> list[ExperimentResult]:
    """Прогоняет эксперименты из ``experiments.yaml`` по порядку.

    Конфиг эксперимента = базовый YAML + ``overrides`` (рекурсивное слияние
    ``config.deep_merge``) + переменные окружения. Эксперименты, которые нельзя или
    не просили запускать, не выбрасываются, а возвращаются со ``skipped``-причиной —
    в отчёте видно, что они пропущены осознанно.

    ``only`` — запустить только эксперименты с этими именами (неизвестное имя → ``ValueError``).
    """
    base_path, specs = load_experiments(experiments_path)
    if only is not None:
        unknown = sorted(set(only) - {spec.name for spec in specs})
        if unknown:
            raise ValueError(f"Неизвестные эксперименты: {unknown}; есть: {[s.name for s in specs]}")
        wanted = set(only)
        specs = [spec for spec in specs if spec.name in wanted]

    # Корпус и эталон читаем один раз на сетку: эксперименты обычно отличаются только поиском.
    docs_cache: dict[str, list[Document]] = {}
    gold_cache: dict[str, list[GoldItem]] = {}
    results: list[ExperimentResult] = []
    for spec in specs:
        # Конфиг проверяем и у пропускаемых экспериментов: опечатка в overrides видна сразу.
        cfg = load_config(base_path, overrides=spec.overrides)
        reason = _skip_reason(spec, cfg, include_heavy)
        if reason is not None:
            logger.info("Эксперимент %r пропущен: %s", spec.name, reason)
            results.append(
                ExperimentResult.for_skipped(spec.name, reason, config=cfg.model_dump(mode="json"))
            )
            continue

        if cfg.corpus_dir not in docs_cache:
            docs_cache[cfg.corpus_dir] = load_corpus(cfg.corpus_dir)
        if cfg.gold_path not in gold_cache:
            gold_cache[cfg.gold_path] = load_gold(cfg.gold_path)
        logger.info("Запуск эксперимента %r", spec.name)
        results.append(
            run_experiment(
                cfg, name=spec.name, gold=gold_cache[cfg.gold_path], docs=docs_cache[cfg.corpus_dir]
            )
        )
    return results


def config_needs_neural(cfg: AppConfig) -> bool:
    """Нужны ли конфигу sentence-transformers (на случай, если requires_neural забыли указать)."""
    uses_st_embedder = cfg.retrieval.mode in ("dense", "hybrid") and cfg.retrieval.dense.embedder == (
        "sentence-transformers"
    )
    return cfg.rerank.enabled or uses_st_embedder


def _skip_reason(spec: ExperimentSpec, cfg: AppConfig, include_heavy: bool) -> str | None:
    if (spec.requires_neural or config_needs_neural(cfg)) and not neural_available():
        return "нужны нейросетевые зависимости (sentence-transformers): pip install -e .[neural]"
    if spec.heavy and not include_heavy:
        return "тяжёлый эксперимент (heavy): запускается только с include_heavy=True"
    # Файла модели нет — пропуск, а не падение всей сетки: модель обучают отдельной командой,
    # и до первого обучения сетка должна проходить. Модель есть, но устарела (другой поиск или
    # корпус) — это уже ошибка run_experiment: устаревшую модель надо переобучить, а не прятать.
    model_path = cfg.abstention.model_path
    if _uses_classifier(cfg) and not (model_path and Path(model_path).is_file()):
        return (
            f"нет файла модели отказа {model_path}: сначала kbrag train-abstention "
            "--config <конфиг с теми же настройками поиска>"
        )
    return None


def _resolve_base(experiments_path: Path, base: Any) -> Path | None:
    if base is None:
        return None
    candidate = Path(str(base))
    if candidate.is_absolute():
        options = [candidate]
    else:
        options = [experiments_path.parent / candidate, Path.cwd() / candidate]
    for option in options:
        if option.is_file():
            return option
    tried = ", ".join(str(option) for option in options)
    raise FileNotFoundError(f"{experiments_path}: базовый конфиг {base!r} не найден (искали: {tried})")


def _reject_unknown_keys(data: dict, allowed: frozenset[str], where: str) -> None:
    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        raise ValueError(f"{where}: неизвестные ключи {unknown}; допустимы {sorted(allowed)}")
