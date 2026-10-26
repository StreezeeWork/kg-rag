"""RAG-пайплайн: вопрос → поиск → [реранкинг] → решение об отказе → генерация ответа.

Это «дирижёр» проекта: сам он почти ничего не вычисляет, а только вызывает в нужном
порядке модули ``ingest``, ``retrieval`` и ``generation``. Поэтому один и тот же
пайплайн используют и HTTP-сервис (``kbrag.api``), и оценка качества (``kbrag.eval.runner``):
метрики считаются ровно на том коде, который отвечает пользователям.

Схема одного ответа (``RAGPipeline.answer``)::

    вопрос ──► retrieve(): поиск candidates_k/top_k ──► [реранкер] ──► top_k фрагментов
           ──► confidence(): насколько поиск «уверен», что нашёл ответ
           ──► guardrails: пометить и вычистить инъекции во фрагментах
           ──► confidence < threshold? ──да──► отказ (генератор не вызывается)
                                       └─нет─► генератор ──► разбор ссылок [n]

Порог отказа подбирается при оценке (``kbrag eval``) на dev-выборке и сохраняется в файл
калибровки. Сервис читает его, только если файл посчитан для того же поиска (совпадает
``calibration_fingerprint``): порог BM25 бессмысленно применять к косинусам dense.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from kbrag.config import AppConfig, config_fingerprint
from kbrag.generation import (
    ABSTAIN_TEXT,
    Generator,
    check_question,
    make_generator,
    parse_citations,
    sanitize_contexts,
    select_within_budget,
)
from kbrag.ingest import chunk_corpus, corpus_fingerprint, load_corpus, make_chunker
from kbrag.retrieval import Reranker, Retriever, build_retriever, make_reranker
from kbrag.types import Answer, Chunk, Document, ScoredChunk

logger = logging.getLogger(__name__)

# Флаг ответа, который поставлен из-за низкой уверенности поиска (а не по решению генератора).
LOW_CONFIDENCE_FLAG = "low_confidence"
# Не все найденные фрагменты влезли в лимит generation.max_context_chars: генератор видел только начало выдачи.
CONTEXT_TRUNCATED_FLAG = "context_truncated"


# ---------------------------------------------------------------------------
# Калибровка порога отказа
# ---------------------------------------------------------------------------


def calibration_fingerprint(cfg: AppConfig) -> str:
    """Отпечаток настроек, от которых зависит шкала «уверенности» поиска.

    Порог имеет смысл только для того поиска, на котором его подобрали: смена нарезки,
    режима поиска или реранкера меняет и распределение score. Настройки генерации
    сюда не входят — от них уверенность поиска не зависит.
    """
    return config_fingerprint(cfg, "chunking", "retrieval", "rerank")


def resolve_score_source(cfg: AppConfig, has_reranker: bool) -> str:
    """Из какого поля ``ScoredChunk.details`` брать уверенность.

    ``score_source: auto`` выбирает самую «откалиброванную» доступную оценку:
    реранкер (cross-encoder видит вопрос и фрагмент вместе) → косинус dense
    (шкала от −1 до 1, не зависит от длины запроса) → BM25 (единственный вариант
    в режиме ``bm25``; его score растёт с числом слов в вопросе, порог для него грубее).
    """
    source = cfg.abstention.score_source
    if source != "auto":
        return source
    if has_reranker:
        return "rerank"
    if cfg.retrieval.mode in ("dense", "hybrid"):
        return "dense"
    return "bm25"


def save_calibration(
    cfg: AppConfig,
    threshold: float | None,
    path: str | Path | None = None,
    experiment: str | None = None,
    corpus_sha256: str | None = None,
) -> Path:
    """Сохраняет подобранный на dev порог для сервиса (по умолчанию в ``abstention.calibration_path``).

    Формат файла (единственное место, где он задаётся; ``kbrag eval --save-calibration`` зовёт эту функцию)::

        {"fingerprint": "3f2a9c…", "score_source": "dense", "threshold": 0.41,
         "corpus_sha256": "9b1e…", "experiment": "offline_baseline",
         "created_at": "2025-06-01T12:00:00+00:00"}

    ``fingerprint``, ``score_source`` и ``corpus_sha256`` нужны при чтении: они защищают от
    применения порога к поиску или корпусу, для которых он не подбирался (``corpus_sha256`` —
    ``ingest.corpus_fingerprint``, пишется, если передан). ``experiment`` и ``created_at`` пишутся,
    только если передан ``experiment``, — это подсказка человеку, откуда взялся порог;
    сервис эти поля не читает.
    """
    target = Path(path) if path is not None else _calibration_path(cfg)
    if target is None:
        raise ValueError("не задан путь к файлу калибровки (abstention.calibration_path)")
    payload: dict[str, object] = {
        "fingerprint": calibration_fingerprint(cfg),
        "score_source": resolve_score_source(cfg, has_reranker=cfg.rerank.enabled),
        "threshold": threshold,
    }
    if corpus_sha256 is not None:
        payload["corpus_sha256"] = corpus_sha256
    if experiment is not None:
        payload["experiment"] = experiment
        payload["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    logger.info("Порог отказа %s сохранён в %s", threshold, target)
    return target


def load_calibration(cfg: AppConfig, corpus_sha256: str | None = None) -> float | None:
    """Читает порог из файла калибровки; None, если файла нет или он от другого поиска или корпуса.

    ``corpus_sha256`` — отпечаток корпуса, который сейчас проиндексирован (``ingest.corpus_fingerprint``).
    Если он передан, а в файле записан другой, порог не применяется: правка документов меняет
    распределение уверенности, и порог, подобранный на старом корпусе, уже не тот.

    Отсутствие порога — не ошибка: сервис просто не отказывается по уверенности
    (генератор всё равно может ответить «В базе знаний нет ответа…»). Поэтому здесь
    не бросаем исключения, а пишем в лог, почему порог не применён.
    """
    path = _calibration_path(cfg)
    if path is None or not path.is_file():
        # WARNING, а не INFO: поведение сервиса меняется (он перестаёт отказываться по порогу),
        # и об этом должно быть видно в `docker logs` даже при уровне логов по умолчанию.
        logger.warning(
            "Файл калибровки %s не найден: отказ по порогу уверенности выключен "
            "(подберите порог: kbrag eval --save-calibration)",
            path,
        )
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        logger.warning("Не удалось прочитать файл калибровки %s (%s): порог не применяется", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("Файл калибровки %s должен содержать JSON-объект: порог не применяется", path)
        return None

    expected = calibration_fingerprint(cfg)
    if data.get("fingerprint") != expected:
        logger.warning(
            "Файл калибровки %s посчитан для других настроек поиска (fingerprint %s, нужен %s): "
            "порог не применяется — перезапустите kbrag eval --config <конфиг> --save-calibration",
            path,
            data.get("fingerprint"),
            expected,
        )
        return None
    # Отпечаток не включает секцию abstention, поэтому источник уверенности сверяем отдельно.
    expected_source = resolve_score_source(cfg, has_reranker=cfg.rerank.enabled)
    if "score_source" in data and data["score_source"] != expected_source:
        logger.warning(
            "Порог в %s подобран для уверенности %r, а сейчас используется %r: порог не применяется",
            path,
            data["score_source"],
            expected_source,
        )
        return None

    saved_corpus = data.get("corpus_sha256")
    if corpus_sha256 is not None and saved_corpus is not None and saved_corpus != corpus_sha256:
        logger.warning(
            "Порог в %s подобран на другом корпусе (corpus_sha256 %s, сейчас %s): порог не применяется — "
            "перезапустите kbrag eval --config <конфиг> --save-calibration",
            path,
            saved_corpus,
            corpus_sha256,
        )
        return None

    threshold = data.get("threshold")
    if threshold is None:
        return None
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        logger.warning("В файле калибровки %s порог не число: %r", path, threshold)
        return None
    logger.info("Порог отказа %.4f загружен из %s", threshold, path)
    return float(threshold)


def resolve_threshold(cfg: AppConfig, corpus_sha256: str | None = None) -> float | None:
    """Порог для сервиса: число из конфига как есть, ``"auto"`` — из файла калибровки, null — без порога."""
    value = cfg.abstention.threshold
    if value == "auto":
        return load_calibration(cfg, corpus_sha256=corpus_sha256)
    return None if value is None else float(value)


def _calibration_path(cfg: AppConfig) -> Path | None:
    raw = cfg.abstention.calibration_path
    return Path(raw) if raw else None


# ---------------------------------------------------------------------------
# Пайплайн
# ---------------------------------------------------------------------------


class RAGPipeline:
    """Связывает поиск, реранкер, порог отказа, guardrails и генератор в один вызов ``answer``.

    Обычно создаётся через ``RAGPipeline.from_config(cfg)``. Конструктор принимает готовые
    компоненты — это удобно в тестах: можно подставить генератор-«шпион» или фальшивый
    реранкер, не трогая остальное. Ретривер в конструктор передаётся **уже проиндексированным**.
    """

    def __init__(
        self,
        cfg: AppConfig,
        retriever: Retriever,
        generator: Generator,
        reranker: Reranker | None = None,
        threshold: float | None = None,
        docs: list[Document] | None = None,
        chunks: list[Chunk] | None = None,
    ) -> None:
        self.cfg = cfg
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.threshold = threshold
        # Фрагменты берём у ретривера, если их не передали явно: они нужны для статистики
        # и для метрик (по chunk_id из ссылки находим исходный фрагмент).
        self.chunks: list[Chunk] = (
            list(chunks) if chunks is not None else list(getattr(retriever, "chunks", []))
        )
        self.docs: list[Document] = list(docs) if docs is not None else []
        self.chunks_by_id: dict[str, Chunk] = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.score_source = resolve_score_source(cfg, has_reranker=reranker is not None)
        self.index_stats: dict = self._compute_index_stats(index_time_ms=0.0)
        self._warn_if_source_unavailable()

    @classmethod
    def from_config(cls, cfg: AppConfig, docs: list[Document] | None = None) -> RAGPipeline:
        """Собирает пайплайн по конфигу: корпус → фрагменты → индекс → реранкер → генератор → порог.

        ``docs`` можно передать готовыми (тесты, оценка нескольких конфигов на одном корпусе);
        иначе корпус читается из ``cfg.corpus_dir``.
        """
        if docs is None:
            docs = load_corpus(cfg.corpus_dir)
        chunks = chunk_corpus(list(docs), make_chunker(cfg.chunking))

        retriever = build_retriever(cfg)
        started = time.perf_counter()
        retriever.index(chunks)
        index_time_ms = _elapsed_ms(started)

        pipeline = cls(
            cfg,
            retriever=retriever,
            generator=make_generator(cfg.generation),
            # Реранкер должен видеть тот же текст фрагмента, что и индекс (с заголовком или без).
            reranker=make_reranker(cfg.rerank, with_header=cfg.chunking.with_header),
            threshold=resolve_threshold(cfg, corpus_sha256=corpus_fingerprint(list(docs))),
            docs=list(docs),
            chunks=chunks,
        )
        pipeline.index_stats = pipeline._compute_index_stats(index_time_ms=index_time_ms)
        logger.info(
            "Пайплайн готов: %d документов, %d фрагментов, индекс за %.0f мс, порог %s, уверенность по %s",
            pipeline.index_stats["n_docs"],
            pipeline.index_stats["n_chunks"],
            index_time_ms,
            pipeline.threshold,
            pipeline.score_source,
        )
        return pipeline

    # --- шаги пайплайна ---

    def retrieve(self, question: str, top_k: int | None = None) -> tuple[list[ScoredChunk], dict[str, float]]:
        """Находит ``top_k`` фрагментов и возвращает их вместе с таймингами этапов.

        С реранкером поиск достаёт больше кандидатов (``candidates_k``): реранкер может
        поднять наверх фрагмент с 20-го места, но только если этот фрагмент ему показали.
        """
        top_k = self._resolve_top_k(top_k)
        n_candidates = max(self.cfg.retrieval.candidates_k, top_k) if self.reranker is not None else top_k

        started = time.perf_counter()
        candidates = self.retriever.search(question, n_candidates)
        timings = {"retrieve_ms": _elapsed_ms(started), "rerank_ms": 0.0}

        if self.reranker is None:
            return candidates[:top_k], timings
        started = time.perf_counter()
        reranked = self.reranker.rerank(question, candidates, top_k)
        timings["rerank_ms"] = _elapsed_ms(started)
        return reranked[:top_k], timings

    def confidence(self, contexts: list[ScoredChunk]) -> float | None:
        """Уверенность поиска: максимум ``details[source]`` по найденным фрагментам.

        Источник выбирается в ``resolve_score_source`` (``rerank`` / ``dense`` / ``bm25``).
        Важная деталь для гибрида: берём **исходный score компонента** (например, косинус
        dense), а не итоговый RRF-score. RRF-score ранговый: ``Σ 1 / (60 + rank)`` у первого
        места почти одинаков (≈ 0.016–0.033) и для точного попадания, и для вопроса, ответа
        на который в базе нет, — ведь кто-то всегда оказывается первым. Абсолютную
        релевантность он не отражает, и порог по нему ничего бы не отсекал.

        None — если ни у одного фрагмента нет нужной оценки (например, поиск ничего
        не нашёл): такой вопрос при заданном пороге получает отказ.

        Ограничение, о котором стоит помнить: это сигнал *поиска*, а не ответа. На вопросе
        «тема есть, а нужного факта нет» (тариф, которого нет в базе) поиск находит тематически
        близкий фрагмент с высоким score, и порог такой вопрос не отсекает.
        """
        values = [item.details[self.score_source] for item in contexts if self.score_source in item.details]
        return max(values) if values else None

    def answer(self, question: str, top_k: int | None = None, apply_threshold: bool = True) -> Answer:
        """Полный ответ на вопрос: поиск + решение об отказе + генерация со ссылками.

        ``apply_threshold=False`` использует оценка: она прогоняет все вопросы без порога,
        а порог подбирает и применяет потом сама (подробнее — ``kbrag.eval.runner``).

        Уверенность всегда считается по первым ``cfg.retrieval.top_k`` фрагментам — на них порог
        калибровался в ``kbrag eval``, — даже если в запросе другой ``top_k``. Иначе запрос с
        ``top_k=1`` или ``20`` сравнивал бы с порогом другую величину: в гибриде лучший по dense
        фрагмент может стоять в выдаче ниже пятого места. Выдача с бо́льшим k — продолжение
        выдачи с меньшим (пул кандидатов у поиска и реранкера фиксирован), поэтому берём
        одну выдачу длины ``max(top_k, cfg.top_k)`` и режем её на две части.
        """
        top_k = self._resolve_top_k(top_k)
        calibration_k = self.cfg.retrieval.top_k
        retrieved, timings = self.retrieve(question, max(top_k, calibration_k))
        return self.answer_with_contexts(
            question,
            retrieved[:top_k],
            timings=timings,
            apply_threshold=apply_threshold,
            confidence_contexts=retrieved[:calibration_k],
        )

    def answer_with_contexts(
        self,
        question: str,
        contexts: list[ScoredChunk],
        timings: dict[str, float] | None = None,
        apply_threshold: bool = True,
        confidence_contexts: list[ScoredChunk] | None = None,
    ) -> Answer:
        """Вторая половина ``answer``: всё, что происходит после поиска.

        Вынесена отдельно, чтобы оценка могла один раз достать длинную выдачу
        (для Recall@10) и отдать генератору её начало — без второго поиска.
        ``timings`` — тайминги поиска из ``retrieve``; ``total_ms`` = поиск + реранкинг +
        всё, что сделано здесь (guardrails, генерация, разбор ссылок).
        ``confidence_contexts`` — по каким фрагментам считать уверенность (по умолчанию — ``contexts``).
        """
        started = time.perf_counter()
        timings = {"retrieve_ms": 0.0, "rerank_ms": 0.0, **(timings or {})}
        search_ms = timings["retrieve_ms"] + timings["rerank_ms"]
        guardrails = self.cfg.guardrails
        flags = check_question(question) if guardrails.enabled else []
        confidence = self.confidence(contexts if confidence_contexts is None else confidence_contexts)

        # Guardrails — ДО решения об отказе: пометки «подозрительный текст» нужны и тогда, когда
        # генератор не вызывается. При отказе пользователь всё равно видит найденные фрагменты
        # («ближайшие фрагменты»), и фрагмент с инъекцией должен быть помечен, а /metrics — посчитан.
        # Генератор получает очищенные КОПИИ фрагментов, а пользователь и метрики — исходные:
        # у копии после удаления строки текст уже не совпадает с документом по смещениям.
        prompt_contexts = contexts
        if guardrails.enabled:
            prompt_contexts, injection_flags = sanitize_contexts(contexts, action=guardrails.action)
            flags.extend(injection_flags)

        if apply_threshold and self.should_abstain(confidence):
            # Отказываемся до генерации: экономим вызов LLM и не даём ей «додумать» ответ
            # по нерелевантным фрагментам. Контексты всё равно возвращаем — для отладки.
            timings["generate_ms"] = 0.0
            timings["total_ms"] = search_ms + _elapsed_ms(started)
            return Answer(
                question=question,
                text=ABSTAIN_TEXT,
                citations=[],
                contexts=contexts,
                abstained=True,
                timings_ms=timings,
                flags=[*flags, LOW_CONFIDENCE_FLAG],
                confidence=confidence,
            )

        # Лимит длины контекста применяем здесь, один раз для любого генератора: extractive и LLM
        # видят одинаковый набор фрагментов, а ссылки [n] проверяются ровно по показанным —
        # иначе ссылка LLM на номер, вырезанный из промпта, засчитывалась бы как настоящая.
        budgeted = select_within_budget(prompt_contexts, self.cfg.generation.max_context_chars)
        if budgeted != prompt_contexts:
            flags.append(CONTEXT_TRUNCATED_FLAG)
        prompt_contexts = budgeted

        gen_started = time.perf_counter()
        result = self.generator.generate(question, prompt_contexts)
        timings["generate_ms"] = _elapsed_ms(gen_started)
        flags.extend(result.flags)
        if result.abstained:
            # Отказ генератора выглядит так же, как отказ по порогу: фиксированная фраза и никаких
            # ссылок. Иначе «К сожалению, в базе знаний нет ответа… [1]» показал бы ссылку на
            # фрагмент, которым ответ якобы подтверждён, а оценка видела бы другой текст, чем сервис.
            text, citations, invalid = ABSTAIN_TEXT, [], 0
        else:
            # Номер [n] в ответе — это позиция в списке, который видел генератор. После drop_chunk
            # он может не совпасть с позицией в Answer.contexts; надёжная связь — Citation.chunk_id.
            text = result.text
            citations, invalid = parse_citations(text, prompt_contexts)
        timings["total_ms"] = search_ms + _elapsed_ms(started)
        return Answer(
            question=question,
            text=text,
            citations=citations,
            contexts=contexts,
            abstained=result.abstained,
            timings_ms=timings,
            flags=flags,
            usage=dict(result.usage),
            invalid_citations=invalid,
            confidence=confidence,
        )

    def should_abstain(self, confidence: float | None) -> bool:
        """Отказаться ли по порогу: порог задан и уверенность ниже (или её нет вовсе).

        Отсутствие уверенности трактуем как «ниже любого порога» — так же, как
        ``metrics.choose_threshold`` при калибровке, иначе сервис и оценка расходились бы.
        """
        if not self.cfg.abstention.enabled or self.threshold is None:
            return False
        return confidence is None or confidence < self.threshold

    # --- вспомогательное ---

    def _resolve_top_k(self, top_k: int | None) -> int:
        value = self.cfg.retrieval.top_k if top_k is None else top_k
        if value < 1:
            raise ValueError(f"top_k должен быть ≥ 1, получено {value}")
        return value

    def _compute_index_stats(self, index_time_ms: float) -> dict:
        n_words = [len(chunk.text.split()) for chunk in self.chunks]
        n_docs = len(self.docs) if self.docs else len({chunk.doc_id for chunk in self.chunks})
        return {
            "n_docs": n_docs,
            "n_chunks": len(self.chunks),
            "avg_chunk_words": round(sum(n_words) / len(n_words), 1) if n_words else 0.0,
            "index_time_ms": round(index_time_ms, 3),
        }

    def _warn_if_source_unavailable(self) -> None:
        # Явно заданный источник, которого в этой конфигурации нет, сделал бы уверенность
        # всегда None — и при заданном пороге сервис отказывался бы на каждый вопрос.
        available = {"rerank"} if self.reranker is not None else set()
        mode = self.cfg.retrieval.mode
        available |= {"bm25", "dense"} if mode == "hybrid" else {mode}
        if self.score_source not in available:
            logger.warning(
                "abstention.score_source=%r недоступен (режим %s, реранкер %s): уверенность будет None",
                self.score_source,
                mode,
                "есть" if self.reranker is not None else "нет",
            )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
