"""Минимальный клиент OpenAI-совместимого API (``POST /chat/completions``).

Почему свой клиент, а не SDK: такой API отдают vLLM, Ollama, LM Studio, OpenAI и многие
другие, и нам нужен ровно один запрос. Свой клиент на ``httpx`` — это ~100 строк без
лишних зависимостей, с понятными повторами и тестами без сети (через ``MockTransport``).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from kbrag.config import GenerationConfig, JudgeConfig

logger = logging.getLogger(__name__)

# Повторять имеет смысл только «временные» ошибки: перегрузка (429) и сбои сервера (5xx).
# Ошибка 400/401/404 при повторе не исчезнет — её сразу отдаём вызывающему коду.
_RETRY_STATUS_TOO_MANY_REQUESTS = 429
_BACKOFF_BASE_S = 0.5
_ERROR_BODY_PREVIEW = 300


class LLMError(RuntimeError):
    """LLM не ответила или ответила чем-то неразборчивым."""


@dataclass
class LLMResponse:
    text: str
    usage: dict[str, int]
    latency_ms: float
    model: str


def _is_retryable_status(status_code: int) -> bool:
    return status_code == _RETRY_STATUS_TOO_MANY_REQUESTS or status_code >= 500


class OpenAICompatibleClient:
    """Синхронный клиент с таймаутом и повторами с экспоненциальной задержкой.

    ``transport`` и ``sleep`` вынесены в параметры ради тестов: можно подменить сеть
    и не ждать реальные секунды между повторами.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries не может быть отрицательным")
        # Без rstrip «http://host/v1/» превратился бы в «http://host/v1//chat/completions».
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or None
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._sleep = sleep

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            # Локальным серверам (vLLM, Ollama) ключ не нужен — тогда и заголовок не шлём.
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._http = httpx.Client(timeout=timeout_s, headers=headers, transport=transport)

    @classmethod
    def from_config(cls, cfg: GenerationConfig | JudgeConfig) -> OpenAICompatibleClient:
        """Создаёт клиент по конфигу; сам ключ берётся из переменной окружения.

        В YAML хранится только *имя* переменной (``api_key_env``): секреты не должны
        попадать в репозиторий вместе с конфигом.
        """
        api_key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
        return cls(
            base_url=cfg.base_url,
            model=cfg.model,
            api_key=api_key,
            timeout_s=cfg.timeout_s,
            # У JudgeConfig нет max_retries — для судьи берём значение по умолчанию.
            max_retries=getattr(cfg, "max_retries", 3),
        )

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Отправляет диалог и возвращает ответ модели.

        Всего делается до ``max_retries + 1`` попыток; перед повтором номер ``attempt``
        (с нуля) ждём ``0.5 * 2**attempt`` секунд: 0.5, 1, 2, … — так перегруженный
        сервер успевает «отдышаться», а мы не долбим его запросами.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        started = time.perf_counter()
        last_error = "неизвестная ошибка"
        for attempt in range(self.max_retries + 1):
            try:
                response = self._http.post(self.chat_url, json=payload)
            except httpx.TimeoutException as exc:
                last_error = f"таймаут запроса ({self.timeout_s} с): {exc!r}"
            except httpx.TransportError as exc:
                # Сервер ещё стартует или сеть моргнула — тоже временная проблема.
                last_error = f"сетевая ошибка: {exc!r}"
            else:
                if response.status_code == 200:
                    latency_ms = (time.perf_counter() - started) * 1000
                    return self._parse_response(response, latency_ms)
                body = response.text[:_ERROR_BODY_PREVIEW]
                last_error = f"HTTP {response.status_code}: {body}"
                if not _is_retryable_status(response.status_code):
                    raise LLMError(f"LLM вернула ошибку {last_error}")

            if attempt < self.max_retries:
                delay = _BACKOFF_BASE_S * 2**attempt
                logger.warning(
                    "Попытка %d/%d не удалась (%s), повтор через %.1f с",
                    attempt + 1,
                    self.max_retries + 1,
                    last_error,
                    delay,
                )
                self._sleep(delay)

        raise LLMError(f"LLM не ответила после {self.max_retries + 1} попыток: {last_error}")

    def _parse_response(self, response: httpx.Response, latency_ms: float) -> LLMResponse:
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Неожиданный формат ответа LLM: {response.text[:_ERROR_BODY_PREVIEW]}") from exc
        if not isinstance(content, str):
            raise LLMError("LLM вернула пустой ответ (content не строка)")

        raw_usage = data.get("usage")
        if not isinstance(raw_usage, dict):
            raw_usage = {}
        # В usage бывают вложенные объекты (например, prompt_tokens_details) — берём только счётчики.
        usage = {
            key: value
            for key, value in raw_usage.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        return LLMResponse(
            text=content, usage=usage, latency_ms=latency_ms, model=data.get("model") or self.model
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> OpenAICompatibleClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
