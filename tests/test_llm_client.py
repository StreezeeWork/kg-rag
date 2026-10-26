"""Тесты LLM-клиента без сети: ответы сервера подменяются через ``httpx.MockTransport``."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from kbrag.config import GenerationConfig, JudgeConfig
from kbrag.generation.llm import LLMError, LLMResponse, OpenAICompatibleClient

BASE_URL = "http://llm.test/v1"
MESSAGES = [{"role": "system", "content": "Правила"}, {"role": "user", "content": "Вопрос?"}]

# Элемент сценария: готовый ответ сервера или имя сетевой ошибки, которую надо «выбросить».
Step = httpx.Response | str


def ok_response(content: str = "Отпуск — 28 календарных дней [1].", **extra: object) -> httpx.Response:
    body = {
        "id": "cmpl-1",
        "model": "served-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 12, "total_tokens": 132},
    }
    body.update(extra)
    return httpx.Response(200, json=body)


class Server:
    """Фейковый сервер: отдаёт ответы по сценарию и запоминает запросы."""

    def __init__(self, steps: list[Step]) -> None:
        self.steps = steps
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self.steps[min(len(self.requests), len(self.steps)) - 1]
        if step == "timeout":
            raise httpx.ReadTimeout("read timed out", request=request)
        if step == "connect_error":
            raise httpx.ConnectError("connection refused", request=request)
        assert isinstance(step, httpx.Response)
        # Свежая копия: один и тот же объект Response нельзя безопасно отдавать несколько раз.
        return httpx.Response(step.status_code, headers=step.headers, content=step.content)


def make_client(server: Server, sleeps: list[float], **kwargs: object) -> OpenAICompatibleClient:
    params: dict = {"base_url": BASE_URL, "model": "test-model", "max_retries": 3}
    params.update(kwargs)
    record_sleep: Callable[[float], None] = sleeps.append  # не ждём реальные секунды
    return OpenAICompatibleClient(transport=httpx.MockTransport(server), sleep=record_sleep, **params)


def test_successful_call_sends_expected_request_and_parses_response() -> None:
    server, sleeps = Server([ok_response()]), []
    client = make_client(server, sleeps)

    response = client.chat(MESSAGES, temperature=0.2, max_tokens=64)

    assert isinstance(response, LLMResponse)
    assert response.text == "Отпуск — 28 календарных дней [1]."
    assert response.usage == {"prompt_tokens": 120, "completion_tokens": 12, "total_tokens": 132}
    assert response.model == "served-model"
    assert response.latency_ms >= 0
    assert sleeps == []

    (request,) = server.requests
    assert request.method == "POST"
    assert str(request.url) == f"{BASE_URL}/chat/completions"
    payload = json.loads(request.content)
    assert payload == {"model": "test-model", "messages": MESSAGES, "temperature": 0.2, "max_tokens": 64}


def test_response_format_is_sent_only_when_given() -> None:
    server = Server([ok_response('{"score": 5}')])
    client = make_client(server, [])

    client.chat(MESSAGES, response_format={"type": "json_object"})

    assert json.loads(server.requests[0].content)["response_format"] == {"type": "json_object"}


def test_trailing_slash_in_base_url_does_not_double() -> None:
    server = Server([ok_response()])
    client = make_client(server, [], base_url=BASE_URL + "/")

    client.chat(MESSAGES)

    assert client.base_url == BASE_URL
    assert server.requests[0].url.path == "/v1/chat/completions"


def test_usage_keeps_only_integer_counters_and_tolerates_absence() -> None:
    nested_usage = {"prompt_tokens": 7, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 0}}
    server = Server([ok_response(usage=nested_usage), ok_response(usage=None)])
    client = make_client(server, [])

    assert client.chat(MESSAGES).usage == {"prompt_tokens": 7, "completion_tokens": 3}
    assert client.chat(MESSAGES).usage == {}


def test_model_falls_back_to_requested_one() -> None:
    server = Server([ok_response(model=None)])
    assert make_client(server, []).chat(MESSAGES).model == "test-model"


def test_retries_after_429_and_503_with_exponential_backoff() -> None:
    server = Server([httpx.Response(429, text="slow down"), httpx.Response(503, text="busy"), ok_response()])
    sleeps: list[float] = []
    client = make_client(server, sleeps)

    response = client.chat(MESSAGES)

    assert response.text.startswith("Отпуск")
    assert len(server.requests) == 3
    assert sleeps == [0.5, 1.0]


def test_raises_after_exhausting_retries() -> None:
    server = Server([httpx.Response(500, text="internal error")])
    sleeps: list[float] = []
    client = make_client(server, sleeps, max_retries=3)

    with pytest.raises(LLMError, match="4 попыток") as exc_info:
        client.chat(MESSAGES)

    assert len(server.requests) == 4  # первая попытка + 3 повтора
    assert sleeps == [0.5, 1.0, 2.0]  # после последней попытки не ждём
    assert "HTTP 500" in str(exc_info.value)


def test_zero_retries_means_single_attempt() -> None:
    server = Server([httpx.Response(503)])
    sleeps: list[float] = []
    with pytest.raises(LLMError):
        make_client(server, sleeps, max_retries=0).chat(MESSAGES)
    assert len(server.requests) == 1 and sleeps == []


def test_timeout_is_retried_and_then_succeeds() -> None:
    server = Server(["timeout", ok_response()])
    sleeps: list[float] = []

    response = make_client(server, sleeps).chat(MESSAGES)

    assert response.text.startswith("Отпуск")
    assert sleeps == [0.5]


def test_persistent_timeout_raises_llm_error() -> None:
    server = Server(["timeout"])
    sleeps: list[float] = []

    with pytest.raises(LLMError, match="таймаут"):
        make_client(server, sleeps, max_retries=2, timeout_s=1.5).chat(MESSAGES)

    assert len(server.requests) == 3
    assert sleeps == [0.5, 1.0]


def test_connection_error_is_retried() -> None:
    server = Server(["connect_error", ok_response()])
    sleeps: list[float] = []
    assert make_client(server, sleeps).chat(MESSAGES).text
    assert sleeps == [0.5]


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_client_errors_are_not_retried(status: int) -> None:
    server = Server([httpx.Response(status, text="bad request")])
    sleeps: list[float] = []

    with pytest.raises(LLMError, match=f"HTTP {status}"):
        make_client(server, sleeps).chat(MESSAGES)

    assert len(server.requests) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>not json</html>"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
        httpx.Response(200, json=["unexpected"]),
    ],
)
def test_malformed_response_raises_llm_error(response: httpx.Response) -> None:
    with pytest.raises(LLMError):
        make_client(Server([response]), []).chat(MESSAGES)


def test_authorization_header_sent_only_with_api_key() -> None:
    with_key, without_key = Server([ok_response()]), Server([ok_response()])

    make_client(with_key, [], api_key="secret-token").chat(MESSAGES)
    make_client(without_key, [], api_key=None).chat(MESSAGES)

    assert with_key.requests[0].headers["Authorization"] == "Bearer secret-token"
    assert "Authorization" not in without_key.requests[0].headers


def test_empty_api_key_is_treated_as_missing() -> None:
    server = Server([ok_response()])
    make_client(server, [], api_key="").chat(MESSAGES)
    assert "Authorization" not in server.requests[0].headers


def test_negative_max_retries_is_rejected() -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleClient(BASE_URL, "m", max_retries=-1)


def test_from_generation_config_reads_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_LLM_KEY", "from-env")
    cfg = GenerationConfig(
        backend="openai", base_url="http://gpu:8000/v1/", model="qwen", api_key_env="MY_LLM_KEY",
        timeout_s=12.5, max_retries=5,
    )

    client = OpenAICompatibleClient.from_config(cfg)

    assert client.base_url == "http://gpu:8000/v1"
    assert client.model == "qwen"
    assert client.api_key == "from-env"
    assert client.timeout_s == 12.5
    assert client.max_retries == 5
    client.close()


def test_from_judge_config_without_key_in_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    cfg = JudgeConfig(enabled=True, base_url="http://judge:9000/v1", model="judge-model")

    with OpenAICompatibleClient.from_config(cfg) as client:
        assert client.api_key is None
        assert client.model == "judge-model"
        assert client.max_retries == 3  # у JudgeConfig нет своего значения
