"""Тесты LLM-судьи: устойчивый разбор JSON, валидация оценок, повторы, каппа Коэна.

Сеть не нужна: вместо LLM подставляется фейковый клиент с заранее заданными ответами.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from sklearn.metrics import cohen_kappa_score

from kbrag.eval.judge import (
    NO_REFERENCE_TEXT,
    JudgeError,
    JudgeVerdict,
    LLMJudge,
    cohen_kappa,
    extract_json_object,
    parse_verdict,
)
from kbrag.types import Chunk, ScoredChunk

VALID = '{"faithfulness": 5, "correctness": 4, "rationale": "Ответ подтверждён фрагментом [1]."}'


@dataclass
class FakeClient:
    """Возвращает заранее заданные ответы по очереди и запоминает, что у него спросили."""

    replies: list[str]
    calls: list[dict] = field(default_factory=list)

    def chat(self, messages, temperature=0.0, max_tokens=512, response_format=None):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
        )
        return SimpleNamespace(text=self.replies[len(self.calls) - 1])


def scored(text: str, cid: str = "it-vpn::001") -> ScoredChunk:
    chunk = Chunk(chunk_id=cid, doc_id="it-vpn", text=text, start=0, end=len(text))
    return ScoredChunk(chunk=chunk, score=1.0, source="bm25")


# ---------------------------------------------------------------------------
# Разбор ответа модели
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        VALID,
        f"```json\n{VALID}\n```",
        f"Вот моя оценка:\n```json\n{VALID}\n```\nНадеюсь, это поможет!",
        f"```\n{VALID}\n```",  # блок кода без пометки json
        f"Оценка: {VALID} — конец.",  # JSON посреди текста без блока кода
        f"Формат {{x}} понятен. {VALID}",  # «мусорная» фигурная скобка до настоящего JSON
    ],
)
def test_parse_verdict_is_robust_to_wrapping(raw: str) -> None:
    verdict = parse_verdict(raw)
    assert verdict == JudgeVerdict(5, 4, "Ответ подтверждён фрагментом [1].")


def test_parse_verdict_handles_braces_inside_strings_and_nested_objects() -> None:
    raw = '{"faithfulness": 3, "correctness": 2, "rationale": "Шаблон {name} не раскрыт", "meta": {"a": 1}}'
    verdict = parse_verdict(raw)
    assert (verdict.faithfulness, verdict.correctness) == (3, 2)
    assert verdict.rationale == "Шаблон {name} не раскрыт"


def test_extract_json_prefers_object_with_scores() -> None:
    raw = 'Пример: {"note": "это не оценка"}. Итог: {"faithfulness": 1, "correctness": 1}'
    assert extract_json_object(raw) == {"faithfulness": 1, "correctness": 1}


@pytest.mark.parametrize(
    ("faithfulness", "correctness"),
    [('"4"', "4.0"), ("4", '" 4 "')],
)
def test_parse_verdict_accepts_integer_like_scores(faithfulness: str, correctness: str) -> None:
    verdict = parse_verdict(f'{{"faithfulness": {faithfulness}, "correctness": {correctness}}}')
    assert (verdict.faithfulness, verdict.correctness) == (4, 4)
    assert verdict.rationale == ""  # rationale необязателен


@pytest.mark.parametrize(
    "raw",
    [
        '{"faithfulness": 6, "correctness": 3}',  # вне шкалы сверху
        '{"faithfulness": 0, "correctness": 3}',  # вне шкалы снизу
        '{"faithfulness": 3.5, "correctness": 3}',  # не целое
        '{"faithfulness": true, "correctness": 3}',  # bool — не оценка
        '{"faithfulness": "4/5", "correctness": 3}',  # не число
        '{"faithfulness": null, "correctness": 3}',
        '{"faithfulness": 4}',  # нет correctness
        "Ответ хороший, ставлю пять.",  # нет JSON вовсе
        "",
    ],
)
def test_parse_verdict_rejects_invalid(raw: str) -> None:
    with pytest.raises(JudgeError):
        parse_verdict(raw)


def test_judge_verdict_validates_range_on_construction() -> None:
    with pytest.raises(JudgeError):
        JudgeVerdict(faithfulness=7, correctness=3)
    with pytest.raises(ValueError):  # JudgeError — подкласс ValueError
        JudgeVerdict(faithfulness=3, correctness=0)


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


def test_judge_sends_question_answer_reference_and_context() -> None:
    client = FakeClient([f"```json\n{VALID}\n```"])
    judge = LLMJudge(client)

    verdict = judge.judge(
        question="Какой клиент нужен для VPN?",
        answer="Клиент «Шлюз» [1].",
        contexts=[scored("Для подключения к VPN используется клиент «Шлюз».")],
        reference="Клиент «Шлюз».",
    )

    assert verdict.faithfulness == 5
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["temperature"] == 0.0  # детерминированная оценка
    assert call["response_format"] is None  # json_mode выключен по умолчанию
    system, user = call["messages"]
    assert system["role"] == "system" and "faithfulness" in system["content"]
    assert "Какой клиент нужен для VPN?" in user["content"]
    assert "Клиент «Шлюз» [1]." in user["content"]
    assert "Эталонный ответ: Клиент «Шлюз»." in user["content"]
    assert "[1] Для подключения к VPN используется клиент «Шлюз»." in user["content"]
    assert "<context>" in user["content"] and "<answer>" in user["content"]


def test_judge_marks_empty_reference_as_unanswerable() -> None:
    client = FakeClient([VALID])
    LLMJudge(client).judge("Сколько стоит обед?", "В базе знаний нет ответа на этот вопрос.", [], "")
    assert NO_REFERENCE_TEXT in client.calls[0]["messages"][1]["content"]


def test_judge_accepts_plain_strings_and_truncates_context() -> None:
    client = FakeClient([VALID])
    judge = LLMJudge(client, max_context_chars=50)
    judge.judge("q", "a", ["короткий фрагмент", "x" * 500], "ref")
    user = client.calls[0]["messages"][1]["content"]
    assert "[1] короткий фрагмент" in user
    assert "[2]" not in user  # второй фрагмент не влез в лимит и отброшен целиком


def test_judge_truncates_single_oversized_context_instead_of_dropping_it() -> None:
    client = FakeClient([VALID])
    LLMJudge(client, max_context_chars=20).judge("q", "a", ["y" * 100], "ref")
    user = client.calls[0]["messages"][1]["content"]
    assert "[1] " + "y" * 16 in user
    assert "y" * 17 not in user


def test_judge_retries_after_unparseable_reply() -> None:
    client = FakeClient(["Извините, я не могу ответить в JSON.", VALID])
    verdict = LLMJudge(client, max_attempts=2).judge("q", "a", [], "ref")
    assert verdict.correctness == 4
    assert len(client.calls) == 2


def test_judge_gives_up_after_max_attempts() -> None:
    client = FakeClient(["мусор", '{"faithfulness": 9, "correctness": 9}', VALID])
    with pytest.raises(JudgeError, match="2 попыток"):
        LLMJudge(client, max_attempts=2).judge("q", "a", [], "ref")
    assert len(client.calls) == 2  # третий ответ так и не запрошен


def test_judge_json_mode_requests_json_object() -> None:
    client = FakeClient([VALID])
    LLMJudge(client, json_mode=True).judge("q", "a", [], "ref")
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_judge_rejects_zero_attempts() -> None:
    with pytest.raises(ValueError):
        LLMJudge(FakeClient([]), max_attempts=0)


# ---------------------------------------------------------------------------
# Каппа Коэна
# ---------------------------------------------------------------------------


def test_cohen_kappa_perfect_agreement() -> None:
    assert cohen_kappa([5, 4, 1, 3], [5, 4, 1, 3]) == 1.0


def test_cohen_kappa_textbook_example() -> None:
    # 50 ответов: оба «да» — 20, оба «нет» — 15, A «да»/B «нет» — 5, A «нет»/B «да» — 10.
    # p_o = 35/50 = 0.7; A ставит «да» в 25/50, B — в 30/50;
    # p_e = 0.5·0.6 + 0.5·0.4 = 0.5; kappa = (0.7 − 0.5) / (1 − 0.5) = 0.4
    a = ["да"] * 20 + ["да"] * 5 + ["нет"] * 10 + ["нет"] * 15
    b = ["да"] * 20 + ["нет"] * 5 + ["да"] * 10 + ["нет"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_systematic_disagreement_is_negative() -> None:
    # p_o = 0, p_e = 0.5·0.5 + 0.5·0.5 = 0.5 → kappa = (0 − 0.5) / 0.5 = −1
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)


def test_cohen_kappa_high_raw_agreement_can_mean_little() -> None:
    # Совпадение 8 из 10 (80%), но почти все оценки «5»: kappa ≈ −0.111 — согласие хуже случайного.
    a = [5] * 9 + [4]
    b = [5] * 8 + [4, 5]
    assert cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))
    assert cohen_kappa(a, b) < 0


def test_cohen_kappa_matches_sklearn_on_judge_scores() -> None:
    human = [5, 4, 4, 3, 1, 2, 5, 5, 3, 4, 2, 1]
    judge = [5, 4, 3, 3, 1, 1, 5, 4, 3, 4, 2, 2]
    assert cohen_kappa(human, judge) == pytest.approx(cohen_kappa_score(human, judge))


def test_cohen_kappa_constant_identical_labels() -> None:
    # p_e = 1 → формула даёт 0/0; полное согласие считаем каппой 1.
    assert cohen_kappa([5, 5, 5], [5, 5, 5]) == 1.0


def test_cohen_kappa_invalid_input() -> None:
    with pytest.raises(ValueError):
        cohen_kappa([1, 2], [1])
    with pytest.raises(ValueError):
        cohen_kappa([], [])


@pytest.mark.parametrize("weights", ["linear", "quadratic"])
def test_weighted_cohen_kappa_matches_sklearn(weights: str) -> None:
    human = [5, 4, 4, 3, 1, 2, 5, 5, 3, 4, 2, 1]
    judge = [5, 4, 3, 3, 1, 1, 5, 4, 3, 4, 2, 2]
    expected = cohen_kappa_score(human, judge, labels=[1, 2, 3, 4, 5], weights=weights)
    assert cohen_kappa(human, judge, weights=weights) == pytest.approx(expected)


def test_weighted_kappa_penalizes_near_misses_less_than_far_misses() -> None:
    # Регрессия: невзвешенная каппа не отличала «4 против 5» от «1 против 5» на шкале 1–5.
    # Частоты оценок у near и far одинаковые, совпадений тоже поровну (8 из 10) — различается
    # только величина промахов: near меняет местами 5 и 4, far — 5 и 1.
    human = [5, 4, 3, 2, 1, 5, 4, 3, 2, 1]
    near = [4, 5, 3, 2, 1, 5, 4, 3, 2, 1]
    far = [1, 4, 3, 2, 5, 5, 4, 3, 2, 1]
    assert cohen_kappa(human, near) == pytest.approx(cohen_kappa(human, far))
    assert cohen_kappa(human, near, weights="quadratic") > cohen_kappa(human, far, weights="quadratic")


def test_weighted_kappa_invalid_input() -> None:
    assert cohen_kappa([3, 3], [3, 3], weights="quadratic") == 1.0
    with pytest.raises(ValueError, match="weights"):
        cohen_kappa([1, 2], [1, 2], weights="cubic")
    with pytest.raises(ValueError, match="числовых"):
        cohen_kappa(["да", "нет"], ["да", "да"], weights="linear")
