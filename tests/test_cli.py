"""Тесты командной строки: вызываем ``main([...])`` напрямую и проверяем коды возврата и файлы.

Все команды работают на мини-корпусе и мини-эталоне из conftest через временный YAML-конфиг
(BM25 + extractive: офлайн, без моделей и за доли секунды).
"""

from __future__ import annotations

import io
import json
import os
import runpy
import sys
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI

from kbrag.api.app import AskResponse
from kbrag.cli import EXIT_ERROR, EXIT_OK, EXIT_QUALITY_GATE, main
from kbrag.config import deep_merge, load_config
from kbrag.eval.dataset import assign_splits, item_from_record, load_gold
from kbrag.ingest import corpus_fingerprint, load_corpus
from kbrag.pipeline import calibration_fingerprint, load_calibration

VACATION_Q = "Сколько календарных дней длится ежегодный отпуск?"

# Неотвечаемый dev-вопрос: без него на dev один класс и порог отказа подбирать не на чем.
EXTRA_DEV_UNANSWERABLE = {
    "id": "t-006",
    "question": "Какой пароль от Wi-Fi в переговорке на пятом этаже?",
    "answer": "",
    "key_facts": [],
    "evidence": [],
    "type": "unanswerable",
    "split": "dev",
}


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return path


def write_config(path: Path, base: dict, **overrides) -> Path:
    path.write_text(yaml.safe_dump(deep_merge(base, overrides), allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture()
def base_config(tmp_path: Path, mini_corpus_dir: Path, mini_gold_path: Path) -> dict:
    return {
        "corpus_dir": str(mini_corpus_dir),
        "gold_path": str(mini_gold_path),
        "cache_dir": str(tmp_path / "cache"),
        "reports_dir": str(tmp_path / "reports"),
        "retrieval": {"mode": "bm25", "top_k": 3},
        "rerank": {"enabled": False},
        "generation": {"backend": "extractive"},
        "abstention": {"threshold": "auto", "calibration_path": str(tmp_path / "calibration.json")},
    }


@pytest.fixture()
def cli_config(tmp_path: Path, base_config: dict) -> Path:
    return write_config(tmp_path / "mini.yaml", base_config)


# ---------------------------------------------------------------------------
# Общее поведение
# ---------------------------------------------------------------------------


def test_command_is_required() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_missing_explicit_config_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate-data", "--config", str(tmp_path / "nope.yaml")]) == EXIT_ERROR
    assert "конфиг не найден" in capsys.readouterr().err


def test_config_defaults_to_kbrag_config_env(cli_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KBRAG_CONFIG", str(cli_config))
    assert main(["validate-data"]) == EXIT_OK


def test_python_dash_m_passes_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["kbrag", "validate-data", "--config", str(tmp_path / "nope.yaml")])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("kbrag", run_name="__main__")
    assert exc.value.code == EXIT_ERROR


def test_output_is_utf8_even_when_stdout_is_a_cp1251_pipe(
    tmp_path: Path, cli_config: Path, parts_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Так выглядит stdout на Windows, когда вывод идёт в pipe или файл (Git Bash, `> answer.json`):
    # кодировка локали. Раньше «→» в сводке build-gold давал UnicodeEncodeError и код 1.
    raw = io.BytesIO()
    fake_stdout = io.TextIOWrapper(raw, encoding="cp1251")
    monkeypatch.setattr(sys, "stdout", fake_stdout)

    assert main(build_gold_args(cli_config, parts_dir, tmp_path / "gold.jsonl")) == EXIT_OK

    fake_stdout.flush()
    assert "→" in raw.getvalue().decode("utf-8")


def test_quiet_mode_silences_huggingface_noise(cli_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HUB_VERBOSITY", "TRANSFORMERS_VERBOSITY", "HF_HUB_DISABLE_PROGRESS_BARS"):
        monkeypatch.delenv(name, raising=False)
    assert main(["validate-data", "--config", str(cli_config)]) == EXIT_OK
    assert os.environ["HF_HUB_VERBOSITY"] == "error"
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"


def test_neural_config_without_neural_extras_fails_fast_with_offline_hint(
    tmp_path: Path, base_config: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import kbrag.pipeline
    import kbrag.retrieval

    def must_not_build(*args: object, **kwargs: object) -> None:
        raise AssertionError("индекс не должен строиться: ошибка должна быть сразу")

    monkeypatch.setattr(kbrag.retrieval, "neural_available", lambda: False)
    monkeypatch.setattr(kbrag.pipeline.RAGPipeline, "from_config", must_not_build)
    neural = write_config(tmp_path / "neural.yaml", base_config, rerank={"enabled": True})

    assert main(["ask", "--config", str(neural), VACATION_Q]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "install-neural" in err and "configs/offline.yaml" in err


# ---------------------------------------------------------------------------
# validate-data
# ---------------------------------------------------------------------------


def test_validate_data_prints_corpus_and_gold_stats(
    cli_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["validate-data", "--config", str(cli_config)]) == EXIT_OK

    out = capsys.readouterr().out
    assert "документов: 4" in out
    assert "по отделам: HR 3, IT 1" in out
    assert "по статусам: active 3, archived 1" in out
    assert "фрагментов: " in out
    assert "Эталонный набор: 5 вопросов" in out
    unanswerable_row = next(line for line in out.splitlines() if line.strip().startswith("unanswerable"))
    assert unanswerable_row.split()[1:] == ["0", "1", "1"]  # dev, test, всего
    assert "в порядке" in out


def test_validate_data_fails_on_broken_gold(
    tmp_path: Path, base_config: dict, mini_gold_records: list[dict], capsys: pytest.CaptureFixture[str]
) -> None:
    mini_gold_records[0]["evidence"] = [{"doc_id": "hr-vacation", "quote": "Такой фразы в документе нет."}]
    mini_gold_records[3]["evidence"] = [{"doc_id": "no-such-doc", "quote": "Шлюз"}]
    gold = write_jsonl(tmp_path / "broken.jsonl", mini_gold_records)
    config = write_config(tmp_path / "broken.yaml", base_config, gold_path=str(gold))

    assert main(["validate-data", "--config", str(config)]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "t-001" in err and "цитата не найдена" in err
    assert "t-004" in err and "no-such-doc" in err


def test_validate_data_hints_to_build_gold_when_file_is_missing(
    tmp_path: Path, base_config: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path / "nogold.yaml", base_config, gold_path=str(tmp_path / "missing.jsonl"))
    assert main(["validate-data", "--config", str(config)]) == EXIT_ERROR
    assert "kbrag build-gold" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# build-gold
# ---------------------------------------------------------------------------


@pytest.fixture()
def parts_dir(tmp_path: Path, mini_gold_records: list[dict]) -> Path:
    """Части эталона, как в data/eval/parts: без поля split, в двух файлах."""
    records = [{k: v for k, v in r.items() if k != "split"} for r in mini_gold_records]
    parts = tmp_path / "parts"
    write_jsonl(parts / "a_hr.jsonl", records[:3])
    write_jsonl(parts / "b_rest.jsonl", records[3:])
    return parts


def build_gold_args(config: Path, parts: Path, out: Path, *extra: str) -> list[str]:
    return ["build-gold", "--config", str(config), "--parts", str(parts), "--out", str(out), *extra]


def test_build_gold_merges_parts_and_assigns_splits(
    tmp_path: Path, cli_config: Path, parts_dir: Path, mini_gold_records: list[dict]
) -> None:
    out = tmp_path / "built" / "gold.jsonl"
    assert main(build_gold_args(cli_config, parts_dir, out)) == EXIT_OK

    items = load_gold(out)
    assert [item.qid for item in items] == ["t-001", "t-002", "t-003", "t-004", "t-005"]
    expected = assign_splits([item_from_record(r) for r in mini_gold_records], dev_fraction=0.3)
    assert [item.split for item in items] == [item.split for item in expected]
    assert {item.split for item in items} <= {"dev", "test"}

    # Повторная сборка даёт тот же файл байт в байт — набор воспроизводим.
    first_bytes = out.read_bytes()
    assert main(build_gold_args(cli_config, parts_dir, out)) == EXIT_OK
    assert out.read_bytes() == first_bytes


def test_build_gold_dev_fraction_changes_split(tmp_path: Path, cli_config: Path, parts_dir: Path) -> None:
    out = tmp_path / "built" / "gold.jsonl"
    assert main(build_gold_args(cli_config, parts_dir, out, "--dev-fraction", "1.0")) == EXIT_OK
    assert {item.split for item in load_gold(out)} == {"dev"}

    with pytest.raises(SystemExit) as exc:
        main(build_gold_args(cli_config, parts_dir, out, "--dev-fraction", "1.5"))
    assert exc.value.code == 2


def test_build_gold_check_detects_stale_file(
    tmp_path: Path, cli_config: Path, parts_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "built" / "gold.jsonl"
    assert main(build_gold_args(cli_config, parts_dir, out)) == EXIT_OK
    built = out.read_bytes()

    assert main(build_gold_args(cli_config, parts_dir, out, "--check")) == EXIT_OK
    assert out.read_bytes() == built  # --check ничего не пишет

    # git на Windows может заменить \n на \r\n — это не делает файл устаревшим.
    out.write_bytes(built.replace(b"\n", b"\r\n"))
    assert main(build_gold_args(cli_config, parts_dir, out, "--check")) == EXIT_OK

    # Удалили последний вопрос — файл устарел.
    out.write_bytes(b"".join(built.splitlines(keepends=True)[:-1]))
    capsys.readouterr()
    assert main(build_gold_args(cli_config, parts_dir, out, "--check")) == EXIT_ERROR
    assert "t-005" in capsys.readouterr().err


def test_build_gold_check_fails_when_file_is_missing(
    tmp_path: Path, cli_config: Path, parts_dir: Path
) -> None:
    out = tmp_path / "built" / "gold.jsonl"
    assert main(build_gold_args(cli_config, parts_dir, out, "--check")) == EXIT_ERROR
    assert not out.exists()


def test_build_gold_refuses_to_write_invalid_set(
    tmp_path: Path, cli_config: Path, parts_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_jsonl(
        parts_dir / "c_bad.jsonl",
        [
            {
                "id": "t-001",  # дубликат id из a_hr.jsonl
                "question": "Вопрос?",
                "answer": "Ответ.",
                "key_facts": ["факт"],
                "evidence": [{"doc_id": "no-such-doc", "quote": "цитата"}],
                "type": "factoid",
            }
        ],
    )
    out = tmp_path / "built" / "gold.jsonl"
    assert main(build_gold_args(cli_config, parts_dir, out)) == EXIT_ERROR
    assert not out.exists()
    err = capsys.readouterr().err
    assert "повторяющийся id" in err and "no-such-doc" in err


def test_build_gold_fails_on_empty_parts_dir(tmp_path: Path, cli_config: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(build_gold_args(cli_config, empty, tmp_path / "built" / "gold.jsonl")) == EXIT_ERROR


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


def test_ask_json_prints_api_response_format(cli_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", VACATION_Q, "--config", str(cli_config), "--json", "--top-k", "2"]) == EXIT_OK

    payload = json.loads(capsys.readouterr().out)  # в stdout — только JSON
    assert set(payload) == set(AskResponse.model_fields)
    AskResponse.model_validate(payload)
    assert "28 календарных дней" in payload["answer"]
    assert payload["abstained"] is False
    assert len(payload["contexts"]) == 2
    assert payload["contexts"][0]["doc_id"] == "hr-vacation"


def test_ask_text_mode_lists_sources(cli_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", VACATION_Q, "--config", str(cli_config)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "28 календарных дней" in out
    assert "Источники:" in out
    assert "[1] Ежегодный оплачиваемый отпуск" in out


def test_ask_rejects_bad_input(cli_config: Path) -> None:
    assert main(["ask", "   ", "--config", str(cli_config)]) == EXIT_ERROR
    with pytest.raises(SystemExit) as exc:
        main(["ask", VACATION_Q, "--config", str(cli_config), "--top-k", "0"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def test_eval_writes_report_and_prints_summary(
    tmp_path: Path, cli_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    assert main(["eval", "--config", str(cli_config), "--name", "mini", "--out", str(out)]) == EXIT_OK

    report_dir = out / "mini"
    assert (report_dir / "results.json").is_file()
    assert (report_dir / "leaderboard.md").is_file()
    per_question = (report_dir / "per_question" / "mini.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(per_question) == 5
    stdout = capsys.readouterr().out
    assert "Эксперимент: mini" in stdout
    assert "recall@5" in stdout


def test_eval_defaults_to_config_name_and_reports_dir(tmp_path: Path, cli_config: Path) -> None:
    assert main(["eval", "--config", str(cli_config)]) == EXIT_OK
    # Имя прогона — имя файла конфига (mini.yaml), каталог — reports_dir из конфига.
    assert (tmp_path / "reports" / "mini" / "leaderboard.md").is_file()


def test_eval_quality_gate(tmp_path: Path, cli_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = str(tmp_path / "out")
    failing = ["eval", "--config", str(cli_config), "--out", out, "--fail-under", "recall@5=1.01"]
    assert main(failing) == EXIT_QUALITY_GATE
    assert "FAIL" in capsys.readouterr().out
    assert (tmp_path / "out" / "mini" / "leaderboard.md").is_file()  # отчёт пишется и при провале

    passing = ["eval", "--config", str(cli_config), "--out", out, "--fail-under", "recall@5=0.0", "mrr=0.0"]
    assert main(passing) == EXIT_OK


@pytest.mark.parametrize(
    ("gate", "message"),
    [
        ("recall@50=0.5", "неизвестные метрики"),
        ("latency_p95_ms=500", "меньше — лучше"),
        ("stale_citation_rate=0.1", "меньше — лучше"),
    ],
)
def test_eval_rejects_bad_gate_metric_before_running(
    tmp_path: Path, cli_config: Path, capsys: pytest.CaptureFixture[str], gate: str, message: str
) -> None:
    out = tmp_path / "out"
    assert main(["eval", "--config", str(cli_config), "--out", str(out), "--fail-under", gate]) == EXIT_ERROR
    assert message in capsys.readouterr().err
    assert not out.exists()  # оценка даже не запускалась


def test_eval_rejects_malformed_gate(cli_config: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["eval", "--config", str(cli_config), "--fail-under", "recall@5"])
    assert exc.value.code == 2


def test_eval_save_calibration_writes_file_the_service_can_read(
    tmp_path: Path, base_config: dict, mini_gold_records: list[dict]
) -> None:
    gold = write_jsonl(tmp_path / "gold_dev_unans.jsonl", [*mini_gold_records, EXTRA_DEV_UNANSWERABLE])
    config_path = write_config(tmp_path / "calib.yaml", base_config, gold_path=str(gold))

    assert main(["eval", "--config", str(config_path), "--name", "calib", "--save-calibration"]) == EXIT_OK

    cfg = load_config(config_path)
    data = json.loads(Path(cfg.abstention.calibration_path).read_text(encoding="utf-8"))
    assert set(data) == {
        "threshold",
        "fingerprint",
        "score_source",
        "corpus_sha256",
        "experiment",
        "created_at",
    }
    assert data["corpus_sha256"] == corpus_fingerprint(load_corpus(cfg.corpus_dir))
    assert isinstance(data["threshold"], float)
    assert data["fingerprint"] == calibration_fingerprint(cfg)
    assert data["score_source"] == "bm25"
    assert data["experiment"] == "calib"
    assert datetime.fromisoformat(data["created_at"]).tzinfo is not None
    # Главное: сервис (pipeline.load_calibration) принимает этот файл и получает тот же порог.
    assert load_calibration(cfg) == pytest.approx(data["threshold"])


@pytest.mark.parametrize(
    ("abstention", "message"),
    [
        ({"threshold": 0.4}, "задан в конфиге"),
        ({"enabled": False}, "выключен"),
        ({"calibration_path": None}, "calibration_path"),
    ],
)
def test_eval_save_calibration_preconditions_are_checked_before_running(
    tmp_path: Path, base_config: dict, capsys: pytest.CaptureFixture[str], abstention: dict, message: str
) -> None:
    config = write_config(tmp_path / "c.yaml", base_config, abstention=abstention)
    out = tmp_path / "out"

    assert main(["eval", "--config", str(config), "--out", str(out), "--save-calibration"]) == EXIT_ERROR

    assert message in capsys.readouterr().err
    assert not out.exists()  # оценка даже не запускалась


def test_eval_save_calibration_is_skipped_when_quality_gate_fails(
    tmp_path: Path, base_config: dict, mini_gold_records: list[dict], capsys: pytest.CaptureFixture[str]
) -> None:
    gold = write_jsonl(tmp_path / "gold_dev_unans.jsonl", [*mini_gold_records, EXTRA_DEV_UNANSWERABLE])
    config = write_config(tmp_path / "calib.yaml", base_config, gold_path=str(gold))
    args = ["eval", "--config", str(config), "--save-calibration", "--fail-under", "recall@5=1.01"]

    assert main(args) == EXIT_QUALITY_GATE

    # Порог из прогона, не прошедшего gate, не должен попасть в сервис.
    assert not Path(load_config(config).abstention.calibration_path).exists()
    assert "не сохранён" in capsys.readouterr().err


def test_eval_does_not_overwrite_calibration_when_no_threshold_was_chosen(
    tmp_path: Path, cli_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # В мини-эталоне на dev нет неотвечаемых вопросов: порог не подбирается. Раньше в файл
    # писалось "threshold": null поверх прежней калибровки.
    calibration = Path(load_config(cli_config).abstention.calibration_path)
    calibration.write_text('{"threshold": 0.5}', encoding="utf-8")

    assert main(["eval", "--config", str(cli_config), "--save-calibration"]) == EXIT_ERROR

    assert calibration.read_text(encoding="utf-8") == '{"threshold": 0.5}'
    assert "не подобран" in capsys.readouterr().err


def test_eval_names_run_after_config_from_env(
    tmp_path: Path, cli_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KBRAG_CONFIG", str(cli_config))
    assert main(["eval"]) == EXIT_OK
    assert (tmp_path / "reports" / "mini" / "leaderboard.md").is_file()  # не «default»


def test_eval_bad_out_dir_fails_before_running(
    tmp_path: Path, cli_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kbrag.eval.runner

    def must_not_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("оценка не должна запускаться с негодным --out")

    monkeypatch.setattr(kbrag.eval.runner, "run_experiment", must_not_run)
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("x", encoding="utf-8")

    assert main(["eval", "--config", str(cli_config), "--out", str(not_a_dir / "sub")]) == EXIT_ERROR


def test_eval_fails_without_gold(
    tmp_path: Path, base_config: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path / "nogold.yaml", base_config, gold_path=str(tmp_path / "missing.jsonl"))
    assert main(["eval", "--config", str(config)]) == EXIT_ERROR
    assert "kbrag build-gold" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------


@pytest.fixture()
def experiments_file(tmp_path: Path, cli_config: Path) -> Path:
    data = {
        "base": str(cli_config),
        "experiments": [
            {"name": "bm25_top3", "overrides": {"retrieval": {"mode": "bm25", "top_k": 3}}},
            # heavy: без --include-heavy пропускается (нейросетевой эксперимент скачивал бы модели).
            {"name": "bm25_heavy", "heavy": True, "overrides": {"retrieval": {"mode": "bm25", "top_k": 2}}},
        ],
    }
    path = tmp_path / "experiments.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_grid_writes_leaderboard_with_skipped_experiments(
    tmp_path: Path, experiments_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "grid"
    assert main(["grid", "--experiments", str(experiments_file), "--out", str(out)]) == EXIT_OK

    leaderboard = (out / "leaderboard.md").read_text(encoding="utf-8")
    assert "bm25_top3" in leaderboard
    assert "Пропущенные эксперименты" in leaderboard and "bm25_heavy" in leaderboard
    assert "пропущен" in capsys.readouterr().out

    assert (
        main(["grid", "--experiments", str(experiments_file), "--out", str(out), "--include-heavy"])
        == EXIT_OK
    )
    assert (out / "per_question" / "bm25_heavy.jsonl").is_file()


def test_grid_only_filters_and_rejects_unknown_names(
    tmp_path: Path, experiments_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "grid"
    assert (
        main(["grid", "--experiments", str(experiments_file), "--out", str(out), "--only", "bm25_top3"])
        == EXIT_OK
    )
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert [e["name"] for e in results["experiments"]] == ["bm25_top3"]

    assert main(["grid", "--experiments", str(experiments_file), "--only", "nope"]) == EXIT_ERROR
    assert "nope" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def test_serve_starts_uvicorn_with_configured_app(
    cli_config: Path, mini_corpus_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    calls: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: calls.update(app=app, host=host, port=port))

    assert main(["serve", "--config", str(cli_config), "--port", "9001"]) == EXIT_OK
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 9001
    assert isinstance(calls["app"], FastAPI)
    state = calls["app"].state.kbrag
    assert state.pipeline is None  # индекс строится при старте uvicorn, а не в CLI
    assert state.cfg.corpus_dir == str(mini_corpus_dir)


def test_serve_startup_failure_returns_documented_error_code(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    def failed_startup(app: object, host: str, port: int) -> None:
        raise SystemExit(3)  # так uvicorn сообщает, что lifespan упал (нет корпуса, битый конфиг)

    monkeypatch.setattr(uvicorn, "run", failed_startup)
    assert main(["serve", "--config", str(cli_config)]) == EXIT_ERROR


def test_serve_logs_info_without_verbose_flag(cli_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    import kbrag.cli

    levels: list[int] = []
    monkeypatch.setattr(kbrag.cli, "_setup_logging", levels.append)
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: None)

    assert main(["serve", "--config", str(cli_config)]) == EXIT_OK
    assert levels == [1]  # INFO: «порог отказа загружен», «сервис готов» — журнал сервиса


def test_serve_rejects_invalid_port(cli_config: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--config", str(cli_config), "--port", "70000"])
    assert exc.value.code == 2
