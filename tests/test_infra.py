"""Проверки «обвязки» проекта: Makefile, .gitignore, CI-скрипты из scripts/.

Это не логика RAG, но её поломки так же реальны: например, если порог отказа не попадает
в git, сервис из свежего клона молча перестаёт отвечать «нет ответа». Docker и make в тестах
не запускаем (их может не быть на машине) — проверяем файлы, от которых они зависят.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from packaging.requirements import Requirement

from kbrag.config import load_config
from kbrag.pipeline import load_calibration

ROOT = Path(__file__).resolve().parents[1]
OFFLINE_CALIBRATION = "reports/calibration_offline.json"


def load_script(name: str) -> ModuleType:
    """Импортирует scripts/<name>.py как модуль (scripts/ — не пакет, обычный import не сработает)."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_recipe(target: str) -> str:
    """Команды цели Makefile (строки с табом сразу после ``target:``)."""
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(target)}:.*\n((?:\t.*\n?)+)", text, flags=re.MULTILINE)
    assert match, f"в Makefile нет цели {target}"
    return match.group(1)


def git_ignores(path: str) -> bool:
    """Игнорирует ли .gitignore путь (по шаблонам, даже если файл уже в git)."""
    if shutil.which("git") is None:
        pytest.skip("git не установлен")
    proc = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT, capture_output=True, check=False
    )
    if proc.returncode not in (0, 1):  # 128 — например, это не git-репозиторий (архив с кодом)
        pytest.skip(f"git check-ignore недоступен: {proc.stderr.decode(errors='replace').strip()}")
    return proc.returncode == 0


# --- Калибровка порога отказа: файл в git, `make eval` его обновляет, CI проверяет -----------------


def test_make_eval_saves_calibration() -> None:
    # Без флага `make eval` не пишет порог, и сервис после `make serve` никогда не отказывается.
    assert "--save-calibration" in make_recipe("eval")


def test_calibration_is_committed_but_scratch_reports_are_not() -> None:
    assert not git_ignores(OFFLINE_CALIBRATION)
    assert not git_ignores("reports/calibration.json")  # порог для configs/default.yaml
    assert git_ignores("reports/grid_offline/leaderboard.md")
    assert git_ignores("reports/_tmp/results.json")


def test_committed_calibration_matches_offline_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # Отпечаток в файле должен совпадать с текущими настройками поиска offline.yaml, иначе сервис
    # порог проигнорирует. Упал тест — запустите `make eval` и закоммитьте файл калибровки.
    monkeypatch.chdir(ROOT)  # calibration_path в конфиге относительный — от корня проекта
    cfg = load_config("configs/offline.yaml")
    assert cfg.abstention.threshold == "auto"
    assert cfg.abstention.calibration_path == OFFLINE_CALIBRATION

    threshold = load_calibration(cfg)

    assert threshold is not None, "порог не загрузился: файла нет или он посчитан для другого поиска"
    assert 0.0 < threshold < 1.0  # офлайн-уверенность — косинус LSA


@pytest.fixture(scope="module")
def check_calibration() -> ModuleType:
    return load_script("check_calibration")


CALIBRATION = {"fingerprint": "62a82ed2f1de", "score_source": "dense", "threshold": 0.4985}


def test_same_calibration_passes_despite_metadata(check_calibration: ModuleType) -> None:
    fresh = {**CALIBRATION, "threshold": 0.49850001, "experiment": "ci", "created_at": "2030-01-01"}
    assert check_calibration.compare_calibrations(CALIBRATION, fresh) == []


@pytest.mark.parametrize(
    "change",
    [
        {"threshold": 0.52},  # правка корпуса сдвинула порог
        {"threshold": None},  # порог больше не подбирается
        {"fingerprint": "000000000000"},  # поменяли настройки поиска
        {"score_source": "bm25"},
    ],
)
def test_stale_calibration_is_reported(check_calibration: ModuleType, change: dict) -> None:
    problems = check_calibration.compare_calibrations(CALIBRATION, {**CALIBRATION, **change})
    assert len(problems) == 1
    assert next(iter(change)) in problems[0]


def test_both_thresholds_none_is_consistent(check_calibration: ModuleType) -> None:
    no_threshold = {**CALIBRATION, "threshold": None}
    assert check_calibration.compare_calibrations(no_threshold, dict(no_threshold)) == []


def test_check_calibration_exit_codes(
    check_calibration: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    committed, fresh = tmp_path / "committed.json", tmp_path / "fresh.json"
    committed.write_text(json.dumps(CALIBRATION), encoding="utf-8")

    fresh.write_text(json.dumps(CALIBRATION), encoding="utf-8")
    assert check_calibration.main([str(committed), str(fresh)]) == 0

    fresh.write_text(json.dumps({**CALIBRATION, "threshold": 0.6}), encoding="utf-8")
    assert check_calibration.main([str(committed), str(fresh)]) == 1
    assert "make eval" in capsys.readouterr().err  # сообщение подсказывает, как починить

    assert check_calibration.main([str(committed), str(tmp_path / "missing.json")]) == 1


# --- Форматирование: локальный `make lint` и CI проверяют одно и то же --------------------------


def test_lint_checks_formatting_locally_and_in_ci() -> None:
    assert "ruff format --check" in make_recipe("lint")
    assert "ruff format --check" in (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


# --- Зависимости: нижние границы не пускают версии с известными уязвимостями ----------------------


def declared_requirements() -> dict[str, Requirement]:
    tomllib = pytest.importorskip("tomllib")  # есть в стандартной библиотеке с Python 3.11
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    lines = project["dependencies"] + [
        r for extra in project["optional-dependencies"].values() for r in extra
    ]
    return {req.name: req for req in map(Requirement, lines)}


@pytest.mark.parametrize(
    ("package", "vulnerable", "fixed"),
    [
        ("starlette", "0.49.0", "0.49.1"),  # CVE-2025-62727
        ("h11", "0.15.0", "0.16.0"),  # CVE-2025-43859
        ("torch", "2.5.1", "2.6.0"),  # CVE-2025-32434
    ],
)
def test_dependency_floors_exclude_vulnerable_versions(package: str, vulnerable: str, fixed: str) -> None:
    spec = declared_requirements()[package].specifier
    assert not spec.contains(vulnerable)
    assert spec.contains(fixed)


def test_torch_floor_is_the_same_everywhere() -> None:
    # torch ставится отдельной командой (CPU-индекс) в Dockerfile и Makefile — граница не должна разъехаться.
    expected = str(declared_requirements()["torch"].specifier)
    for name in ("Dockerfile", "Makefile"):
        floors = re.findall(r'"torch(>=[\d.]+)"', (ROOT / name).read_text(encoding="utf-8"))
        assert floors and set(floors) == {expected}, f"{name}: {floors}, ожидалось {expected}"


# --- Docker и секреты: что попадает в git и в образ, кому открыт порт ------------------------------


def dockerignore_patterns() -> set[str]:
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")}


def compose_services() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]


@pytest.mark.parametrize("secret_file", [".env", ".env.local", ".env.prod"])
def test_env_files_never_reach_git_or_image(secret_file: str) -> None:
    assert git_ignores(secret_file)
    patterns = dockerignore_patterns()
    assert ".env" in patterns and ".env.*" in patterns


def test_env_template_stays_in_git() -> None:
    assert not git_ignores(".env.example")


def test_image_gets_corpus_and_calibration_but_not_gold() -> None:
    patterns = dockerignore_patterns()
    assert "data/eval" in patterns  # эталон сервису не нужен
    assert "!reports/calibration*.json" in patterns  # порог отказа сервису нужен


def test_ports_are_published_on_localhost_only() -> None:
    # У /ask нет авторизации: `-p 8080:8080` открыл бы сервис всей сети.
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    published = re.findall(r"docker run[^\n]*?\s-p\s+(\S+)", dockerfile)  # не путать с `mkdir -p`
    assert published, "в шапке Dockerfile нет примеров docker run"
    assert all(port.startswith("127.0.0.1:") for port in published), published
    for name, service in compose_services().items():
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), f"{name}: {port}"


def test_compose_hardening() -> None:
    services = compose_services()
    assert services["app"]["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in services["app"]["security_opt"]
    assert "ipc" not in services["llm"]  # отдельный /dev/shm (shm_size) вместо общего IPC с хостом
    assert services["llm"]["shm_size"]


# --- scripts/check_part.py: авторские проверки эталона --------------------------------------------

PART_DOC = (
    "---\nid: hr-test\ntitle: Тестовый документ\ndepartment: HR\nstatus: active\nupdated: 2025-01-01\n---\n\n"
    "# Тестовый документ\n\nОтпуск длится 28 календарных дней и делится на части по желанию сотрудника.\n"
)


def write_part(path: Path, quote: str = "Отпуск длится 28 календарных дней") -> Path:
    record = {
        "id": "q-1",
        "question": "Сколько длится отпуск?",
        "answer": "28 календарных дней.",
        "key_facts": ["28 календарных дней"],
        "evidence": [{"doc_id": "hr-test", "quote": quote}],
        "type": "numeric",
    }
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def check_part() -> ModuleType:
    return load_script("check_part")


def test_check_part_reads_document_with_bom(
    check_part: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Блокнот Windows сохраняет UTF-8 с BOM; основной загрузчик это понимает — скрипт тоже должен.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "hr-test.md").write_text("﻿" + PART_DOC, encoding="utf-8")
    part = write_part(tmp_path / "part.jsonl")

    assert check_part.main([str(part), "--corpus", str(corpus)]) == 0
    assert "ERROR" not in capsys.readouterr().out


def test_check_part_counts_missing_front_matter_as_error(
    check_part: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Раньше скрипт печатал ERROR, но всё равно заканчивал «OK» с кодом 0.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "no-meta.md").write_text("# Документ без шапки\n\nТекст.\n", encoding="utf-8")
    part = tmp_path / "empty.jsonl"
    part.write_text("", encoding="utf-8")

    assert check_part.main([str(part), "--corpus", str(corpus)]) == 1
    out = capsys.readouterr().out
    assert "нет front matter" in out
    assert "ОШИБОК: 1" in out


def test_check_part_warning_states_the_range_it_checks(
    check_part: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "hr-test.md").write_text(PART_DOC, encoding="utf-8")
    part = write_part(tmp_path / "part.jsonl", quote="28 календарных дней")  # 3 слова < 5

    assert check_part.main([str(part), "--corpus", str(corpus)]) == 0  # предупреждение не ошибка
    lo, hi = check_part.QUOTE_WORDS
    assert f"цитата из 3 слов (ожидается {lo}–{hi})" in capsys.readouterr().out
