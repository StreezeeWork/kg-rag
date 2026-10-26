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
