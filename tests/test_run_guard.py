from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from embodied_vla.training import run_guard
from embodied_vla.training.run_guard import claim_run_directory


def test_run_directory_is_locked_then_released() -> None:
    with TemporaryDirectory() as temporary_directory:
        output_dir = Path(temporary_directory) / "run"
        with claim_run_directory(output_dir):
            lock_path = output_dir / ".run.lock"
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            assert payload["pid"] > 0
            with pytest.raises(FileExistsError):
                with claim_run_directory(output_dir):
                    pass
        assert not lock_path.exists()


def test_nonempty_run_directory_is_rejected() -> None:
    with TemporaryDirectory() as temporary_directory:
        output_dir = Path(temporary_directory) / "run"
        output_dir.mkdir()
        (output_dir / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
        with pytest.raises(FileExistsError, match="not empty"):
            with claim_run_directory(output_dir):
                pass


def test_explicit_resume_replaces_stale_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TemporaryDirectory() as temporary_directory:
        output_dir = Path(temporary_directory) / "run"
        output_dir.mkdir()
        lock_path = output_dir / ".run.lock"
        lock_path.write_text('{"pid": 1234}', encoding="utf-8")
        (output_dir / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(run_guard, "_pid_is_running", lambda _pid: False)

        with claim_run_directory(output_dir, resume=True):
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            assert payload["pid"] > 0
        assert not lock_path.exists()


def test_explicit_resume_rejects_live_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TemporaryDirectory() as temporary_directory:
        output_dir = Path(temporary_directory) / "run"
        output_dir.mkdir()
        (output_dir / ".run.lock").write_text('{"pid": 1234}', encoding="utf-8")
        monkeypatch.setattr(run_guard, "_pid_is_running", lambda _pid: True)

        with pytest.raises(RuntimeError, match="PID 1234 is running"):
            with claim_run_directory(output_dir, resume=True):
                pass
