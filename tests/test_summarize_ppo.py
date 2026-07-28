from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from embodied_vla.cli.summarize_ppo import summarize_ppo_runs


def _write_run(
    root: Path,
    seed: int,
    success_rate: float,
    *,
    include_end_effector_position: bool | None = None,
) -> Path:
    run_dir = root / f"seed_{seed}"
    run_dir.mkdir()
    environment = {"task_level": "reach"}
    if include_end_effector_position is not None:
        environment["include_end_effector_position_in_proprio"] = (
            include_end_effector_position
        )
    summary = {
        "seed": seed,
        "environment": environment,
        "ppo": {"total_steps": 100_000},
        "final_evaluation": {
            "success_rate": success_rate,
            "successes": success_rate * 20,
            "episodes": 20.0,
            "mean_return": success_rate * 5.0,
            "mean_length": 60.0,
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    return run_dir


def test_summarize_ppo_runs_preserves_seed_variance() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        runs = [
            _write_run(root, 1, 0.5),
            _write_run(root, 2, 0.75),
            _write_run(root, 3, 1.0),
        ]
        summary = summarize_ppo_runs(runs)
        assert summary["training_seeds"] == 3
        assert summary["success_rate_across_seeds"]["mean"] == pytest.approx(0.75)
        assert summary["success_rate_across_seeds"]["sample_std"] == pytest.approx(
            0.25
        )
        assert summary["pooled_episode_success"]["successes"] == 45
        assert summary["pooled_episode_success"]["episodes"] == 60


def test_summarize_ppo_runs_rejects_duplicate_seeds() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        first = _write_run(root, 1, 0.5)
        second = root / "duplicate"
        second.mkdir()
        (second / "summary.json").write_text(
            (first / "summary.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unique"):
            summarize_ppo_runs([first, second])


def test_summarize_ppo_runs_normalizes_legacy_proprio_flag() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        legacy = _write_run(root, 1, 0.5)
        explicit_false = _write_run(
            root,
            2,
            0.5,
            include_end_effector_position=False,
        )

        summary = summarize_ppo_runs([legacy, explicit_false])

        assert not summary["environment"]["include_end_effector_position_in_proprio"]


def test_summarize_ppo_runs_rejects_semantically_different_proprio() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        legacy = _write_run(root, 1, 0.5)
        cartesian_proprio = _write_run(
            root,
            2,
            0.5,
            include_end_effector_position=True,
        )

        with pytest.raises(ValueError, match="environments do not match"):
            summarize_ppo_runs([legacy, cartesian_proprio])
