from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from embodied_vla.evaluation.expert import ExpertBenchmarkConfig, benchmark_expert


def test_expert_benchmark_writes_episode_evidence() -> None:
    with TemporaryDirectory() as temporary_directory:
        output_dir = Path(temporary_directory) / "expert"
        summary = benchmark_expert(
            output_dir=output_dir,
            benchmark_config=ExpertBenchmarkConfig(
                episodes=1,
                seed=17,
                max_episode_steps=300,
            ),
        )
        assert summary["episodes"] == 1
        assert summary["successes"] in {0, 1}
        assert 0.0 <= summary["success_rate"] <= 1.0
        empty_tasks = [
            task
            for task in summary["task_success"].values()
            if task["episodes"] == 0
        ]
        assert empty_tasks
        assert all(task["success_rate"] is None for task in empty_tasks)
        assert (output_dir / "episodes.jsonl").exists()
        assert (output_dir / "summary.json").exists()
        assert not (output_dir / ".run.lock").exists()
