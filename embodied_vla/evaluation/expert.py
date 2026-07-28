from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from embodied_vla.data.trajectory import TASK_COLORS, TASK_SIDES
from embodied_vla.envs import SOArmEnvConfig, SOArmPickPlaceEnv
from embodied_vla.experts import PickPlaceExpert
from embodied_vla.metrics import wilson_score_interval
from embodied_vla.training.run_guard import claim_run_directory


@dataclass(frozen=True)
class ExpertBenchmarkConfig:
    episodes: int = 100
    seed: int = 10_000
    grasp_mode: Literal["contact", "contact_assisted"] = "contact_assisted"
    domain_randomization: bool = False
    max_episode_steps: int = 300

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")


def benchmark_expert(
    *,
    output_dir: Path,
    benchmark_config: ExpertBenchmarkConfig,
) -> dict[str, Any]:
    with claim_run_directory(output_dir):
        return _benchmark_expert_in_claimed_directory(
            output_dir=output_dir,
            benchmark_config=benchmark_config,
        )


def _benchmark_expert_in_claimed_directory(
    *,
    output_dir: Path,
    benchmark_config: ExpertBenchmarkConfig,
) -> dict[str, Any]:
    env_config = SOArmEnvConfig(
        observation_mode="state",
        task_level="pick_place",
        grasp_mode=benchmark_config.grasp_mode,
        max_episode_steps=benchmark_config.max_episode_steps,
        domain_randomization=benchmark_config.domain_randomization,
    )
    env = SOArmPickPlaceEnv(env_config)
    expert = PickPlaceExpert(env_config)
    records: list[dict[str, Any]] = []
    records_path = output_dir / "episodes.jsonl"
    start_time = time.perf_counter()
    try:
        for episode_index in range(benchmark_config.episodes):
            task_index = episode_index % (len(TASK_COLORS) * len(TASK_SIDES))
            color = TASK_COLORS[task_index // len(TASK_SIDES)]
            side = TASK_SIDES[task_index % len(TASK_SIDES)]
            episode_seed = benchmark_config.seed + episode_index
            _, info = env.reset(
                seed=episode_seed,
                options={"target_color": color, "goal_side": side},
            )
            expert.reset()
            episode_return = 0.0
            terminated = False
            truncated = False
            for _ in range(benchmark_config.max_episode_steps):
                action = expert.act(info)
                _, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            record = {
                "episode": episode_index,
                "seed": episode_seed,
                "target_color": color,
                "goal_side": side,
                "success": bool(info["success"]),
                "return": float(episode_return),
                "length": int(info["step"]),
                "termination_reason": info["termination_reason"],
                "final_expert_phase": expert.phase.name.lower(),
                "has_grasped": bool(info["has_grasped"]),
                "has_lifted": bool(info["has_lifted"]),
            }
            records.append(record)
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            print(
                f"episode={episode_index:03d} task={color}->{side} "
                f"success={record['success']} steps={record['length']} "
                f"phase={record['final_expert_phase']}"
            )
    finally:
        env.close()

    failures = [record for record in records if not record["success"]]
    failure_phase_histogram = {
        phase: sum(record["final_expert_phase"] == phase for record in failures)
        for phase in sorted({record["final_expert_phase"] for record in failures})
    }
    task_success = {}
    for color in TASK_COLORS:
        for side in TASK_SIDES:
            task_records = [
                record
                for record in records
                if record["target_color"] == color and record["goal_side"] == side
            ]
            task_success[f"{color}->{side}"] = {
                "episodes": len(task_records),
                "success_rate": (
                    float(np.mean([record["success"] for record in task_records]))
                    if task_records
                    else None
                ),
            }
    successes = sum(record["success"] for record in records)
    confidence_low, confidence_high = wilson_score_interval(
        successes,
        benchmark_config.episodes,
    )
    summary = {
        "episodes": benchmark_config.episodes,
        "successes": successes,
        "success_rate": successes / benchmark_config.episodes,
        "success_wilson_95": [confidence_low, confidence_high],
        "mean_return": float(np.mean([record["return"] for record in records])),
        "mean_length": float(np.mean([record["length"] for record in records])),
        "failure_phase_histogram": failure_phase_histogram,
        "task_success": task_success,
        "elapsed_seconds": time.perf_counter() - start_time,
        "benchmark": asdict(benchmark_config),
        "environment": asdict(env_config),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary
