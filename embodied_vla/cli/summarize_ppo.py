from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from embodied_vla.metrics import wilson_score_interval


def summarize_ppo_runs(run_dirs: list[Path]) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("at least two PPO runs are required for aggregation")
    summaries = [
        json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
    ]
    seeds = [int(summary["seed"]) for summary in summaries]
    if len(set(seeds)) != len(seeds):
        raise ValueError("PPO run seeds must be unique")
    normalized_environments = [
        _normalize_environment(summary["environment"]) for summary in summaries
    ]
    reference_environment = normalized_environments[0]
    reference_ppo = summaries[0]["ppo"]
    if any(
        environment != reference_environment
        for environment in normalized_environments[1:]
    ):
        raise ValueError("PPO run environments do not match")
    if any(summary["ppo"] != reference_ppo for summary in summaries[1:]):
        raise ValueError("PPO run configurations do not match")

    evaluations = [summary["final_evaluation"] for summary in summaries]
    success_rates = np.asarray(
        [evaluation["success_rate"] for evaluation in evaluations],
        dtype=np.float64,
    )
    returns = np.asarray(
        [evaluation["mean_return"] for evaluation in evaluations],
        dtype=np.float64,
    )
    lengths = np.asarray(
        [evaluation["mean_length"] for evaluation in evaluations],
        dtype=np.float64,
    )
    episodes = [int(evaluation["episodes"]) for evaluation in evaluations]
    successes = [
        int(
            evaluation.get(
                "successes",
                round(evaluation["success_rate"] * evaluation["episodes"]),
            )
        )
        for evaluation in evaluations
    ]
    pooled_successes = sum(successes)
    pooled_episodes = sum(episodes)
    pooled_low, pooled_high = wilson_score_interval(
        pooled_successes,
        pooled_episodes,
    )
    return {
        "runs": [
            {
                "run": run_dir.name,
                "seed": seed,
                "successes": success_count,
                "episodes": episode_count,
                "success_rate": float(rate),
                "mean_return": float(mean_return),
                "mean_length": float(mean_length),
            }
            for (
                run_dir,
                seed,
                success_count,
                episode_count,
                rate,
                mean_return,
                mean_length,
            ) in zip(
                run_dirs,
                seeds,
                successes,
                episodes,
                success_rates,
                returns,
                lengths,
                strict=True,
            )
        ],
        "training_seeds": len(seeds),
        "success_rate_across_seeds": _distribution_summary(success_rates),
        "mean_return_across_seeds": _distribution_summary(returns),
        "mean_length_across_seeds": _distribution_summary(lengths),
        "pooled_episode_success": {
            "successes": pooled_successes,
            "episodes": pooled_episodes,
            "success_rate": pooled_successes / pooled_episodes,
            "wilson_95": [pooled_low, pooled_high],
            "note": "Conditional on the trained policies; seed-level std remains primary.",
        },
        "environment": reference_environment,
        "ppo": reference_ppo,
    }


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _normalize_environment(environment: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(environment)
    normalized.setdefault("include_end_effector_position_in_proprio", False)
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate final evaluations across PPO training seeds."
    )
    parser.add_argument("--ppo-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_ppo_runs(args.ppo_runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
