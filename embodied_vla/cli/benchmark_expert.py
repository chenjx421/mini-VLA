from __future__ import annotations

import argparse
from pathlib import Path

from embodied_vla.evaluation.expert import ExpertBenchmarkConfig, benchmark_expert


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the physical waypoint expert without rendering."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument(
        "--grasp-mode",
        choices=("contact", "contact_assisted"),
        default="contact_assisted",
    )
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=300)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    benchmark_expert(
        output_dir=args.output_dir,
        benchmark_config=ExpertBenchmarkConfig(
            episodes=args.episodes,
            seed=args.seed,
            grasp_mode=args.grasp_mode,
            domain_randomization=args.domain_randomization,
            max_episode_steps=args.max_episode_steps,
        ),
    )


if __name__ == "__main__":
    main()
