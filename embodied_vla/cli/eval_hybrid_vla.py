from __future__ import annotations

import argparse
from pathlib import Path

from embodied_vla.evaluation.hybrid_vla import (
    HybridVLAEvalConfig,
    evaluate_hybrid_vla,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate learned VLA grounding with calibrated visual servo."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=40_000)
    parser.add_argument("--video-episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--random-tasks", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--smoothing-alpha", type=float, default=0.35)
    parser.add_argument("--recovery-search-radius-m", type=float, default=0.0)
    parser.add_argument("--close-retry-steps", type=int, default=35)
    parser.add_argument("--grounding-calibration", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate_hybrid_vla(
        args.checkpoint,
        output_dir=args.output_dir,
        eval_config=HybridVLAEvalConfig(
            episodes=args.episodes,
            seed=args.seed,
            video_episodes=args.video_episodes,
            max_episode_steps=args.max_episode_steps,
            domain_randomization=args.domain_randomization,
            balanced_tasks=not args.random_tasks,
            torch_threads=args.torch_threads,
            smoothing_alpha=args.smoothing_alpha,
            recovery_search_radius_m=args.recovery_search_radius_m,
            close_retry_steps=args.close_retry_steps,
        ),
        device=args.device,
        grounding_calibration_path=args.grounding_calibration,
    )


if __name__ == "__main__":
    main()
