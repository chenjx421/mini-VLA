from __future__ import annotations

import argparse
from pathlib import Path

from embodied_vla.evaluation import VLAEvalConfig, evaluate_tiny_vla


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Tiny-VLA in closed-loop MuJoCo.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vla_evaluation"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--execution-horizon", type=int, default=1)
    parser.add_argument("--video-episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument(
        "--grasp-mode",
        choices=("contact", "contact_assisted"),
        default="contact_assisted",
    )
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument(
        "--random-tasks",
        action="store_true",
        help="Sample tasks randomly instead of cycling all color/goal pairs.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--cartesian-action-gain",
        type=float,
        default=1.0,
        help="Multiply executed dx/dy/dz actions, then clip to [-1, 1].",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate_tiny_vla(
        args.checkpoint,
        output_dir=args.output_dir,
        eval_config=VLAEvalConfig(
            episodes=args.episodes,
            seed=args.seed,
            execution_horizon=args.execution_horizon,
            video_episodes=args.video_episodes,
            max_episode_steps=args.max_episode_steps,
            grasp_mode=args.grasp_mode,
            domain_randomization=args.domain_randomization,
            balanced_tasks=not args.random_tasks,
            torch_threads=args.torch_threads,
            cartesian_action_gain=args.cartesian_action_gain,
        ),
        device=args.device,
    )


if __name__ == "__main__":
    main()
