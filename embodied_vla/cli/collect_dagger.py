from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_vla.data import DAggerCollectConfig, collect_dagger_corrections


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect DAgger expert corrections on Tiny-VLA rollout states."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--expert-mixing-probability", type=float, default=0.35)
    parser.add_argument("--random-tasks", action="store_true")
    parser.add_argument("--no-domain-randomization", action="store_true")
    parser.add_argument("--video-episodes", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = collect_dagger_corrections(
        args.checkpoint,
        output_dir=args.output_dir,
        config=DAggerCollectConfig(
            episodes=args.episodes,
            seed=args.seed,
            max_steps=args.max_steps,
            expert_mixing_probability=args.expert_mixing_probability,
            balanced_tasks=not args.random_tasks,
            domain_randomization=not args.no_domain_randomization,
            video_episodes=args.video_episodes,
            torch_threads=args.torch_threads,
        ),
        device=args.device,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
