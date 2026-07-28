from __future__ import annotations

import argparse
from pathlib import Path

from embodied_vla.data import collect_expert_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect successful expert episodes.")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/so_arm_pick_place"))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--unbalanced-tasks", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    collect_expert_dataset(
        args.output_dir,
        episodes=args.episodes,
        seed=args.seed,
        image_size=args.image_size,
        balanced_tasks=not args.unbalanced_tasks,
        domain_randomization=args.domain_randomization,
    )


if __name__ == "__main__":
    main()
