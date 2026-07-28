from __future__ import annotations

import argparse
from pathlib import Path

from embodied_vla.evaluation.counterfactual import (
    CounterfactualEvalConfig,
    evaluate_language_counterfactuals,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Tiny-VLA language counterfactuals on fixed scenes."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/language_counterfactuals"),
    )
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--visualized-scenes", type=int, default=2)
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate_language_counterfactuals(
        args.checkpoint,
        output_dir=args.output_dir,
        eval_config=CounterfactualEvalConfig(
            scenes=args.scenes,
            seed=args.seed,
            visualized_scenes=args.visualized_scenes,
            domain_randomization=args.domain_randomization,
            torch_threads=args.torch_threads,
        ),
        device=args.device,
    )


if __name__ == "__main__":
    main()
