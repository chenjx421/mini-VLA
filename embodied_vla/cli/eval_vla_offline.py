from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_vla.evaluation import OfflineVLAEvalConfig, evaluate_tiny_vla_offline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose Tiny-VLA action and phase errors on an offline split."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vla_offline_evaluation"),
    )
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate_tiny_vla_offline(
        args.checkpoint,
        args.dataset,
        output_dir=args.output_dir,
        eval_config=OfflineVLAEvalConfig(
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            progress_bins=args.progress_bins,
            seed=args.seed,
            torch_threads=args.torch_threads,
        ),
        device=args.device,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
