from __future__ import annotations

import argparse
from pathlib import Path

from embodied_vla.grounding_calibration import (
    GroundingCalibrationFitConfig,
    fit_grounding_calibration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit leakage-safe affine calibration for Tiny-VLA grounding."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fit_grounding_calibration(
        args.checkpoint,
        args.dataset,
        output_dir=args.output_dir,
        config=GroundingCalibrationFitConfig(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            torch_threads=args.torch_threads,
            ridge=args.ridge,
        ),
        device=args.device,
    )


if __name__ == "__main__":
    main()
