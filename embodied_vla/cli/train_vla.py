from __future__ import annotations

import argparse
from pathlib import Path

from embodied_vla.models import TinyVLAConfig
from embodied_vla.training.vla_trainer import TinyVLATrainConfig, train_tiny_vla


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the from-scratch Tiny-VLA.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tiny_vla"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--include-end-effector-position",
        action="store_true",
        help="Append normalized end-effector XYZ to the 12D joint proprioception.",
    )
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--phase-weight", type=float, default=0.25)
    parser.add_argument("--grounding-coordinate-weight", type=float, default=0.5)
    parser.add_argument("--grounding-heatmap-weight", type=float, default=0.25)
    parser.add_argument("--grounding-world-weight", type=float, default=0.5)
    parser.add_argument("--early-window-steps", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--initial-state-weight", type=float, default=1.0)
    parser.add_argument("--early-state-weight", type=float, default=1.0)
    parser.add_argument("--correction-sample-weight", type=float, default=1.0)
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        help="Use weighted replacement sampling with this many samples per epoch.",
    )
    parser.add_argument(
        "--action-head",
        choices=("deterministic", "flow_matching"),
        default="deterministic",
    )
    parser.add_argument("--flow-matching-steps", type=int, default=8)
    parser.add_argument(
        "--grounding-action-conditioning",
        action="store_true",
        help="Feed predicted target/goal coordinates and proprio into action queries.",
    )
    parser.add_argument(
        "--grounding-coordinate-refinement",
        action="store_true",
        help="Refine attention soft-argmax coordinates within one visual patch.",
    )
    parser.add_argument(
        "--high-resolution-grounding",
        action="store_true",
        help="Refine language grounding on a 4x-downsampled convolutional grid.",
    )
    parser.add_argument(
        "--freeze-backbone-for-high-resolution-grounding",
        action="store_true",
        help="Train only the high-resolution grounding branch and its blend gate.",
    )
    parser.add_argument(
        "--world-grounding",
        action="store_true",
        help="Predict target and goal positions in normalized world coordinates.",
    )
    parser.add_argument(
        "--world-grounding-action-conditioning",
        action="store_true",
        help="Feed predicted 3D target/goal positions into the action queries.",
    )
    parser.add_argument(
        "--phase-action-conditioning",
        action="store_true",
        help="Feed predicted phase probabilities into the action queries.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Resume an interrupted run in the same --output-dir.",
    )
    parser.add_argument(
        "--initialize-checkpoint",
        type=Path,
        help="Initialize model weights for a new fine-tuning run.",
    )
    parser.add_argument(
        "--correction-dataset",
        type=Path,
        action="append",
        default=[],
        help="DAgger correction dataset; may be supplied more than once.",
    )
    parser.add_argument(
        "--correction-repeat",
        type=int,
        default=1,
        help="Repeat each correction dataset in the mixed training set.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_config = TinyVLAConfig(
        image_size=args.image_size,
        proprio_dim=15 if args.include_end_effector_position else 12,
        action_horizon=args.action_horizon,
        action_head=args.action_head,
        flow_matching_steps=args.flow_matching_steps,
        grounding_action_conditioning=args.grounding_action_conditioning,
        grounding_coordinate_refinement=args.grounding_coordinate_refinement,
        high_resolution_grounding=args.high_resolution_grounding,
        world_grounding=args.world_grounding,
        world_grounding_action_conditioning=(
            args.world_grounding_action_conditioning
        ),
        phase_action_conditioning=args.phase_action_conditioning,
    )
    train_config = TinyVLATrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        action_weight=args.action_weight,
        phase_weight=args.phase_weight,
        grounding_coordinate_weight=args.grounding_coordinate_weight,
        grounding_heatmap_weight=args.grounding_heatmap_weight,
        grounding_world_weight=args.grounding_world_weight,
        early_window_steps=args.early_window_steps,
        torch_threads=args.torch_threads,
        initial_state_weight=args.initial_state_weight,
        early_state_weight=args.early_state_weight,
        correction_sample_weight=args.correction_sample_weight,
        samples_per_epoch=args.samples_per_epoch,
        freeze_backbone_for_high_resolution_grounding=(
            args.freeze_backbone_for_high_resolution_grounding
        ),
    )
    train_tiny_vla(
        args.dataset,
        output_dir=args.output_dir,
        model_config=model_config,
        train_config=train_config,
        seed=args.seed,
        device=args.device,
        resume_checkpoint=args.resume_checkpoint,
        initialize_checkpoint=args.initialize_checkpoint,
        correction_dataset_roots=tuple(args.correction_dataset),
        correction_repeat=args.correction_repeat,
    )


if __name__ == "__main__":
    main()
