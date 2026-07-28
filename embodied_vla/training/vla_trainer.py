from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from embodied_vla.data import (
    ActionChunkDataset,
    DAggerCorrectionDataset,
    audit_expert_dataset,
)
from embodied_vla.models import TinyVLA, TinyVLAConfig, TinyVLAOutput
from embodied_vla.proprioception import (
    CARTESIAN_PROPRIO_DIM,
    JOINT_PROPRIO_DIM,
)
from embodied_vla.reproducibility import runtime_metadata
from embodied_vla.training.run_guard import claim_run_directory


@dataclass(frozen=True)
class TinyVLATrainConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    action_weight: float = 1.0
    phase_weight: float = 0.25
    grounding_coordinate_weight: float = 0.5
    grounding_heatmap_weight: float = 0.25
    grounding_world_weight: float = 0.5
    num_workers: int = 0
    early_window_steps: int = 10
    torch_threads: int = 1
    initial_state_weight: float = 1.0
    early_state_weight: float = 1.0
    correction_sample_weight: float = 1.0
    samples_per_epoch: int | None = None
    freeze_backbone_for_high_resolution_grounding: bool = False

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.early_window_steps <= 0:
            raise ValueError("early_window_steps must be positive")
        if self.torch_threads <= 0:
            raise ValueError("torch_threads must be positive")
        if self.initial_state_weight <= 0:
            raise ValueError("initial_state_weight must be positive")
        if self.early_state_weight <= 0:
            raise ValueError("early_state_weight must be positive")
        if self.correction_sample_weight <= 0:
            raise ValueError("correction_sample_weight must be positive")
        if self.samples_per_epoch is not None and self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive or None")
        if self.grounding_world_weight < 0:
            raise ValueError("grounding_world_weight cannot be negative")


def tiny_vla_loss(
    output: TinyVLAOutput,
    batch: dict[str, Tensor],
    *,
    model_config: TinyVLAConfig,
    train_config: TinyVLATrainConfig,
    phase_class_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    action_mask = batch["action_mask"].float()
    if model_config.action_head == "flow_matching":
        if output.flow_velocity is None or output.flow_target is None:
            raise ValueError("flow-matching training requires velocity and target tensors")
        action_error = functional.mse_loss(
            output.flow_velocity,
            output.flow_target,
            reduction="none",
        ).mean(dim=-1)
    else:
        action_error = functional.smooth_l1_loss(
            output.action_chunk,
            batch["action_chunk"],
            reduction="none",
        ).mean(dim=-1)
    valid_actions_per_sample = action_mask.sum(dim=1).clamp_min(1.0)
    action_loss = (
        (action_error * action_mask).sum(dim=1) / valid_actions_per_sample
    ).mean()
    phase_loss = functional.cross_entropy(
        output.phase_logits,
        batch["phase"],
        weight=phase_class_weights,
    )

    ground_truth_coordinates = torch.stack(
        (batch["target_pixel"], batch["goal_pixel"]),
        dim=1,
    )
    pixel_valid = batch["pixel_valid"].float()
    coordinate_error = functional.smooth_l1_loss(
        output.grounding_coordinates,
        ground_truth_coordinates,
        reduction="none",
    ).mean(dim=-1)
    coordinate_loss = (coordinate_error * pixel_valid).sum() / pixel_valid.sum().clamp_min(
        1.0
    )

    heatmap_height, heatmap_width = output.grounding_heatmaps.shape[-2:]
    if heatmap_height != heatmap_width:
        raise ValueError("grounding heatmap must use a square spatial grid")
    grid_size = heatmap_height
    patch_x = (ground_truth_coordinates[..., 0] * grid_size).long().clamp(
        0,
        grid_size - 1,
    )
    patch_y = (ground_truth_coordinates[..., 1] * grid_size).long().clamp(
        0,
        grid_size - 1,
    )
    patch_index = patch_y * grid_size + patch_x
    heatmap_probabilities = output.grounding_heatmaps.flatten(2).clamp_min(1e-8)
    selected_probabilities = heatmap_probabilities.gather(
        2,
        patch_index.unsqueeze(-1),
    ).squeeze(-1)
    heatmap_error = -selected_probabilities.log()
    heatmap_loss = (heatmap_error * pixel_valid).sum() / pixel_valid.sum().clamp_min(1.0)

    world_grounding_loss = output.action_chunk.new_zeros(())
    if output.grounding_world_positions is not None:
        ground_truth_world = torch.stack(
            (batch["target_world"], batch["goal_world"]),
            dim=1,
        )
        world_error = functional.smooth_l1_loss(
            output.grounding_world_positions,
            ground_truth_world,
            reduction="none",
        ).mean(dim=-1)
        world_grounding_loss = (
            (world_error * pixel_valid).sum()
            / pixel_valid.sum().clamp_min(1.0)
        )
    elif model_config.world_grounding:
        raise ValueError("world-grounding model did not return world predictions")

    total = (
        train_config.action_weight * action_loss
        + train_config.phase_weight * phase_loss
        + train_config.grounding_coordinate_weight * coordinate_loss
        + train_config.grounding_heatmap_weight * heatmap_loss
        + train_config.grounding_world_weight * world_grounding_loss
    )
    losses = {
        "total": total,
        "action": action_loss,
        "phase": phase_loss,
        "grounding_coordinate": coordinate_loss,
        "grounding_heatmap": heatmap_loss,
        "grounding_world": world_grounding_loss,
    }
    return total, losses


def train_tiny_vla(
    dataset_root: Path,
    *,
    output_dir: Path,
    model_config: TinyVLAConfig,
    train_config: TinyVLATrainConfig,
    seed: int,
    device: str = "cpu",
    resume_checkpoint: Path | None = None,
    initialize_checkpoint: Path | None = None,
    correction_dataset_roots: tuple[Path, ...] = (),
    correction_repeat: int = 1,
) -> dict[str, Any]:
    if resume_checkpoint is not None and initialize_checkpoint is not None:
        raise ValueError("resume_checkpoint and initialize_checkpoint are mutually exclusive")
    if correction_repeat <= 0:
        raise ValueError("correction_repeat must be positive")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(train_config.torch_threads)
    try:
        with claim_run_directory(output_dir, resume=resume_checkpoint is not None):
            return _train_tiny_vla_in_claimed_directory(
                dataset_root,
                output_dir=output_dir,
                model_config=model_config,
                train_config=train_config,
                seed=seed,
                device=device,
                resume_checkpoint=resume_checkpoint,
                initialize_checkpoint=initialize_checkpoint,
                correction_dataset_roots=correction_dataset_roots,
                correction_repeat=correction_repeat,
            )
    finally:
        torch.set_num_threads(previous_threads)


def _train_tiny_vla_in_claimed_directory(
    dataset_root: Path,
    *,
    output_dir: Path,
    model_config: TinyVLAConfig,
    train_config: TinyVLATrainConfig,
    seed: int,
    device: str,
    resume_checkpoint: Path | None,
    initialize_checkpoint: Path | None,
    correction_dataset_roots: tuple[Path, ...],
    correction_repeat: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch_device = torch.device(device)

    base_train_dataset = ActionChunkDataset(
        dataset_root,
        action_horizon=model_config.action_horizon,
        split="train",
        proprio_dim=model_config.proprio_dim,
    )
    validation_dataset = ActionChunkDataset(
        dataset_root,
        action_horizon=model_config.action_horizon,
        split="validation",
        proprio_dim=model_config.proprio_dim,
    )
    statistics_path = dataset_root / "statistics.json"
    dataset_statistics = (
        json.loads(statistics_path.read_text(encoding="utf-8"))
        if statistics_path.exists()
        else audit_expert_dataset(dataset_root)
    )
    dataset_metadata = {
        "name": dataset_root.name,
        "fingerprint_sha256": dataset_statistics["dataset_fingerprint_sha256"],
        "manifest_sha256": dataset_statistics["manifest_sha256"],
        "episodes": dataset_statistics["episodes"],
        "total_steps": dataset_statistics["total_steps"],
        "task_counts": dataset_statistics["task_counts"],
        "split_strategy": base_train_dataset.split_strategy,
        "train_episodes": len(base_train_dataset.records),
        "validation_episodes": len(validation_dataset.records),
    }
    correction_datasets = [
        DAggerCorrectionDataset(
            root,
            action_horizon=model_config.action_horizon,
            proprio_dim=model_config.proprio_dim,
        )
        for root in correction_dataset_roots
    ]
    dataset_metadata["correction_repeat"] = correction_repeat
    dataset_metadata["corrections"] = [
        {
            "path": str(dataset.root.resolve()),
            "episodes": len(dataset.records),
            "samples": len(dataset),
            "dataset_fingerprint_sha256": dataset.metadata[
                "dataset_fingerprint_sha256"
            ],
        }
        for dataset in correction_datasets
    ]
    if correction_datasets:
        train_components = [base_train_dataset] + [
            dataset
            for _ in range(correction_repeat)
            for dataset in correction_datasets
        ]
        train_dataset = ConcatDataset(
            train_components
        )
    else:
        train_components = [base_train_dataset]
        train_dataset = base_train_dataset
    phase_counts = torch.as_tensor(
        dataset_statistics["phase_counts"][: model_config.phase_count],
        dtype=torch.float32,
        device=torch_device,
    ).clamp_min(1.0)
    for correction_dataset in correction_datasets:
        correction_phase_counts = torch.as_tensor(
            correction_dataset.metadata["phase_counts"][: model_config.phase_count],
            dtype=torch.float32,
            device=torch_device,
        )
        phase_counts += correction_repeat * correction_phase_counts
    phase_class_weights = torch.sqrt(phase_counts.sum() / phase_counts)
    phase_class_weights = phase_class_weights / phase_class_weights.mean()
    dataset_metadata["phase_class_weights"] = phase_class_weights.cpu().tolist()
    sampling_weights = _build_training_sampling_weights(
        train_components,
        train_config=train_config,
    )
    use_weighted_sampler = bool(
        train_config.samples_per_epoch is not None
        or not torch.allclose(
            sampling_weights,
            torch.ones_like(sampling_weights),
        )
    )
    dataset_metadata["sampling"] = {
        "strategy": "weighted_replacement" if use_weighted_sampler else "shuffle",
        "initial_state_weight": train_config.initial_state_weight,
        "early_state_weight": train_config.early_state_weight,
        "early_window_steps": train_config.early_window_steps,
        "correction_sample_weight": train_config.correction_sample_weight,
        "samples_per_epoch": (
            train_config.samples_per_epoch
            if train_config.samples_per_epoch is not None
            else len(train_dataset)
        ),
    }
    generator = torch.Generator().manual_seed(seed)
    sampler = (
        WeightedRandomSampler(
            sampling_weights,
            num_samples=(
                train_config.samples_per_epoch
                if train_config.samples_per_epoch is not None
                else len(train_dataset)
            ),
            replacement=True,
            generator=generator,
        )
        if use_weighted_sampler
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=train_config.num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )
    model = TinyVLA(model_config).to(torch_device)
    initialization_metadata = None
    if initialize_checkpoint is not None:
        initialization = torch.load(
            initialize_checkpoint,
            map_location=torch_device,
            weights_only=True,
        )
        initialization_metadata = _initialize_vla_weights(
            model,
            initialization,
            model_config=model_config,
            checkpoint_path=initialize_checkpoint,
        )
    if train_config.freeze_backbone_for_high_resolution_grounding:
        _freeze_backbone_for_high_resolution_grounding(model)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config.epochs,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    best_validation_loss = float("inf")
    best_validation_action_mae = float("inf")
    best_validation_early_action_mae = float("inf")
    best_validation_initial_action_mae = float("inf")
    best_validation_grounding_l2 = float("inf")
    metrics_path = output_dir / "metrics.jsonl"
    start_epoch = 1
    initial_validation_metrics = None
    if resume_checkpoint is not None:
        checkpoint = torch.load(
            resume_checkpoint,
            map_location=torch_device,
            weights_only=False,
        )
        _validate_resume_checkpoint(
            checkpoint,
            model_config=model_config,
            train_config=train_config,
            dataset_metadata=dataset_metadata,
            seed=seed,
            metrics_path=metrics_path,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        completed_epoch = int(checkpoint["epoch"])
        start_epoch = completed_epoch + 1
        best_validation_loss = float(checkpoint["validation"]["total"])
        best_validation_action_mae = float(
            checkpoint["validation"]["action_mae"]
        )
        best_validation_early_action_mae = float(
            checkpoint["validation"].get(
                "early_action_mae",
                checkpoint["validation"]["action_mae"],
            )
        )
        best_validation_initial_action_mae = float(
            checkpoint["validation"].get(
                "initial_action_mae",
                checkpoint["validation"]["action_mae"],
            )
        )
        best_validation_grounding_l2 = float(
            checkpoint["validation"]["grounding_l2"]
        )
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            scheduler.last_epoch = completed_epoch
            scheduler._last_lr = [
                float(group["lr"]) for group in optimizer.param_groups
            ]
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if "data_generator_state" in checkpoint:
            generator.set_state(checkpoint["data_generator_state"].cpu())
        if not (checkpoint_dir / "best_action.pt").exists():
            action_checkpoint = dict(checkpoint)
            action_checkpoint["selection_metric"] = "validation_action_mae"
            torch.save(action_checkpoint, checkpoint_dir / "best_action.pt")
    elif initialize_checkpoint is not None:
        initial_validation_metrics = _run_epoch(
            model,
            validation_loader,
            model_config=model_config,
            train_config=train_config,
            device=torch_device,
            optimizer=None,
            phase_class_weights=phase_class_weights,
        )
        best_validation_loss = initial_validation_metrics["total"]
        best_validation_action_mae = initial_validation_metrics["action_mae"]
        best_validation_early_action_mae = initial_validation_metrics[
            "early_action_mae"
        ]
        best_validation_initial_action_mae = initial_validation_metrics[
            "initial_action_mae"
        ]
        best_validation_grounding_l2 = initial_validation_metrics[
            "grounding_l2"
        ]
        for filename, selection_metric in (
            ("initial.pt", "initialization_validation"),
            ("best.pt", "validation_total"),
            ("best_action.pt", "validation_action_mae"),
            ("best_early_action.pt", "validation_early_action_mae"),
            ("best_initial_action.pt", "validation_initial_action_mae"),
            ("best_grounding.pt", "validation_grounding_l2"),
        ):
            _save_vla_checkpoint(
                checkpoint_dir / filename,
                model,
                optimizer,
                model_config,
                train_config,
                seed,
                0,
                initial_validation_metrics,
                dataset_metadata,
                scheduler,
                generator,
                selection_metric=selection_metric,
            )
        (output_dir / "initial_validation.json").write_text(
            json.dumps(initial_validation_metrics, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(
            "epoch=000 initialization "
            f"validation={initial_validation_metrics['total']:.4f} "
            f"action_mae={initial_validation_metrics['action_mae']:.4f} "
            f"early_mae={initial_validation_metrics['early_action_mae']:.4f}"
        )

    start_time = time.perf_counter()

    for epoch in range(start_epoch, train_config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            model_config=model_config,
            train_config=train_config,
            device=torch_device,
            optimizer=optimizer,
            phase_class_weights=phase_class_weights,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            model_config=model_config,
            train_config=train_config,
            device=torch_device,
            optimizer=None,
            phase_class_weights=phase_class_weights,
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        if validation_metrics["total"] < best_validation_loss:
            best_validation_loss = validation_metrics["total"]
            _save_vla_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                model_config,
                train_config,
                seed,
                epoch,
                validation_metrics,
                dataset_metadata,
                scheduler,
                generator,
                selection_metric="validation_total",
            )
        if validation_metrics["action_mae"] < best_validation_action_mae:
            best_validation_action_mae = validation_metrics["action_mae"]
            _save_vla_checkpoint(
                checkpoint_dir / "best_action.pt",
                model,
                optimizer,
                model_config,
                train_config,
                seed,
                epoch,
                validation_metrics,
                dataset_metadata,
                scheduler,
                generator,
                selection_metric="validation_action_mae",
            )
        if validation_metrics["early_action_mae"] < best_validation_early_action_mae:
            best_validation_early_action_mae = validation_metrics["early_action_mae"]
            _save_vla_checkpoint(
                checkpoint_dir / "best_early_action.pt",
                model,
                optimizer,
                model_config,
                train_config,
                seed,
                epoch,
                validation_metrics,
                dataset_metadata,
                scheduler,
                generator,
                selection_metric="validation_early_action_mae",
            )
        if (
            validation_metrics["initial_action_mae"]
            < best_validation_initial_action_mae
        ):
            best_validation_initial_action_mae = validation_metrics[
                "initial_action_mae"
            ]
            _save_vla_checkpoint(
                checkpoint_dir / "best_initial_action.pt",
                model,
                optimizer,
                model_config,
                train_config,
                seed,
                epoch,
                validation_metrics,
                dataset_metadata,
                scheduler,
                generator,
                selection_metric="validation_initial_action_mae",
            )
        if validation_metrics["grounding_l2"] < best_validation_grounding_l2:
            best_validation_grounding_l2 = validation_metrics["grounding_l2"]
            _save_vla_checkpoint(
                checkpoint_dir / "best_grounding.pt",
                model,
                optimizer,
                model_config,
                train_config,
                seed,
                epoch,
                validation_metrics,
                dataset_metadata,
                scheduler,
                generator,
                selection_metric="validation_grounding_l2",
            )
        _save_vla_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            model_config,
            train_config,
            seed,
            epoch,
            validation_metrics,
            dataset_metadata,
            scheduler,
            generator,
            selection_metric="last_epoch",
        )
        print(
            f"epoch={epoch:03d} train={train_metrics['total']:.4f} "
            f"validation={validation_metrics['total']:.4f} "
            f"action_mae={validation_metrics['action_mae']:.4f} "
            f"early_mae={validation_metrics['early_action_mae']:.4f} "
            f"phase_acc={validation_metrics['phase_accuracy']:.1%} "
            f"grounding_l2={validation_metrics['grounding_l2']:.4f}"
        )

    summary = {
        "seed": seed,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "best_validation_loss": best_validation_loss,
        "best_validation_action_mae": best_validation_action_mae,
        "best_validation_early_action_mae": best_validation_early_action_mae,
        "best_validation_initial_action_mae": best_validation_initial_action_mae,
        "best_validation_grounding_l2": best_validation_grounding_l2,
        "elapsed_seconds": time.perf_counter() - start_time,
        "train_samples": len(train_dataset),
        "train_samples_per_epoch": len(train_loader.sampler),
        "validation_samples": len(validation_dataset),
        "dataset": dataset_metadata,
        "model": asdict(model_config),
        "training": asdict(train_config),
        "resumed_from": (
            str(resume_checkpoint.resolve()) if resume_checkpoint is not None else None
        ),
        "initialization": initialization_metadata,
        "initial_validation": initial_validation_metrics,
        "start_epoch": start_epoch,
        "runtime": runtime_metadata(device=torch_device),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def _run_epoch(
    model: TinyVLA,
    loader: DataLoader[dict[str, Tensor]],
    *,
    model_config: TinyVLAConfig,
    train_config: TinyVLATrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    phase_class_weights: Tensor,
) -> dict[str, float]:
    if (
        optimizer is not None
        and train_config.freeze_backbone_for_high_resolution_grounding
    ):
        model.eval()
        for module in (
            model.high_resolution_grounding_stem,
            model.high_resolution_grounding_key,
            model.high_resolution_grounding_query,
        ):
            if module is not None:
                module.train()
    else:
        model.train(optimizer is not None)
    totals = {
        "total": 0.0,
        "action": 0.0,
        "phase": 0.0,
        "grounding_coordinate": 0.0,
        "grounding_heatmap": 0.0,
        "grounding_world": 0.0,
        "action_mae": 0.0,
        "action_mae_dx": 0.0,
        "action_mae_dy": 0.0,
        "action_mae_dz": 0.0,
        "action_mae_wrist": 0.0,
        "action_mae_jaw": 0.0,
        "phase_accuracy": 0.0,
        "grounding_l2": 0.0,
        "grounding_world_l2_m": 0.0,
    }
    sample_count = 0
    early_action_error_sum = 0.0
    early_action_sample_count = 0
    initial_action_error_sum = 0.0
    initial_action_sample_count = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(optimizer is not None):
            output = model(
                batch["rgb"],
                batch["proprio"],
                batch["language"],
                batch["language_mask"],
                action_targets=(
                    batch["action_chunk"]
                    if model_config.action_head == "flow_matching"
                    else None
                ),
            )
            loss, losses = tiny_vla_loss(
                output,
                batch,
                model_config=model_config,
                train_config=train_config,
                phase_class_weights=phase_class_weights,
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        batch_size = batch["rgb"].shape[0]
        sample_count += batch_size
        for name, value in losses.items():
            totals[name] += float(value.detach()) * batch_size
        action_mask = batch["action_mask"].unsqueeze(-1)
        action_mae = (
            (output.action_chunk - batch["action_chunk"]).abs() * action_mask
        ).sum() / (
            action_mask.sum().clamp_min(1) * output.action_chunk.shape[-1]
        )
        action_mae_by_dimension = (
            (output.action_chunk - batch["action_chunk"]).abs() * action_mask
        ).sum(dim=(0, 1)) / action_mask.sum().clamp_min(1)
        phase_accuracy = (
            output.phase_logits.argmax(dim=-1) == batch["phase"]
        ).float().mean()
        ground_truth = torch.stack(
            (batch["target_pixel"], batch["goal_pixel"]),
            dim=1,
        )
        first_action_error = (
            output.action_chunk[:, 0] - batch["action_chunk"][:, 0]
        ).abs().mean(dim=-1)
        early_mask = batch["time_index"] < train_config.early_window_steps
        initial_mask = batch["time_index"] == 0
        early_action_error_sum += float(
            first_action_error[early_mask].sum().detach()
        )
        early_action_sample_count += int(early_mask.sum())
        initial_action_error_sum += float(
            first_action_error[initial_mask].sum().detach()
        )
        initial_action_sample_count += int(initial_mask.sum())
        grounding_l2 = torch.linalg.vector_norm(
            output.grounding_coordinates - ground_truth,
            dim=-1,
        )
        grounding_l2 = (
            grounding_l2 * batch["pixel_valid"].float()
        ).sum() / batch["pixel_valid"].float().sum().clamp_min(1)
        world_grounding_l2_m = output.action_chunk.new_zeros(())
        if output.grounding_world_positions is not None:
            ground_truth_world = torch.stack(
                (batch["target_world"], batch["goal_world"]),
                dim=1,
            )
            world_grounding_l2_m = (
                torch.linalg.vector_norm(
                    output.grounding_world_positions - ground_truth_world,
                    dim=-1,
                )
                * batch["pixel_valid"].float()
            ).sum() / batch["pixel_valid"].float().sum().clamp_min(1)
            world_grounding_l2_m = world_grounding_l2_m * 0.5
        totals["action_mae"] += float(action_mae.detach()) * batch_size
        for name, value in zip(
            (
                "action_mae_dx",
                "action_mae_dy",
                "action_mae_dz",
                "action_mae_wrist",
                "action_mae_jaw",
            ),
            action_mae_by_dimension,
            strict=True,
        ):
            totals[name] += float(value.detach()) * batch_size
        totals["phase_accuracy"] += float(phase_accuracy.detach()) * batch_size
        totals["grounding_l2"] += float(grounding_l2.detach()) * batch_size
        totals["grounding_world_l2_m"] += (
            float(world_grounding_l2_m.detach()) * batch_size
        )
    metrics = {
        name: value / max(1, sample_count)
        for name, value in totals.items()
    }
    metrics["early_action_mae"] = early_action_error_sum / max(
        1,
        early_action_sample_count,
    )
    metrics["initial_action_mae"] = initial_action_error_sum / max(
        1,
        initial_action_sample_count,
    )
    return metrics


def _save_vla_checkpoint(
    path: Path,
    model: TinyVLA,
    optimizer: torch.optim.Optimizer,
    model_config: TinyVLAConfig,
    train_config: TinyVLATrainConfig,
    seed: int,
    epoch: int,
    validation_metrics: dict[str, float],
    dataset_metadata: dict[str, Any],
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    selection_metric: str,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "seed": seed,
            "epoch": epoch,
            "validation": validation_metrics,
            "dataset": dataset_metadata,
            "scheduler": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "data_generator_state": generator.get_state(),
            "selection_metric": selection_metric,
        },
        path,
    )


def _validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    *,
    model_config: TinyVLAConfig,
    train_config: TinyVLATrainConfig,
    dataset_metadata: dict[str, Any],
    seed: int,
    metrics_path: Path,
) -> None:
    if checkpoint["model_config"] != asdict(model_config):
        raise ValueError("resume checkpoint model configuration does not match")
    previous_training = dict(checkpoint["train_config"])
    current_training = asdict(train_config)
    previous_training.setdefault("grounding_world_weight", 0.5)
    previous_training.setdefault(
        "freeze_backbone_for_high_resolution_grounding",
        False,
    )
    previous_training.pop("epochs", None)
    current_training.pop("epochs", None)
    if previous_training != current_training:
        raise ValueError("resume checkpoint training configuration does not match")
    if int(checkpoint["seed"]) != seed:
        raise ValueError("resume checkpoint seed does not match")
    if (
        checkpoint["dataset"]["fingerprint_sha256"]
        != dataset_metadata["fingerprint_sha256"]
    ):
        raise ValueError("resume checkpoint dataset fingerprint does not match")
    if checkpoint["dataset"].get("corrections", []) != dataset_metadata.get(
        "corrections",
        [],
    ):
        raise ValueError("resume checkpoint correction datasets do not match")
    completed_epoch = int(checkpoint["epoch"])
    if train_config.epochs <= completed_epoch:
        raise ValueError(
            f"requested epochs ({train_config.epochs}) must exceed checkpoint "
            f"epoch ({completed_epoch})"
        )
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"resume metrics file is missing from output directory: {metrics_path}"
        )
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or int(records[-1]["epoch"]) != completed_epoch:
        raise ValueError(
            "metrics history does not end at the resume checkpoint epoch"
        )


def _initialize_vla_weights(
    model: TinyVLA,
    checkpoint: dict[str, Any],
    *,
    model_config: TinyVLAConfig,
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint_config = TinyVLAConfig(**checkpoint["model_config"])
    previous = asdict(checkpoint_config)
    current = asdict(model_config)
    changed_fields = {
        key: {"from": previous[key], "to": current[key]}
        for key in current
        if previous[key] != current[key]
    }
    boolean_upgrade_fields = {
        "grounding_action_conditioning",
        "grounding_coordinate_refinement",
        "high_resolution_grounding",
        "world_grounding",
        "world_grounding_action_conditioning",
        "phase_action_conditioning",
    }
    proprio_upgrade = changed_fields.get("proprio_dim") == {
        "from": JOINT_PROPRIO_DIM,
        "to": CARTESIAN_PROPRIO_DIM,
    }
    allowed_upgrade = bool(changed_fields) and all(
        (
            key in boolean_upgrade_fields
            and change == {"from": False, "to": True}
        )
        or (
            key == "proprio_dim"
            and change
            == {
                "from": JOINT_PROPRIO_DIM,
                "to": CARTESIAN_PROPRIO_DIM,
            }
        )
        for key, change in changed_fields.items()
    )
    if changed_fields and not allowed_upgrade:
        raise ValueError(
            "initialization checkpoint model configuration does not match: "
            f"{changed_fields}"
        )
    initialization_state = dict(checkpoint["model"])
    adapted_parameters: list[str] = []
    if proprio_upgrade:
        current_state = model.state_dict()
        proprio_key = "proprio_projection.0.weight"
        previous_proprio_weight = initialization_state[proprio_key]
        expanded_proprio_weight = current_state[proprio_key].clone()
        if previous_proprio_weight.shape[1] != JOINT_PROPRIO_DIM:
            raise ValueError(
                "cannot expand unexpected proprio projection shape: "
                f"{tuple(previous_proprio_weight.shape)}"
            )
        expanded_proprio_weight.zero_()
        expanded_proprio_weight[:, :JOINT_PROPRIO_DIM] = previous_proprio_weight
        initialization_state[proprio_key] = expanded_proprio_weight
        adapted_parameters.append(proprio_key)

        if checkpoint_config.grounding_action_conditioning:
            grounding_key = "grounding_action_projection.0.weight"
            previous_grounding_weight = initialization_state[grounding_key]
            expanded_grounding_weight = current_state[grounding_key].clone()
            expected_previous_inputs = JOINT_PROPRIO_DIM + 4
            if previous_grounding_weight.shape[1] != expected_previous_inputs:
                raise ValueError(
                    "cannot expand unexpected grounding-action projection shape: "
                    f"{tuple(previous_grounding_weight.shape)}"
                )
            expanded_grounding_weight.zero_()
            expanded_grounding_weight[:, :JOINT_PROPRIO_DIM] = (
                previous_grounding_weight[:, :JOINT_PROPRIO_DIM]
            )
            expanded_grounding_weight[:, CARTESIAN_PROPRIO_DIM:] = (
                previous_grounding_weight[:, JOINT_PROPRIO_DIM:]
            )
            initialization_state[grounding_key] = expanded_grounding_weight
            adapted_parameters.append(grounding_key)

    incompatible = model.load_state_dict(
        initialization_state,
        strict=not allowed_upgrade,
    )
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    allowed_missing_prefixes = tuple(
        prefix
        for field, prefix in (
            ("grounding_action_conditioning", "grounding_action_projection."),
            ("grounding_coordinate_refinement", "grounding_coordinate_refiner."),
            ("high_resolution_grounding", "high_resolution_grounding"),
            ("world_grounding", "world_grounding_head."),
            (
                "world_grounding_action_conditioning",
                "world_grounding_action_projection.",
            ),
            ("phase_action_conditioning", "phase_action_projection."),
        )
        if changed_fields.get(field) == {"from": False, "to": True}
    )
    if allowed_upgrade and (
        unexpected_keys
        or any(
            not key.startswith(allowed_missing_prefixes)
            for key in missing_keys
        )
    ):
        raise ValueError(
            "unexpected parameter mismatch during grounding-action upgrade: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "changed_model_fields": changed_fields,
        "randomly_initialized_parameters": missing_keys,
        "shape_adapted_parameters": adapted_parameters,
        "zero_initialized_proprio_features": (
            ["normalized_end_effector_x", "normalized_end_effector_y", "normalized_end_effector_z"]
            if proprio_upgrade
            else []
        ),
    }


def _freeze_backbone_for_high_resolution_grounding(model: TinyVLA) -> None:
    if not model.config.high_resolution_grounding:
        raise ValueError(
            "freezing for high-resolution grounding requires "
            "high_resolution_grounding=True"
        )
    trainable_prefixes = (
        "high_resolution_grounding_stem.",
        "high_resolution_grounding_key.",
        "high_resolution_grounding_query.",
        "high_resolution_grounding_gate",
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(trainable_prefixes))


def _build_training_sampling_weights(
    datasets: list[ActionChunkDataset | DAggerCorrectionDataset],
    *,
    train_config: TinyVLATrainConfig,
) -> Tensor:
    weights: list[Tensor] = []
    for dataset in datasets:
        source_weight = (
            train_config.correction_sample_weight
            if isinstance(dataset, DAggerCorrectionDataset)
            else 1.0
        )
        time_indices = torch.as_tensor(
            dataset.sample_time_indices,
            dtype=torch.long,
        )
        dataset_weights = torch.full(
            (len(dataset),),
            source_weight,
            dtype=torch.double,
        )
        early_mask = time_indices < train_config.early_window_steps
        initial_mask = time_indices == 0
        dataset_weights[early_mask] *= train_config.early_state_weight
        dataset_weights[initial_mask] = (
            source_weight * train_config.initial_state_weight
        )
        weights.append(dataset_weights)
    return torch.cat(weights)
