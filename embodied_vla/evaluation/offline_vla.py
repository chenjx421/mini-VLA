from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from embodied_vla.data import ActionChunkDataset
from embodied_vla.evaluation.vla import load_tiny_vla
from embodied_vla.experts import ExpertPhase
from embodied_vla.reproducibility import runtime_metadata
from embodied_vla.training.run_guard import claim_run_directory

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ACTION_NAMES = ("dx", "dy", "dz", "wrist", "jaw")


@dataclass(frozen=True)
class OfflineVLAEvalConfig:
    split: Literal["train", "validation"] = "validation"
    batch_size: int = 64
    num_workers: int = 0
    progress_bins: int = 10
    seed: int = 2026
    torch_threads: int = 1

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.progress_bins <= 0:
            raise ValueError("progress_bins must be positive")
        if self.torch_threads <= 0:
            raise ValueError("torch_threads must be positive")


def evaluate_tiny_vla_offline(
    checkpoint_path: Path,
    dataset_root: Path,
    *,
    output_dir: Path,
    eval_config: OfflineVLAEvalConfig | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    if eval_config is None:
        eval_config = OfflineVLAEvalConfig()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(eval_config.torch_threads)
    try:
        with claim_run_directory(output_dir):
            return _evaluate_tiny_vla_offline_in_claimed_directory(
                checkpoint_path,
                dataset_root,
                output_dir=output_dir,
                eval_config=eval_config,
                device=device,
            )
    finally:
        torch.set_num_threads(previous_threads)


def _evaluate_tiny_vla_offline_in_claimed_directory(
    checkpoint_path: Path,
    dataset_root: Path,
    *,
    output_dir: Path,
    eval_config: OfflineVLAEvalConfig,
    device: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(eval_config.seed)
    torch_device = torch.device(device)
    model, model_config = load_tiny_vla(checkpoint_path, device=torch_device)
    dataset = ActionChunkDataset(
        dataset_root,
        action_horizon=model_config.action_horizon,
        split=eval_config.split,
        proprio_dim=model_config.proprio_dim,
    )
    loader = DataLoader(
        dataset,
        batch_size=eval_config.batch_size,
        shuffle=False,
        num_workers=eval_config.num_workers,
    )

    phase_count = model_config.phase_count
    confusion = np.zeros((phase_count, phase_count), dtype=np.int64)
    sample_records: list[dict[str, Any]] = []
    sample_path = output_dir / "samples.jsonl"
    with sample_path.open("w", encoding="utf-8") as sample_handle:
        for batch in loader:
            batch_device = {
                key: value.to(torch_device)
                for key, value in batch.items()
                if key not in {"episode_index", "time_index", "episode_length"}
            }
            with torch.inference_mode():
                output = model(
                    batch_device["rgb"],
                    batch_device["proprio"],
                    batch_device["language"],
                    batch_device["language_mask"],
                )
            records = _batch_records(
                output.action_chunk,
                output.phase_logits,
                output.grounding_coordinates,
                output.grounding_world_positions,
                batch_device,
                batch,
                dataset,
                progress_bins=eval_config.progress_bins,
            )
            for record in records:
                true_phase = int(record["true_phase"])
                predicted_phase = int(record["predicted_phase"])
                confusion[true_phase, predicted_phase] += 1
                sample_records.append(record)
                sample_handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    summary = _summarize_records(
        sample_records,
        confusion,
        checkpoint_path=checkpoint_path,
        dataset_root=dataset_root,
        dataset=dataset,
        config=eval_config,
        action_dim=model_config.action_dim,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    _plot_diagnostics(summary, output_dir / "offline_diagnostics.png")
    return summary


def _batch_records(
    predicted_chunks: Tensor,
    phase_logits: Tensor,
    predicted_grounding: Tensor,
    predicted_world_grounding: Tensor | None,
    batch_device: dict[str, Tensor],
    batch_cpu: dict[str, Tensor],
    dataset: ActionChunkDataset,
    *,
    progress_bins: int,
) -> list[dict[str, Any]]:
    target_chunks = batch_device["action_chunk"]
    action_mask = batch_device["action_mask"].bool()
    absolute_error = (predicted_chunks - target_chunks).abs()
    valid_dimensions = action_mask.sum(dim=1).clamp_min(1) * predicted_chunks.shape[-1]
    chunk_mae = (
        absolute_error * action_mask.unsqueeze(-1)
    ).sum(dim=(1, 2)) / valid_dimensions
    first_action_error = absolute_error[:, 0]
    predicted_phases = phase_logits.argmax(dim=-1)
    true_phases = batch_device["phase"]

    ground_truth = torch.stack(
        (batch_device["target_pixel"], batch_device["goal_pixel"]),
        dim=1,
    )
    pixel_valid = batch_device["pixel_valid"].float()
    grounding_error = torch.linalg.vector_norm(
        predicted_grounding - ground_truth,
        dim=-1,
    )
    grounding_l2 = (
        grounding_error * pixel_valid
    ).sum(dim=1) / pixel_valid.sum(dim=1).clamp_min(1.0)
    world_grounding_l2_m = None
    if predicted_world_grounding is not None:
        ground_truth_world = torch.stack(
            (batch_device["target_world"], batch_device["goal_world"]),
            dim=1,
        )
        world_grounding_error = torch.linalg.vector_norm(
            predicted_world_grounding - ground_truth_world,
            dim=-1,
        )
        world_grounding_l2_m = (
            (world_grounding_error * pixel_valid).sum(dim=1)
            / pixel_valid.sum(dim=1).clamp_min(1.0)
            * 0.5
        )

    predicted_chunks_cpu = predicted_chunks.detach().cpu().numpy()
    target_chunks_cpu = target_chunks.detach().cpu().numpy()
    first_action_error_cpu = first_action_error.detach().cpu().numpy()
    records: list[dict[str, Any]] = []
    for batch_index in range(predicted_chunks.shape[0]):
        local_episode_index = int(batch_cpu["episode_index"][batch_index])
        time_index = int(batch_cpu["time_index"][batch_index])
        episode_length = int(batch_cpu["episode_length"][batch_index])
        progress = time_index / max(1, episode_length - 1)
        progress_bin = min(int(progress * progress_bins), progress_bins - 1)
        episode_record = dataset.records[local_episode_index]
        records.append(
            {
                "episode": int(episode_record.get("episode", local_episode_index)),
                "episode_path": str(episode_record["path"]),
                "time_index": time_index,
                "episode_length": episode_length,
                "progress": progress,
                "progress_bin": progress_bin,
                "target_color": episode_record.get("target_color"),
                "goal_side": episode_record.get("goal_side"),
                "true_phase": int(true_phases[batch_index]),
                "true_phase_name": _phase_name(int(true_phases[batch_index])),
                "predicted_phase": int(predicted_phases[batch_index]),
                "predicted_phase_name": _phase_name(
                    int(predicted_phases[batch_index])
                ),
                "phase_correct": bool(
                    predicted_phases[batch_index] == true_phases[batch_index]
                ),
                "chunk_mae": float(chunk_mae[batch_index]),
                "first_action_mae": float(first_action_error[batch_index].mean()),
                "first_action_mae_by_dimension": first_action_error_cpu[
                    batch_index
                ].tolist(),
                "predicted_first_action": predicted_chunks_cpu[batch_index, 0].tolist(),
                "expert_first_action": target_chunks_cpu[batch_index, 0].tolist(),
                "grounding_l2": float(grounding_l2[batch_index]),
                "world_grounding_l2_m": (
                    float(world_grounding_l2_m[batch_index])
                    if world_grounding_l2_m is not None
                    else None
                ),
            }
        )
    return records


def _summarize_records(
    records: list[dict[str, Any]],
    confusion: np.ndarray,
    *,
    checkpoint_path: Path,
    dataset_root: Path,
    dataset: ActionChunkDataset,
    config: OfflineVLAEvalConfig,
    action_dim: int,
) -> dict[str, Any]:
    if not records:
        raise RuntimeError("offline evaluation dataset is empty")
    first_action_errors = np.asarray(
        [record["first_action_mae_by_dimension"] for record in records],
        dtype=np.float64,
    )
    phases = sorted({int(record["true_phase"]) for record in records})
    by_phase = {
        _phase_name(phase): _group_summary(
            [record for record in records if int(record["true_phase"]) == phase],
            action_dim,
        )
        for phase in phases
    }
    by_progress = {}
    for progress_bin in range(config.progress_bins):
        lower = 100 * progress_bin / config.progress_bins
        upper = 100 * (progress_bin + 1) / config.progress_bins
        label = f"{lower:.0f}-{upper:.0f}%"
        bin_records = [
            record
            for record in records
            if int(record["progress_bin"]) == progress_bin
        ]
        if bin_records:
            by_progress[label] = _group_summary(bin_records, action_dim)

    initial_records = [record for record in records if int(record["time_index"]) == 0]
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset": str(dataset_root.resolve()),
        "split": config.split,
        "split_strategy": dataset.split_strategy,
        "episodes": len(dataset.records),
        "samples": len(records),
        "metrics": _group_summary(records, action_dim),
        "initial_state_metrics": _group_summary(initial_records, action_dim),
        "first_action_mae_by_dimension": {
            name: float(first_action_errors[:, index].mean())
            for index, name in enumerate(ACTION_NAMES[:action_dim])
        },
        "by_true_phase": by_phase,
        "by_progress": by_progress,
        "phase_confusion_matrix": confusion.tolist(),
        "phase_labels": [_phase_name(index) for index in range(confusion.shape[0])],
        "evaluation": asdict(config),
        "runtime": runtime_metadata(),
    }


def _group_summary(records: list[dict[str, Any]], action_dim: int) -> dict[str, Any]:
    if not records:
        return {
            "samples": 0,
            "chunk_mae": None,
            "first_action_mae": None,
            "phase_accuracy": None,
            "grounding_l2": None,
            "world_grounding_l2_m": None,
            "first_action_mae_by_dimension": {},
        }
    dimension_errors = np.asarray(
        [record["first_action_mae_by_dimension"] for record in records],
        dtype=np.float64,
    )
    predicted_actions = np.asarray(
        [record["predicted_first_action"] for record in records],
        dtype=np.float64,
    )
    expert_actions = np.asarray(
        [record["expert_first_action"] for record in records],
        dtype=np.float64,
    )
    world_grounding_values = [
        float(record["world_grounding_l2_m"])
        for record in records
        if record["world_grounding_l2_m"] is not None
    ]
    return {
        "samples": len(records),
        "chunk_mae": float(np.mean([record["chunk_mae"] for record in records])),
        "first_action_mae": float(
            np.mean([record["first_action_mae"] for record in records])
        ),
        "phase_accuracy": float(np.mean([record["phase_correct"] for record in records])),
        "grounding_l2": float(np.mean([record["grounding_l2"] for record in records])),
        "world_grounding_l2_m": (
            float(np.mean(world_grounding_values))
            if world_grounding_values
            else None
        ),
        "first_action_mae_by_dimension": {
            name: float(dimension_errors[:, index].mean())
            for index, name in enumerate(ACTION_NAMES[:action_dim])
        },
        "predicted_first_action_std": {
            name: float(predicted_actions[:, index].std())
            for index, name in enumerate(ACTION_NAMES[:action_dim])
        },
        "expert_first_action_std": {
            name: float(expert_actions[:, index].std())
            for index, name in enumerate(ACTION_NAMES[:action_dim])
        },
        "first_action_correlation": {
            name: _pearson_correlation(
                predicted_actions[:, index],
                expert_actions[:, index],
            )
            for index, name in enumerate(ACTION_NAMES[:action_dim])
        },
    }


def _plot_diagnostics(summary: dict[str, Any], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    phase_metrics = summary["by_true_phase"]
    phase_labels = list(phase_metrics)
    phase_mae = [phase_metrics[label]["first_action_mae"] for label in phase_labels]
    axes[0, 0].bar(phase_labels, phase_mae, color="#2878B5")
    axes[0, 0].set_title("First-action MAE by expert phase")
    axes[0, 0].set_ylabel("MAE")
    axes[0, 0].tick_params(axis="x", rotation=30)

    progress_metrics = summary["by_progress"]
    progress_labels = list(progress_metrics)
    progress_mae = [
        progress_metrics[label]["first_action_mae"] for label in progress_labels
    ]
    axes[0, 1].plot(progress_labels, progress_mae, marker="o", color="#C82423")
    axes[0, 1].set_title("First-action MAE by trajectory progress")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].tick_params(axis="x", rotation=35)

    confusion = np.asarray(summary["phase_confusion_matrix"], dtype=np.int64)
    image = axes[1, 0].imshow(confusion, cmap="Blues")
    axes[1, 0].set_title("Phase confusion: true rows, predicted columns")
    axes[1, 0].set_xlabel("Predicted phase")
    axes[1, 0].set_ylabel("True phase")
    tick_labels = summary["phase_labels"]
    axes[1, 0].set_xticks(range(len(tick_labels)), tick_labels, rotation=45, ha="right")
    axes[1, 0].set_yticks(range(len(tick_labels)), tick_labels)
    figure.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    dimension_metrics = summary["first_action_mae_by_dimension"]
    axes[1, 1].bar(
        list(dimension_metrics),
        list(dimension_metrics.values()),
        color=["#2878B5", "#9AC9DB", "#F8AC8C", "#C82423", "#8A9197"],
    )
    axes[1, 1].set_title("First-action MAE by action dimension")
    axes[1, 1].set_ylabel("MAE")

    figure.suptitle(
        f"Tiny-VLA offline diagnostics ({summary['split']}, n={summary['samples']})",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _phase_name(phase: int) -> str:
    try:
        return ExpertPhase(phase).name.lower()
    except ValueError:
        return f"phase_{phase}"


def _pearson_correlation(
    predicted: np.ndarray,
    expected: np.ndarray,
) -> float | None:
    if predicted.size < 2 or predicted.std() < 1e-12 or expected.std() < 1e-12:
        return None
    return float(np.corrcoef(predicted, expected)[0, 1])
