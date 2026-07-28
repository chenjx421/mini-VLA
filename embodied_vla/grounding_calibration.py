from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from embodied_vla.data import ActionChunkDataset
from embodied_vla.envs import SOArmEnvConfig, SOArmPickPlaceEnv
from embodied_vla.experts import ExpertPhase
from embodied_vla.models import TinyVLA, TinyVLAConfig
from embodied_vla.reproducibility import runtime_metadata
from embodied_vla.training.run_guard import claim_run_directory

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

CALIBRATION_SCHEMA_VERSION = 1
IDENTITY_AFFINE = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class GroundingCalibrationFitConfig:
    batch_size: int = 128
    num_workers: int = 0
    torch_threads: int = 1
    ridge: float = 1e-3

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.torch_threads <= 0:
            raise ValueError("torch_threads must be positive")
        if self.ridge < 0.0:
            raise ValueError("ridge cannot be negative")


class AffinePixelGroundingCalibration:
    """Post-hoc pixel calibration fitted without evaluation-scene labels."""

    def __init__(self, payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise ValueError("unsupported grounding calibration schema")
        transforms = payload.get("transforms")
        if not isinstance(transforms, dict):
            raise ValueError("calibration payload is missing transforms")
        self.payload = payload
        self._matrices: dict[str, NDArray[np.float64]] = {}
        self._features: dict[str, tuple[str, ...]] = {}
        for role in ("target", "goal"):
            feature_names = tuple(
                transforms[role].get("features", ("bias", "u", "v"))
            )
            matrix = np.asarray(transforms[role]["matrix"], dtype=np.float64)
            if matrix.shape != (len(feature_names), 2) or not np.isfinite(
                matrix
            ).all():
                raise ValueError(
                    f"{role} calibration matrix must have shape "
                    f"({len(feature_names)}, 2)"
                )
            self._features[role] = feature_names
            self._matrices[role] = matrix

    @classmethod
    def load(cls, path: Path) -> AffinePixelGroundingCalibration:
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def verify_checkpoint(self, checkpoint_path: Path) -> None:
        expected_sha256 = self.payload.get("checkpoint_sha256")
        if expected_sha256 is None:
            return
        actual_sha256 = _sha256_file(checkpoint_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "grounding calibration checkpoint mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

    def correct(
        self,
        coordinates: NDArray[np.floating],
        *,
        target_color: str | None = None,
        goal_side: str | None = None,
    ) -> NDArray[np.float32]:
        values = np.asarray(coordinates, dtype=np.float64)
        if values.shape != (2, 2) or not np.isfinite(values).all():
            raise ValueError("coordinates must be a finite array with shape (2, 2)")
        corrected = np.empty_like(values)
        for index, role in enumerate(("target", "goal")):
            features = _single_feature_vector(
                values[index],
                self._features[role],
                target_color=target_color,
                goal_side=goal_side,
            )
            corrected[index] = features @ self._matrices[role]
        return np.clip(corrected, 0.0, 1.0).astype(np.float32)


def fit_grounding_calibration(
    checkpoint_path: Path,
    dataset_root: Path,
    *,
    output_dir: Path,
    config: GroundingCalibrationFitConfig | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    if config is None:
        config = GroundingCalibrationFitConfig()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.torch_threads)
    try:
        with claim_run_directory(output_dir):
            return _fit_grounding_calibration_in_claimed_directory(
                checkpoint_path,
                dataset_root,
                output_dir=output_dir,
                config=config,
                device=device,
            )
    finally:
        torch.set_num_threads(previous_threads)


def _fit_grounding_calibration_in_claimed_directory(
    checkpoint_path: Path,
    dataset_root: Path,
    *,
    output_dir: Path,
    config: GroundingCalibrationFitConfig,
    device: str,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    torch_device = torch.device(device)
    model, model_config = _load_tiny_vla(checkpoint_path, device=torch_device)
    split_samples = {
        split: _collect_grounding_samples(
            model,
            model_config.action_horizon,
            model_config.proprio_dim,
            dataset_root,
            split=split,
            config=config,
            device=torch_device,
        )
        for split in ("train", "validation")
    }

    env = SOArmPickPlaceEnv(
        SOArmEnvConfig(
            observation_mode="multimodal",
            image_size=model_config.image_size,
        )
    )
    try:
        env.reset(seed=0)
        transforms: dict[str, dict[str, Any]] = {}
        role_reports: dict[str, Any] = {}
        corrected_validation: dict[str, NDArray[np.float64]] = {}
        for role, plane_height in (
            ("target", env.config.cube_half_size),
            ("goal", 0.004),
        ):
            train_role = split_samples["train"][role]
            validation_role = split_samples["validation"][role]
            affine_features = ("bias", "u", "v")
            language_features = (
                ("bias", "u", "v", "is_green", "is_blue", "is_right")
                if role == "target"
                else ("bias", "u", "v", "is_right")
            )
            affine_matrix = _fit_linear(
                train_role,
                affine_features,
                ridge=config.ridge,
            )
            language_matrix = _fit_linear(
                train_role,
                language_features,
                ridge=config.ridge,
            )
            candidates = {
                "identity": (affine_features, IDENTITY_AFFINE),
                "global_affine": (affine_features, affine_matrix),
                "language_conditioned_affine": (
                    language_features,
                    language_matrix,
                ),
            }
            candidate_reports = {
                name: _calibration_metrics(
                    env,
                    validation_role,
                    matrix,
                    feature_names=feature_names,
                    plane_height=plane_height,
                )
                for name, (feature_names, matrix) in candidates.items()
            }
            selected_name = min(
                candidates,
                key=lambda name: candidate_reports[name][
                    "mean_world_xy_error_m"
                ],
            )
            selected_features, selected_matrix = candidates[selected_name]
            transforms[role] = {
                "kind": "linear_predicted_pixel_to_corrected_pixel",
                "enabled": selected_name != "identity",
                "features": list(selected_features),
                "matrix": selected_matrix.tolist(),
            }
            role_reports[role] = {
                "train_samples": int(len(train_role["predicted_pixels"])),
                "validation_samples": int(
                    len(validation_role["predicted_pixels"])
                ),
                "candidates": candidate_reports,
                "selected": selected_name,
            }
            corrected_validation[role] = _apply_affine(
                validation_role,
                selected_features,
                selected_matrix,
            )
    finally:
        env.close()

    statistics_path = dataset_root / "statistics.json"
    dataset_statistics = (
        json.loads(statistics_path.read_text(encoding="utf-8"))
        if statistics_path.exists()
        else {}
    )
    payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "dataset": str(dataset_root.resolve()),
        "dataset_fingerprint_sha256": dataset_statistics.get(
            "dataset_fingerprint_sha256"
        ),
        "fit_scope": {
            "parameter_fit_split": "train",
            "model_selection_split": "validation",
            "target_phases": [
                ExpertPhase.APPROACH.name.lower(),
                ExpertPhase.DESCEND_GRASP.name.lower(),
                ExpertPhase.CLOSE_GRIPPER.name.lower(),
            ],
            "evaluation_scene_labels_used": False,
            "language_features": ["target color", "goal side"],
        },
        "transforms": transforms,
        "validation": role_reports,
        "fit": asdict(config),
        "elapsed_seconds": time.perf_counter() - start_time,
        "runtime": runtime_metadata(device=torch_device),
    }
    (output_dir / "calibration.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    _plot_calibration(
        split_samples["validation"],
        corrected_validation,
        output_dir / "grounding_calibration.png",
    )
    return payload


def _collect_grounding_samples(
    model: torch.nn.Module,
    action_horizon: int,
    proprio_dim: int,
    dataset_root: Path,
    *,
    split: str,
    config: GroundingCalibrationFitConfig,
    device: torch.device,
) -> dict[str, dict[str, NDArray[Any]]]:
    dataset = ActionChunkDataset(
        dataset_root,
        action_horizon=action_horizon,
        split=split,
        proprio_dim=proprio_dim,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    buffers: dict[str, dict[str, list[NDArray[Any]]]] = {
        role: {
            "predicted_pixels": [],
            "true_pixels": [],
            "true_world_xy": [],
            "target_color": [],
            "goal_side": [],
        }
        for role in ("target", "goal")
    }
    for batch in loader:
        rgb = batch["rgb"].to(device)
        proprio = batch["proprio"].to(device)
        language = batch["language"].to(device)
        language_mask = batch["language_mask"].to(device)
        with torch.inference_mode():
            output = model(rgb, proprio, language, language_mask)
        predicted = output.grounding_coordinates.detach().cpu()
        early_target = batch["phase"] <= int(ExpertPhase.CLOSE_GRIPPER)
        role_specs = (
            ("target", 0, early_target),
            ("goal", 1, torch.ones_like(early_target, dtype=torch.bool)),
        )
        for role, role_index, phase_mask in role_specs:
            valid = phase_mask & batch["pixel_valid"][:, role_index].bool()
            episode_indices = batch["episode_index"][valid].tolist()
            buffers[role]["predicted_pixels"].append(
                predicted[valid, role_index].numpy()
            )
            buffers[role]["true_pixels"].append(
                batch[f"{role}_pixel"][valid].numpy()
            )
            buffers[role]["true_world_xy"].append(
                batch[f"{role}_world"][valid, :2].numpy() * 0.5
            )
            buffers[role]["target_color"].append(
                np.asarray(
                    [
                        dataset.records[int(index)]["target_color"]
                        for index in episode_indices
                    ]
                )
            )
            buffers[role]["goal_side"].append(
                np.asarray(
                    [
                        dataset.records[int(index)]["goal_side"]
                        for index in episode_indices
                    ]
                )
            )
    result: dict[str, dict[str, NDArray[Any]]] = {}
    for role, role_buffers in buffers.items():
        result[role] = {}
        for name, chunks in role_buffers.items():
            values = np.concatenate(chunks)
            if name in {"predicted_pixels", "true_pixels", "true_world_xy"}:
                values = values.astype(np.float32, copy=False)
            result[role][name] = values
    return result


def _fit_linear(
    samples: dict[str, NDArray[Any]],
    feature_names: tuple[str, ...],
    *,
    ridge: float,
) -> NDArray[np.float64]:
    design = _feature_matrix(samples, feature_names)
    target = np.asarray(samples["true_pixels"], dtype=np.float64)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * ridge
    regularizer[0, 0] = 0.0
    return np.linalg.solve(
        design.T @ design + regularizer,
        design.T @ target,
    )


def _apply_affine(
    samples: dict[str, NDArray[Any]],
    feature_names: tuple[str, ...],
    matrix: NDArray[np.floating],
) -> NDArray[np.float64]:
    return np.clip(
        _feature_matrix(samples, feature_names)
        @ np.asarray(matrix, dtype=np.float64),
        0.0,
        1.0,
    )


def _feature_matrix(
    samples: dict[str, NDArray[Any]],
    feature_names: tuple[str, ...],
) -> NDArray[np.float64]:
    pixels = np.asarray(samples["predicted_pixels"], dtype=np.float64)
    columns = {
        "bias": np.ones(len(pixels), dtype=np.float64),
        "u": pixels[:, 0],
        "v": pixels[:, 1],
        "is_green": np.asarray(samples["target_color"]) == "green",
        "is_blue": np.asarray(samples["target_color"]) == "blue",
        "is_right": np.asarray(samples["goal_side"]) == "right",
    }
    try:
        return np.column_stack([columns[name] for name in feature_names])
    except KeyError as error:
        raise ValueError(f"unsupported calibration feature: {error.args[0]}") from error


def _single_feature_vector(
    pixel: NDArray[np.float64],
    feature_names: tuple[str, ...],
    *,
    target_color: str | None,
    goal_side: str | None,
) -> NDArray[np.float64]:
    values = {
        "bias": 1.0,
        "u": float(pixel[0]),
        "v": float(pixel[1]),
        "is_green": float(target_color == "green"),
        "is_blue": float(target_color == "blue"),
        "is_right": float(goal_side == "right"),
    }
    if (
        any(name in feature_names for name in ("is_green", "is_blue"))
        and target_color not in {"red", "green", "blue"}
    ):
        raise ValueError("language-conditioned calibration requires target_color")
    if "is_right" in feature_names and goal_side not in {"left", "right"}:
        raise ValueError("language-conditioned calibration requires goal_side")
    return np.asarray([values[name] for name in feature_names], dtype=np.float64)


def _calibration_metrics(
    env: SOArmPickPlaceEnv,
    samples: dict[str, NDArray[Any]],
    matrix: NDArray[np.float64],
    *,
    feature_names: tuple[str, ...],
    plane_height: float,
) -> dict[str, float]:
    corrected = _apply_affine(samples, feature_names, matrix)
    pixel_error = np.linalg.norm(corrected - samples["true_pixels"], axis=1)
    world_xy = np.asarray(
        [
            env.unproject_normalized_pixel_to_plane(
                pixel,
                world_z=plane_height,
            )[:2]
            for pixel in corrected
        ]
    )
    world_error = np.linalg.norm(world_xy - samples["true_world_xy"], axis=1)
    return {
        "mean_pixel_l2": float(pixel_error.mean()),
        "median_pixel_l2": float(np.median(pixel_error)),
        "mean_world_xy_error_m": float(world_error.mean()),
        "median_world_xy_error_m": float(np.median(world_error)),
        "p90_world_xy_error_m": float(np.percentile(world_error, 90)),
        "fraction_within_13mm": float(np.mean(world_error <= 0.013)),
    }


def _plot_calibration(
    validation: dict[str, dict[str, NDArray[np.float32]]],
    corrected: dict[str, NDArray[np.float64]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for role, color in (("target", "#C82423"), ("goal", "#2878B5")):
        samples = validation[role]
        before = np.linalg.norm(
            samples["predicted_pixels"] - samples["true_pixels"],
            axis=1,
        )
        after = np.linalg.norm(
            corrected[role] - samples["true_pixels"],
            axis=1,
        )
        axes[0].hist(
            before,
            bins=35,
            alpha=0.45,
            color=color,
            label=f"{role} before",
        )
        axes[0].hist(
            after,
            bins=35,
            histtype="step",
            linewidth=2,
            color=color,
            label=f"{role} after",
        )
    axes[0].set_title("Validation pixel-error distribution")
    axes[0].set_xlabel("Normalized pixel L2")
    axes[0].set_ylabel("Frames")
    axes[0].legend()

    target = validation["target"]
    axes[1].scatter(
        target["true_pixels"][:, 0],
        target["true_pixels"][:, 1],
        s=8,
        alpha=0.35,
        color="#8A9197",
        label="ground truth",
    )
    axes[1].scatter(
        corrected["target"][:, 0],
        corrected["target"][:, 1],
        s=8,
        alpha=0.35,
        color="#C82423",
        label="calibrated prediction",
    )
    axes[1].invert_yaxis()
    axes[1].set_aspect("equal")
    axes[1].set_title("Target grounding on held-out episodes")
    axes[1].set_xlabel("u")
    axes[1].set_ylabel("v")
    axes[1].legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_tiny_vla(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[TinyVLA, TinyVLAConfig]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model_config = TinyVLAConfig(**checkpoint["model_config"])
    model = TinyVLA(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, model_config
