from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from numpy.typing import NDArray

from embodied_vla.control import GroundedVisualServoController
from embodied_vla.envs import SOArmEnvConfig, SOArmPickPlaceEnv
from embodied_vla.envs.so_arm_pick_place import COLORS, SIDES
from embodied_vla.evaluation.vla import load_tiny_vla, predict_tiny_vla
from embodied_vla.experts import ExpertPhase, PickPlaceExpert
from embodied_vla.grounding_calibration import AffinePixelGroundingCalibration
from embodied_vla.metrics import wilson_score_interval
from embodied_vla.proprioception import uses_end_effector_position
from embodied_vla.reproducibility import runtime_metadata
from embodied_vla.training.run_guard import claim_run_directory
from embodied_vla.visualization import make_vla_attention_panel


@dataclass(frozen=True)
class HybridVLAEvalConfig:
    episodes: int = 20
    seed: int = 40_000
    video_episodes: int = 3
    max_episode_steps: int = 300
    domain_randomization: bool = False
    balanced_tasks: bool = True
    torch_threads: int = 1
    smoothing_alpha: float = 0.35
    recovery_search_radius_m: float = 0.0
    close_retry_steps: int = 35

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.video_episodes < 0:
            raise ValueError("video_episodes cannot be negative")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.torch_threads <= 0:
            raise ValueError("torch_threads must be positive")
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must lie in (0, 1]")
        if self.recovery_search_radius_m < 0.0:
            raise ValueError("recovery_search_radius_m cannot be negative")
        if self.close_retry_steps <= 0:
            raise ValueError("close_retry_steps must be positive")


def evaluate_hybrid_vla(
    checkpoint_path: Path,
    *,
    output_dir: Path,
    eval_config: HybridVLAEvalConfig,
    device: str = "cpu",
    grounding_calibration_path: Path | None = None,
) -> dict[str, Any]:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(eval_config.torch_threads)
    try:
        with claim_run_directory(output_dir):
            return _evaluate_hybrid_vla_in_claimed_directory(
                checkpoint_path,
                output_dir=output_dir,
                eval_config=eval_config,
                device=device,
                grounding_calibration_path=grounding_calibration_path,
            )
    finally:
        torch.set_num_threads(previous_threads)


def _evaluate_hybrid_vla_in_claimed_directory(
    checkpoint_path: Path,
    *,
    output_dir: Path,
    eval_config: HybridVLAEvalConfig,
    device: str,
    grounding_calibration_path: Path | None,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    model, model_config = load_tiny_vla(checkpoint_path, device=torch_device)
    env_config = SOArmEnvConfig(
        observation_mode="multimodal",
        task_level="pick_place",
        grasp_mode="contact_assisted",
        image_size=model_config.image_size,
        max_episode_steps=eval_config.max_episode_steps,
        domain_randomization=eval_config.domain_randomization,
        include_end_effector_position_in_proprio=uses_end_effector_position(
            model_config.proprio_dim
        ),
    )
    env = SOArmPickPlaceEnv(env_config)
    controller = GroundedVisualServoController(
        env_config,
        smoothing_alpha=eval_config.smoothing_alpha,
        recovery_search_radius_m=eval_config.recovery_search_radius_m,
        close_retry_steps=eval_config.close_retry_steps,
    )
    grounding_calibration = (
        AffinePixelGroundingCalibration.load(grounding_calibration_path)
        if grounding_calibration_path is not None
        else None
    )
    if grounding_calibration is not None:
        grounding_calibration.verify_checkpoint(checkpoint_path)
    diagnostic_expert = PickPlaceExpert(env_config)
    records: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    metrics_path = output_dir / "episodes.jsonl"
    start_time = time.perf_counter()
    try:
        for episode_index in range(eval_config.episodes):
            task_index = episode_index % (len(COLORS) * len(SIDES))
            reset_options = (
                {
                    "target_color": COLORS[task_index // len(SIDES)],
                    "goal_side": SIDES[task_index % len(SIDES)],
                }
                if eval_config.balanced_tasks
                else None
            )
            episode_seed = eval_config.seed + episode_index
            observation, info = env.reset(
                seed=episode_seed,
                options=reset_options,
            )
            instruction_target_color, instruction_goal_side = (
                _task_from_instruction(str(info["instruction"]))
            )
            controller.reset()
            diagnostic_expert.reset()
            episode_return = 0.0
            raw_pixel_errors: list[float] = []
            calibrated_pixel_errors: list[float] = []
            pregrasp_target_world_errors: list[float] = []
            close_target_world_errors: list[float] = []
            goal_world_errors: list[float] = []
            controller_expert_errors: list[float] = []
            controller_phases: list[int] = []
            model_phases: list[int] = []
            trace_records: list[dict[str, Any]] = []
            frames: list[NDArray[np.uint8]] = []
            bilateral_contact_steps = 0
            ever_grasped = False
            ever_lifted = False
            terminated = False
            truncated = False

            while not (terminated or truncated):
                inference_start = time.perf_counter()
                output = predict_tiny_vla(model, observation, torch_device)
                latencies_ms.append(
                    (time.perf_counter() - inference_start) * 1_000.0
                )
                raw_predicted_coordinates = (
                    output.grounding_coordinates.squeeze(0).detach().cpu().numpy()
                )
                predicted_coordinates = (
                    grounding_calibration.correct(
                        raw_predicted_coordinates,
                        target_color=instruction_target_color,
                        goal_side=instruction_goal_side,
                    )
                    if grounding_calibration is not None
                    else raw_predicted_coordinates
                )
                model_phase = int(output.phase_logits.argmax(dim=-1).item())
                model_phases.append(model_phase)
                controller_phase = int(controller.phase)
                controller_phases.append(controller_phase)

                target_estimate = env.unproject_normalized_pixel_to_plane(
                    predicted_coordinates[0],
                    world_z=env_config.cube_half_size,
                )
                goal_estimate = env.unproject_normalized_pixel_to_plane(
                    predicted_coordinates[1],
                    world_z=0.004,
                )
                target_pixel, target_visible = env.project_world_point(
                    info["target_position"]
                )
                goal_pixel, goal_visible = env.project_world_point(
                    info["goal_position"]
                )
                ground_truth_pixels = np.stack((target_pixel, goal_pixel))
                pixel_valid = np.asarray(
                    (target_visible, goal_visible),
                    dtype=np.bool_,
                )
                if pixel_valid.any():
                    raw_pixel_errors.extend(
                        np.linalg.norm(
                            raw_predicted_coordinates - ground_truth_pixels,
                            axis=-1,
                        )[pixel_valid].tolist()
                    )
                    calibrated_pixel_errors.extend(
                        np.linalg.norm(
                            predicted_coordinates - ground_truth_pixels,
                            axis=-1,
                        )[pixel_valid].tolist()
                    )
                target_world_error = float(
                    np.linalg.norm(
                        target_estimate[:2]
                        - np.asarray(info["target_position"])[:2]
                    )
                )
                if not bool(info["has_grasped"]):
                    pregrasp_target_world_errors.append(target_world_error)
                    if controller.phase in {
                        ExpertPhase.DESCEND_GRASP,
                        ExpertPhase.CLOSE_GRIPPER,
                    }:
                        close_target_world_errors.append(target_world_error)
                goal_world_errors.append(
                    float(
                        np.linalg.norm(
                            goal_estimate[:2]
                            - np.asarray(info["goal_position"])[:2]
                        )
                    )
                )

                allowed_controller_info = {
                    "end_effector_position": info["end_effector_position"],
                    "bilateral_contact": info["bilateral_contact"],
                    "assisted_grasp_active": info["assisted_grasp_active"],
                }
                search_offset = controller.search_offset.copy()
                action = controller.act(
                    allowed_controller_info,
                    target_position_estimate=target_estimate,
                    goal_position_estimate=goal_estimate,
                )
                expert_action = diagnostic_expert.act(info)
                controller_expert_errors.append(
                    float(np.mean(np.abs(action - expert_action)))
                )
                direct_action = (
                    output.action_chunk[0, 0].detach().cpu().numpy()
                )
                trace_records.append(
                    {
                        "step": int(info["step"]),
                        "controller_phase": ExpertPhase(
                            controller_phase
                        ).name.lower(),
                        "model_phase": ExpertPhase(model_phase).name.lower(),
                        "hybrid_action": action.tolist(),
                        "direct_vla_action": direct_action.tolist(),
                        "diagnostic_expert_action": expert_action.tolist(),
                        "raw_predicted_target_pixel": (
                            raw_predicted_coordinates[0].tolist()
                        ),
                        "raw_predicted_goal_pixel": (
                            raw_predicted_coordinates[1].tolist()
                        ),
                        "controller_target_pixel": predicted_coordinates[0].tolist(),
                        "controller_goal_pixel": predicted_coordinates[1].tolist(),
                        "calibrated_target_world": target_estimate.tolist(),
                        "calibrated_goal_world": goal_estimate.tolist(),
                        "diagnostic_true_target_world": np.asarray(
                            info["target_position"]
                        ).tolist(),
                        "diagnostic_true_goal_world": np.asarray(
                            info["goal_position"]
                        ).tolist(),
                        "target_world_error_xy_m": target_world_error,
                        "goal_world_error_xy_m": goal_world_errors[-1],
                        "recovery_search_offset_m": search_offset.tolist(),
                        "bilateral_contact": bool(info["bilateral_contact"]),
                        "has_grasped": bool(info["has_grasped"]),
                        "has_lifted": bool(info["has_lifted"]),
                    }
                )
                if episode_index < eval_config.video_episodes:
                    frames.append(
                        make_vla_attention_panel(
                            observation["rgb"],
                            output.grounding_heatmaps.squeeze(0)
                            .detach()
                            .cpu()
                            .numpy(),
                            instruction=str(info["instruction"]),
                            predicted_phase=controller_phase,
                            step=int(info["step"]),
                            predicted_coordinates=predicted_coordinates,
                            ground_truth_coordinates=ground_truth_pixels,
                            pixel_valid=pixel_valid,
                        )
                    )

                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                bilateral_contact_steps += int(bool(info["bilateral_contact"]))
                ever_grasped = ever_grasped or bool(info["has_grasped"])
                ever_lifted = ever_lifted or bool(info["has_lifted"])

            record = {
                "episode": episode_index,
                "seed": episode_seed,
                "success": bool(info["success"]),
                "return": float(episode_return),
                "length": int(info["step"]),
                "termination_reason": info["termination_reason"],
                "target_color": info["target_color"],
                "goal_side": info["goal_side"],
                "mean_raw_pixel_grounding_l2": _safe_mean(raw_pixel_errors),
                "mean_calibrated_pixel_grounding_l2": _safe_mean(
                    calibrated_pixel_errors
                ),
                "mean_pregrasp_target_world_xy_error_m": _safe_mean(
                    pregrasp_target_world_errors
                ),
                "mean_descend_close_target_world_xy_error_m": _safe_mean(
                    close_target_world_errors
                ),
                "mean_goal_world_xy_error_m": _safe_mean(goal_world_errors),
                "mean_controller_expert_mae": float(
                    np.mean(controller_expert_errors)
                ),
                "bilateral_contact_steps": bilateral_contact_steps,
                "ever_grasped": ever_grasped,
                "ever_lifted": ever_lifted,
                "controller_retries": controller.retries,
                "final_controller_phase": controller.phase.name.lower(),
                "controller_phase_histogram": _phase_histogram(controller_phases),
                "model_phase_histogram": _phase_histogram(model_phases),
            }
            records.append(record)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            with (
                output_dir / f"episode_{episode_index:03d}_trace.jsonl"
            ).open("w", encoding="utf-8") as handle:
                for trace_record in trace_records:
                    handle.write(json.dumps(trace_record, ensure_ascii=True) + "\n")
            if frames:
                imageio.mimsave(
                    output_dir / f"episode_{episode_index:03d}.gif",
                    frames,
                    duration=0.10,
                    loop=0,
                )
                imageio.imwrite(
                    output_dir / f"episode_{episode_index:03d}_attention.png",
                    frames[len(frames) // 2],
                )
            print(
                f"hybrid episode={episode_index:03d} seed={episode_seed} "
                f"success={bool(info['success'])} return={episode_return:.3f} "
                f"steps={int(info['step'])}"
            )
    finally:
        env.close()

    successes = sum(record["success"] for record in records)
    confidence_low, confidence_high = wilson_score_interval(
        successes,
        eval_config.episodes,
    )
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "episodes": eval_config.episodes,
        "successes": successes,
        "success_rate": successes / eval_config.episodes,
        "success_wilson_95": [confidence_low, confidence_high],
        "task_success": _task_success(records),
        "mean_return": float(np.mean([record["return"] for record in records])),
        "mean_length": float(np.mean([record["length"] for record in records])),
        "mean_raw_pixel_grounding_l2": float(
            np.mean(
                [record["mean_raw_pixel_grounding_l2"] for record in records]
            )
        ),
        "mean_calibrated_pixel_grounding_l2": float(
            np.mean(
                [
                    record["mean_calibrated_pixel_grounding_l2"]
                    for record in records
                ]
            )
        ),
        "mean_pregrasp_target_world_xy_error_m": float(
            np.mean(
                [
                    record["mean_pregrasp_target_world_xy_error_m"]
                    for record in records
                ]
            )
        ),
        "mean_descend_close_target_world_xy_error_m": _safe_mean(
            [
                record["mean_descend_close_target_world_xy_error_m"]
                for record in records
                if record["mean_descend_close_target_world_xy_error_m"] is not None
            ]
        ),
        "mean_goal_world_xy_error_m": float(
            np.mean([record["mean_goal_world_xy_error_m"] for record in records])
        ),
        "episodes_with_bilateral_contact": sum(
            record["bilateral_contact_steps"] > 0 for record in records
        ),
        "episodes_ever_grasped": sum(record["ever_grasped"] for record in records),
        "episodes_ever_lifted": sum(record["ever_lifted"] for record in records),
        "inference_latency_ms": {
            "mean": float(np.mean(latencies_ms)),
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "samples": len(latencies_ms),
        },
        "elapsed_seconds": time.perf_counter() - start_time,
        "model": asdict(model_config),
        "evaluation": asdict(eval_config),
        "grounding_calibration": (
            {
                "path": str(grounding_calibration_path.resolve()),
                "fit_scope": grounding_calibration.payload["fit_scope"],
                "transforms": grounding_calibration.payload["transforms"],
            }
            if grounding_calibration_path is not None
            and grounding_calibration is not None
            else None
        ),
        "policy_boundary": {
            "learned": "language-conditioned target/goal pixel grounding",
            "calibrated": (
                "train-split affine pixel correction and camera ray-plane "
                "unprojection"
                if grounding_calibration is not None
                else "camera ray-plane unprojection"
            ),
            "engineered": (
                "phase state machine, Cartesian visual servo, and "
                "contact-failure local search"
            ),
            "action_uses_privileged_target_or_goal_coordinates": False,
            "privileged_coordinates_used_for_metrics_only": True,
            "evaluation_scene_labels_used_for_calibration": False,
        },
        "runtime": runtime_metadata(device=torch_device),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def _phase_histogram(phases: list[int]) -> dict[str, int]:
    return {
        ExpertPhase(phase).name.lower(): phases.count(phase)
        for phase in sorted(set(phases))
    }


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _task_from_instruction(instruction: str) -> tuple[str, str]:
    words = set(instruction.lower().split())
    colors = [color for color in COLORS if color in words]
    sides = [side for side in SIDES if side in words]
    if len(colors) != 1 or len(sides) != 1:
        raise ValueError(f"cannot parse task from instruction: {instruction!r}")
    return colors[0], sides[0]


def _task_success(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for color in COLORS:
        for side in SIDES:
            selected = [
                record
                for record in records
                if record["target_color"] == color and record["goal_side"] == side
            ]
            if selected:
                successes = sum(record["success"] for record in selected)
                result[f"{color}->{side}"] = {
                    "episodes": len(selected),
                    "successes": successes,
                    "success_rate": successes / len(selected),
                }
    return result
