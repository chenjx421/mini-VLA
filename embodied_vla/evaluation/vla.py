from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import imageio.v2 as imageio
import numpy as np
import torch
from numpy.typing import NDArray

from embodied_vla.envs import SOArmEnvConfig, SOArmPickPlaceEnv
from embodied_vla.envs.so_arm_pick_place import COLORS, SIDES
from embodied_vla.experts import ExpertPhase, PickPlaceExpert
from embodied_vla.experts.pick_place import APPROACH_HEIGHT_OFFSET
from embodied_vla.metrics import wilson_score_interval
from embodied_vla.models import TinyVLA, TinyVLAConfig, TinyVLAOutput
from embodied_vla.proprioception import uses_end_effector_position
from embodied_vla.reproducibility import runtime_metadata
from embodied_vla.training.run_guard import claim_run_directory
from embodied_vla.visualization import make_vla_attention_panel


@dataclass(frozen=True)
class VLAEvalConfig:
    episodes: int = 20
    seed: int = 20_000
    execution_horizon: int = 1
    video_episodes: int = 3
    max_episode_steps: int = 300
    grasp_mode: Literal["contact", "contact_assisted"] = "contact_assisted"
    domain_randomization: bool = False
    balanced_tasks: bool = True
    torch_threads: int = 1
    cartesian_action_gain: float = 1.0

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")
        if self.video_episodes < 0:
            raise ValueError("video_episodes cannot be negative")
        if self.torch_threads <= 0:
            raise ValueError("torch_threads must be positive")
        if self.cartesian_action_gain <= 0:
            raise ValueError("cartesian_action_gain must be positive")


def load_tiny_vla(
    checkpoint_path: Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[TinyVLA, TinyVLAConfig]:
    torch_device = torch.device(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch_device,
        weights_only=True,
    )
    model_config = TinyVLAConfig(**checkpoint["model_config"])
    model = TinyVLA(model_config).to(torch_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, model_config


def evaluate_tiny_vla(
    checkpoint_path: Path,
    *,
    output_dir: Path,
    eval_config: VLAEvalConfig,
    device: str = "cpu",
) -> dict[str, Any]:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(eval_config.torch_threads)
    try:
        with claim_run_directory(output_dir):
            return _evaluate_tiny_vla_in_claimed_directory(
                checkpoint_path,
                output_dir=output_dir,
                eval_config=eval_config,
                device=device,
            )
    finally:
        torch.set_num_threads(previous_threads)


def _evaluate_tiny_vla_in_claimed_directory(
    checkpoint_path: Path,
    *,
    output_dir: Path,
    eval_config: VLAEvalConfig,
    device: str,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    model, model_config = load_tiny_vla(checkpoint_path, device=torch_device)
    if eval_config.execution_horizon > model_config.action_horizon:
        raise ValueError("execution_horizon cannot exceed the model action horizon")

    env_config = SOArmEnvConfig(
        observation_mode="multimodal",
        task_level="pick_place",
        grasp_mode=eval_config.grasp_mode,
        image_size=model_config.image_size,
        max_episode_steps=eval_config.max_episode_steps,
        domain_randomization=eval_config.domain_randomization,
        include_end_effector_position_in_proprio=uses_end_effector_position(
            model_config.proprio_dim
        ),
    )
    env = SOArmPickPlaceEnv(env_config)
    diagnostic_expert = PickPlaceExpert(env_config)
    episode_records: list[dict[str, Any]] = []
    inference_latencies_ms: list[float] = []
    metrics_path = output_dir / "episodes.jsonl"
    start_time = time.perf_counter()
    try:
        for episode_index in range(eval_config.episodes):
            episode_seed = eval_config.seed + episode_index
            reset_options = None
            if eval_config.balanced_tasks:
                task_index = episode_index % (len(COLORS) * len(SIDES))
                reset_options = {
                    "target_color": COLORS[task_index // len(SIDES)],
                    "goal_side": SIDES[task_index % len(SIDES)],
                }
            observation, info = env.reset(
                seed=episode_seed,
                options=reset_options,
            )
            diagnostic_expert.reset()
            episode_return = 0.0
            grounding_errors: list[float] = []
            world_grounding_errors_m: list[float] = []
            on_policy_expert_errors: list[float] = []
            executed_expert_errors: list[float] = []
            predicted_phases: list[int] = []
            trace_records: list[dict[str, Any]] = []
            frames: list[NDArray[np.uint8]] = []
            minimum_reach_distance = float("inf")
            minimum_approach_waypoint_distance = float("inf")
            minimum_approach_waypoint_xy_error = float("inf")
            minimum_approach_waypoint_z_error = float("inf")
            bilateral_contact_steps = 0
            ever_grasped = False
            ever_lifted = False
            first_phase_mismatch_step: int | None = None
            diagnostic_phases: list[int] = []
            terminated = False
            truncated = False

            while not (terminated or truncated):
                minimum_reach_distance = min(
                    minimum_reach_distance,
                    float(info["reach_distance"]),
                )
                bilateral_contact_steps += int(bool(info["bilateral_contact"]))
                ever_grasped = ever_grasped or bool(info["has_grasped"])
                ever_lifted = ever_lifted or bool(info["has_lifted"])
                inference_start = time.perf_counter()
                output = predict_tiny_vla(model, observation, torch_device)
                inference_latencies_ms.append(
                    (time.perf_counter() - inference_start) * 1_000.0
                )
                predicted_phase = int(output.phase_logits.argmax(dim=-1).item())
                predicted_phases.append(predicted_phase)
                ground_truth, pixel_valid = _grounding_labels(env, info)
                predicted_coordinates = (
                    output.grounding_coordinates.squeeze(0).detach().cpu().numpy()
                )
                if pixel_valid.any():
                    errors = np.linalg.norm(predicted_coordinates - ground_truth, axis=-1)
                    grounding_errors.extend(errors[pixel_valid].tolist())
                predicted_world_positions = None
                if output.grounding_world_positions is not None:
                    predicted_world_positions = (
                        output.grounding_world_positions.squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                        * 0.5
                    )
                    ground_truth_world_positions = np.stack(
                        (
                            np.asarray(info["target_position"], dtype=np.float32),
                            np.asarray(info["goal_position"], dtype=np.float32),
                        )
                    )
                    world_errors = np.linalg.norm(
                        predicted_world_positions - ground_truth_world_positions,
                        axis=-1,
                    )
                    world_grounding_errors_m.extend(
                        world_errors[pixel_valid].tolist()
                    )

                if episode_index < eval_config.video_episodes:
                    frames.append(
                        make_vla_attention_panel(
                            observation["rgb"],
                            output.grounding_heatmaps.squeeze(0).detach().cpu().numpy(),
                            instruction=str(info["instruction"]),
                            predicted_phase=predicted_phase,
                            step=int(info["step"]),
                            predicted_coordinates=predicted_coordinates,
                            ground_truth_coordinates=ground_truth,
                            pixel_valid=pixel_valid,
                        )
                    )

                action_chunk = output.action_chunk.squeeze(0).detach().cpu().numpy()
                executed_action_chunk = _apply_cartesian_action_gain(
                    action_chunk,
                    eval_config.cartesian_action_gain,
                )
                diagnostic_phase = int(diagnostic_expert.phase)
                diagnostic_phases.append(diagnostic_phase)
                if diagnostic_phase == int(ExpertPhase.APPROACH):
                    end_effector = np.asarray(
                        info["end_effector_position"],
                        dtype=np.float64,
                    )
                    approach_waypoint = np.asarray(
                        info["target_position"],
                        dtype=np.float64,
                    ) + np.array([0.0, 0.0, APPROACH_HEIGHT_OFFSET])
                    waypoint_error = approach_waypoint - end_effector
                    minimum_approach_waypoint_distance = min(
                        minimum_approach_waypoint_distance,
                        float(np.linalg.norm(waypoint_error)),
                    )
                    minimum_approach_waypoint_xy_error = min(
                        minimum_approach_waypoint_xy_error,
                        float(np.linalg.norm(waypoint_error[:2])),
                    )
                    minimum_approach_waypoint_z_error = min(
                        minimum_approach_waypoint_z_error,
                        abs(float(waypoint_error[2])),
                    )
                expert_action = diagnostic_expert.act(info)
                if (
                    first_phase_mismatch_step is None
                    and predicted_phase != diagnostic_phase
                ):
                    first_phase_mismatch_step = int(info["step"])
                on_policy_expert_mae = float(
                    np.mean(np.abs(action_chunk[0] - expert_action))
                )
                on_policy_expert_errors.append(on_policy_expert_mae)
                executed_expert_mae = float(
                    np.mean(np.abs(executed_action_chunk[0] - expert_action))
                )
                executed_expert_errors.append(executed_expert_mae)
                trace_records.append(
                    {
                        "step": int(info["step"]),
                        "predicted_phase": ExpertPhase(predicted_phase).name.lower(),
                        "diagnostic_expert_phase": ExpertPhase(
                            diagnostic_phase
                        ).name.lower(),
                        "model_action": action_chunk[0].tolist(),
                        "executed_action": executed_action_chunk[0].tolist(),
                        "diagnostic_expert_action": expert_action.tolist(),
                        "on_policy_expert_mae": on_policy_expert_mae,
                        "executed_on_policy_expert_mae": executed_expert_mae,
                        "end_effector_position": info[
                            "end_effector_position"
                        ].tolist(),
                        "target_position": info["target_position"].tolist(),
                        "goal_position": info["goal_position"].tolist(),
                        "predicted_target_world_position": (
                            predicted_world_positions[0].tolist()
                            if predicted_world_positions is not None
                            else None
                        ),
                        "predicted_goal_world_position": (
                            predicted_world_positions[1].tolist()
                            if predicted_world_positions is not None
                            else None
                        ),
                        "reach_distance": float(info["reach_distance"]),
                        "goal_distance": float(info["goal_distance"]),
                        "jaw_joint_position": float(env.joint_positions[-1]),
                        "bilateral_contact": bool(info["bilateral_contact"]),
                        "has_grasped": bool(info["has_grasped"]),
                        "has_lifted": bool(info["has_lifted"]),
                    }
                )
                for action in executed_action_chunk[: eval_config.execution_horizon]:
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_return += reward
                    if terminated or truncated:
                        break

            record = {
                "episode": episode_index,
                "seed": episode_seed,
                "success": bool(info["success"]),
                "return": float(episode_return),
                "length": int(info["step"]),
                "termination_reason": info["termination_reason"],
                "target_color": info["target_color"],
                "goal_side": info["goal_side"],
                "mean_grounding_l2": (
                    float(np.mean(grounding_errors)) if grounding_errors else None
                ),
                "mean_world_grounding_l2_m": (
                    float(np.mean(world_grounding_errors_m))
                    if world_grounding_errors_m
                    else None
                ),
                "mean_on_policy_expert_mae": float(
                    np.mean(on_policy_expert_errors)
                ),
                "mean_executed_on_policy_expert_mae": float(
                    np.mean(executed_expert_errors)
                ),
                "minimum_reach_distance": minimum_reach_distance,
                "minimum_approach_waypoint_distance": (
                    minimum_approach_waypoint_distance
                ),
                "minimum_approach_waypoint_xy_error": (
                    minimum_approach_waypoint_xy_error
                ),
                "minimum_approach_waypoint_z_error": (
                    minimum_approach_waypoint_z_error
                ),
                "diagnostic_expert_left_approach": any(
                    phase != int(ExpertPhase.APPROACH)
                    for phase in diagnostic_phases
                ),
                "bilateral_contact_steps": bilateral_contact_steps,
                "ever_grasped": ever_grasped,
                "ever_lifted": ever_lifted,
                "first_phase_mismatch_step": first_phase_mismatch_step,
                "predicted_phase_histogram": {
                    ExpertPhase(phase).name.lower(): predicted_phases.count(phase)
                    for phase in sorted(set(predicted_phases))
                },
                "diagnostic_expert_phase_histogram": {
                    ExpertPhase(phase).name.lower(): diagnostic_phases.count(phase)
                    for phase in sorted(set(diagnostic_phases))
                },
            }
            episode_records.append(record)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            trace_path = output_dir / f"episode_{episode_index:03d}_trace.jsonl"
            with trace_path.open("w", encoding="utf-8") as handle:
                for trace_record in trace_records:
                    handle.write(json.dumps(trace_record, ensure_ascii=True) + "\n")

            if frames:
                video_path = output_dir / f"episode_{episode_index:03d}.gif"
                imageio.mimsave(video_path, frames, duration=0.10, loop=0)
                imageio.imwrite(
                    output_dir / f"episode_{episode_index:03d}_attention.png",
                    frames[len(frames) // 2],
                )
            print(
                f"episode={episode_index:03d} seed={episode_seed} "
                f"success={bool(info['success'])} return={episode_return:.3f} "
                f"steps={int(info['step'])}"
            )
    finally:
        env.close()

    grounding_values = [
        record["mean_grounding_l2"]
        for record in episode_records
        if record["mean_grounding_l2"] is not None
    ]
    expert_error_values = [
        record["mean_on_policy_expert_mae"] for record in episode_records
    ]
    world_grounding_values = [
        record["mean_world_grounding_l2_m"]
        for record in episode_records
        if record["mean_world_grounding_l2_m"] is not None
    ]
    executed_expert_error_values = [
        record["mean_executed_on_policy_expert_mae"] for record in episode_records
    ]
    successes = sum(record["success"] for record in episode_records)
    confidence_low, confidence_high = wilson_score_interval(
        successes,
        eval_config.episodes,
    )
    task_success = {}
    for color in COLORS:
        for side in SIDES:
            task_records = [
                record
                for record in episode_records
                if record["target_color"] == color and record["goal_side"] == side
            ]
            if task_records:
                task_success[f"{color}->{side}"] = {
                    "episodes": len(task_records),
                    "successes": sum(record["success"] for record in task_records),
                    "success_rate": float(
                        np.mean([record["success"] for record in task_records])
                    ),
                }
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "episodes": eval_config.episodes,
        "successes": successes,
        "success_rate": successes / eval_config.episodes,
        "success_wilson_95": [confidence_low, confidence_high],
        "task_success": task_success,
        "mean_return": float(np.mean([record["return"] for record in episode_records])),
        "mean_length": float(np.mean([record["length"] for record in episode_records])),
        "mean_grounding_l2": (
            float(np.mean(grounding_values)) if grounding_values else None
        ),
        "mean_world_grounding_l2_m": (
            float(np.mean(world_grounding_values))
            if world_grounding_values
            else None
        ),
        "mean_on_policy_expert_mae": float(np.mean(expert_error_values)),
        "mean_executed_on_policy_expert_mae": float(
            np.mean(executed_expert_error_values)
        ),
        "mean_minimum_reach_distance": float(
            np.mean(
                [record["minimum_reach_distance"] for record in episode_records]
            )
        ),
        "mean_minimum_approach_waypoint_distance": float(
            np.mean(
                [
                    record["minimum_approach_waypoint_distance"]
                    for record in episode_records
                ]
            )
        ),
        "mean_minimum_approach_waypoint_xy_error": float(
            np.mean(
                [
                    record["minimum_approach_waypoint_xy_error"]
                    for record in episode_records
                ]
            )
        ),
        "mean_minimum_approach_waypoint_z_error": float(
            np.mean(
                [
                    record["minimum_approach_waypoint_z_error"]
                    for record in episode_records
                ]
            )
        ),
        "episodes_where_diagnostic_expert_left_approach": sum(
            record["diagnostic_expert_left_approach"]
            for record in episode_records
        ),
        "episodes_with_bilateral_contact": sum(
            record["bilateral_contact_steps"] > 0 for record in episode_records
        ),
        "episodes_ever_grasped": sum(
            record["ever_grasped"] for record in episode_records
        ),
        "episodes_ever_lifted": sum(
            record["ever_lifted"] for record in episode_records
        ),
        "elapsed_seconds": time.perf_counter() - start_time,
        "inference_latency_ms": {
            "mean": float(np.mean(inference_latencies_ms)),
            "p50": float(np.percentile(inference_latencies_ms, 50)),
            "p95": float(np.percentile(inference_latencies_ms, 95)),
            "samples": len(inference_latencies_ms),
        },
        "model": asdict(model_config),
        "evaluation": asdict(eval_config),
        "runtime": runtime_metadata(device=torch_device),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def predict_tiny_vla(
    model: TinyVLA,
    observation: dict[str, NDArray[Any]],
    device: torch.device,
) -> TinyVLAOutput:
    rgb = torch.as_tensor(
        observation["rgb"],
        dtype=torch.float32,
        device=device,
    )
    rgb = rgb.permute(2, 0, 1).unsqueeze(0) / 255.0
    proprio = torch.as_tensor(
        observation["proprio"],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    language = torch.as_tensor(
        observation["language"],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    language_mask = torch.as_tensor(
        observation["language_mask"],
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)
    with torch.inference_mode():
        return model(rgb, proprio, language, language_mask)


def _grounding_labels(
    env: SOArmPickPlaceEnv,
    info: dict[str, Any],
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    target_pixel, target_valid = env.project_world_point(info["target_position"])
    goal_pixel, goal_valid = env.project_world_point(info["goal_position"])
    coordinates = np.stack((target_pixel, goal_pixel)).astype(np.float32)
    valid = np.asarray((target_valid, goal_valid), dtype=np.bool_)
    return coordinates, valid


def _apply_cartesian_action_gain(
    action_chunk: NDArray[np.floating],
    gain: float,
) -> NDArray[np.float32]:
    executed = np.asarray(action_chunk, dtype=np.float32).copy()
    executed[:, :3] = np.clip(executed[:, :3] * gain, -1.0, 1.0)
    return executed
