from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from embodied_vla.envs import SOArmEnvConfig, SOArmPickPlaceEnv
from embodied_vla.envs.so_arm_pick_place import COLORS, SIDES
from embodied_vla.evaluation.vla import load_tiny_vla, predict_tiny_vla
from embodied_vla.training.run_guard import claim_run_directory
from embodied_vla.visualization import make_vla_attention_panel


@dataclass(frozen=True)
class CounterfactualEvalConfig:
    scenes: int = 10
    seed: int = 30_000
    visualized_scenes: int = 2
    domain_randomization: bool = False
    torch_threads: int = 1

    def __post_init__(self) -> None:
        if self.scenes <= 0:
            raise ValueError("scenes must be positive")
        if self.visualized_scenes < 0:
            raise ValueError("visualized_scenes cannot be negative")
        if self.torch_threads <= 0:
            raise ValueError("torch_threads must be positive")


def evaluate_language_counterfactuals(
    checkpoint_path: Path,
    *,
    output_dir: Path,
    eval_config: CounterfactualEvalConfig,
    device: str = "cpu",
) -> dict[str, Any]:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(eval_config.torch_threads)
    try:
        with claim_run_directory(output_dir):
            return _evaluate_language_counterfactuals_in_claimed_directory(
                checkpoint_path,
                output_dir=output_dir,
                eval_config=eval_config,
                device=device,
            )
    finally:
        torch.set_num_threads(previous_threads)


def _evaluate_language_counterfactuals_in_claimed_directory(
    checkpoint_path: Path,
    *,
    output_dir: Path,
    eval_config: CounterfactualEvalConfig,
    device: str,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    model, model_config = load_tiny_vla(checkpoint_path, device=torch_device)
    env = SOArmPickPlaceEnv(
        SOArmEnvConfig(
            observation_mode="multimodal",
            task_level="pick_place",
            grasp_mode="contact_assisted",
            image_size=model_config.image_size,
            domain_randomization=eval_config.domain_randomization,
        )
    )
    scene_records: list[dict[str, Any]] = []
    records_path = output_dir / "counterfactuals.jsonl"
    try:
        for scene_index in range(eval_config.scenes):
            scene_seed = eval_config.seed + scene_index
            predictions: dict[tuple[str, str], dict[str, Any]] = {}
            reference_rgb: NDArray[np.uint8] | None = None
            reference_proprio: NDArray[np.float32] | None = None
            visual_panels: list[NDArray[np.uint8]] = []

            for color in COLORS:
                for side in SIDES:
                    observation, info = env.reset(
                        seed=scene_seed,
                        options={"target_color": color, "goal_side": side},
                    )
                    if reference_rgb is None:
                        reference_rgb = observation["rgb"].copy()
                        reference_proprio = observation["proprio"].copy()
                    elif not np.array_equal(observation["rgb"], reference_rgb):
                        raise RuntimeError("counterfactual task changed the RGB scene")
                    elif not np.array_equal(observation["proprio"], reference_proprio):
                        raise RuntimeError("counterfactual task changed proprioception")

                    torch.manual_seed(scene_seed)
                    output = predict_tiny_vla(model, observation, torch_device)
                    target_pixel, target_valid = env.project_world_point(
                        info["target_position"]
                    )
                    goal_pixel, goal_valid = env.project_world_point(
                        info["goal_position"]
                    )
                    ground_truth = np.stack((target_pixel, goal_pixel)).astype(
                        np.float32
                    )
                    valid = np.asarray((target_valid, goal_valid), dtype=np.bool_)
                    coordinates = (
                        output.grounding_coordinates.squeeze(0).detach().cpu().numpy()
                    )
                    action_chunk = output.action_chunk.squeeze(0).detach().cpu().numpy()
                    predictions[(color, side)] = {
                        "instruction": str(info["instruction"]),
                        "coordinates": coordinates,
                        "ground_truth": ground_truth,
                        "valid": valid,
                        "action_chunk": action_chunk,
                    }
                    if scene_index < eval_config.visualized_scenes:
                        visual_panels.append(
                            make_vla_attention_panel(
                                observation["rgb"],
                                output.grounding_heatmaps.squeeze(0)
                                .detach()
                                .cpu()
                                .numpy(),
                                instruction=str(info["instruction"]),
                                predicted_phase=int(
                                    output.phase_logits.argmax(dim=-1).item()
                                ),
                                step=0,
                                predicted_coordinates=coordinates,
                                ground_truth_coordinates=ground_truth,
                                pixel_valid=valid,
                                scale=2,
                            )
                        )

            record = _score_scene(scene_index, scene_seed, predictions)
            scene_records.append(record)
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            if visual_panels:
                _save_panel_grid(
                    visual_panels,
                    output_dir / f"scene_{scene_index:03d}_counterfactual.png",
                )
            print(
                f"scene={scene_index:03d} target_acc={record['target_accuracy']:.1%} "
                f"goal_acc={record['goal_accuracy']:.1%} "
                f"color_action_delta={record['mean_color_action_rms']:.4f} "
                f"side_action_delta={record['mean_side_action_rms']:.4f}"
            )
    finally:
        env.close()

    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "scenes": eval_config.scenes,
        "task_variants_per_scene": len(COLORS) * len(SIDES),
        "target_grounding_accuracy": float(
            np.mean([record["target_accuracy"] for record in scene_records])
        ),
        "goal_grounding_accuracy": float(
            np.mean([record["goal_accuracy"] for record in scene_records])
        ),
        "mean_color_action_rms": float(
            np.mean([record["mean_color_action_rms"] for record in scene_records])
        ),
        "mean_side_action_rms": float(
            np.mean([record["mean_side_action_rms"] for record in scene_records])
        ),
        "evaluation": asdict(eval_config),
        "controlled_variables": [
            "rgb",
            "proprioception",
            "robot_state",
            "object_positions",
            "flow_sampling_noise",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def _score_scene(
    scene_index: int,
    scene_seed: int,
    predictions: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    target_candidates = np.stack(
        [predictions[(color, SIDES[0])]["ground_truth"][0] for color in COLORS]
    )
    goal_candidates = np.stack(
        [predictions[(COLORS[0], side)]["ground_truth"][1] for side in SIDES]
    )
    target_correct: list[bool] = []
    goal_correct: list[bool] = []
    variant_records: list[dict[str, Any]] = []
    for (color, side), prediction in predictions.items():
        coordinates = prediction["coordinates"]
        target_index = int(
            np.argmin(np.linalg.norm(target_candidates - coordinates[0], axis=-1))
        )
        goal_index = int(
            np.argmin(np.linalg.norm(goal_candidates - coordinates[1], axis=-1))
        )
        target_is_correct = COLORS[target_index] == color
        goal_is_correct = SIDES[goal_index] == side
        target_correct.append(target_is_correct)
        goal_correct.append(goal_is_correct)
        variant_records.append(
            {
                "target_color": color,
                "goal_side": side,
                "instruction": prediction["instruction"],
                "predicted_target": coordinates[0].tolist(),
                "predicted_goal": coordinates[1].tolist(),
                "ground_truth_target": prediction["ground_truth"][0].tolist(),
                "ground_truth_goal": prediction["ground_truth"][1].tolist(),
                "target_nearest_color": COLORS[target_index],
                "goal_nearest_side": SIDES[goal_index],
                "first_action": prediction["action_chunk"][0].tolist(),
            }
        )

    color_action_distances = []
    for side in SIDES:
        chunks = [predictions[(color, side)]["action_chunk"] for color in COLORS]
        color_action_distances.extend(
            _root_mean_square(left - right) for left, right in combinations(chunks, 2)
        )
    side_action_distances = [
        _root_mean_square(
            predictions[(color, SIDES[0])]["action_chunk"]
            - predictions[(color, SIDES[1])]["action_chunk"]
        )
        for color in COLORS
    ]
    return {
        "scene": scene_index,
        "seed": scene_seed,
        "rgb_invariant": True,
        "proprio_invariant": True,
        "target_accuracy": float(np.mean(target_correct)),
        "goal_accuracy": float(np.mean(goal_correct)),
        "mean_color_action_rms": float(np.mean(color_action_distances)),
        "mean_side_action_rms": float(np.mean(side_action_distances)),
        "variants": variant_records,
    }


def _root_mean_square(values: NDArray[np.floating]) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _save_panel_grid(panels: list[NDArray[np.uint8]], path: Path) -> None:
    images = [Image.fromarray(panel, mode="RGB") for panel in panels]
    columns = 2
    rows = (len(images) + columns - 1) // columns
    tile_width = max(image.width for image in images)
    tile_height = max(image.height for image in images)
    canvas = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height),
        color=(22, 24, 28),
    )
    for index, image in enumerate(images):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        canvas.paste(image, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
