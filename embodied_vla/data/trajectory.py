from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from embodied_vla.envs import SOArmEnvConfig, SOArmPickPlaceEnv
from embodied_vla.experts import PickPlaceExpert
from embodied_vla.proprioception import (
    END_EFFECTOR_STATE_SLICE,
    GOAL_WORLD_STATE_SLICE,
    JOINT_PROPRIO_DIM,
    TARGET_WORLD_STATE_SLICE,
    assemble_model_proprio,
)
from embodied_vla.training.run_guard import claim_run_directory

SCHEMA_VERSION = 1
TASK_COLORS = ("red", "green", "blue")
TASK_SIDES = ("left", "right")


def collect_expert_dataset(
    output_dir: Path,
    *,
    episodes: int,
    seed: int,
    image_size: int = 64,
    max_attempt_multiplier: int = 3,
    balanced_tasks: bool = True,
    domain_randomization: bool = False,
) -> dict[str, Any]:
    """Collect successful contact-assisted demonstrations as NPZ episodes."""

    with claim_run_directory(output_dir):
        return _collect_expert_dataset_in_claimed_directory(
            output_dir,
            episodes=episodes,
            seed=seed,
            image_size=image_size,
            max_attempt_multiplier=max_attempt_multiplier,
            balanced_tasks=balanced_tasks,
            domain_randomization=domain_randomization,
        )


def _collect_expert_dataset_in_claimed_directory(
    output_dir: Path,
    *,
    episodes: int,
    seed: int,
    image_size: int,
    max_attempt_multiplier: int,
    balanced_tasks: bool,
    domain_randomization: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    config = SOArmEnvConfig(
        observation_mode="multimodal",
        task_level="pick_place",
        grasp_mode="contact_assisted",
        image_size=image_size,
        max_episode_steps=300,
        domain_randomization=domain_randomization,
    )
    env = SOArmPickPlaceEnv(config)
    expert = PickPlaceExpert(config)
    accepted = 0
    attempted = 0
    lengths: list[int] = []
    records: list[dict[str, Any]] = []
    max_attempts = episodes * max_attempt_multiplier
    try:
        while accepted < episodes and attempted < max_attempts:
            episode_seed = seed + attempted
            reset_options = None
            if balanced_tasks:
                task_index = accepted % (len(TASK_COLORS) * len(TASK_SIDES))
                reset_options = {
                    "target_color": TASK_COLORS[task_index // len(TASK_SIDES)],
                    "goal_side": TASK_SIDES[task_index % len(TASK_SIDES)],
                }
            observation, info = env.reset(seed=episode_seed, options=reset_options)
            expert.reset()
            buffers: dict[str, list[Any]] = {
                "rgb": [],
                "proprio": [],
                "state": [],
                "action": [],
                "phase": [],
                "reward": [],
                "terminated": [],
                "truncated": [],
                "target_pixel": [],
                "goal_pixel": [],
                "pixel_valid": [],
            }
            for _ in range(config.max_episode_steps):
                phase = int(expert.phase)
                action = expert.act(info)
                target_pixel, target_visible = env.project_world_point(
                    info["target_position"]
                )
                goal_pixel, goal_visible = env.project_world_point(info["goal_position"])
                buffers["rgb"].append(observation["rgb"])
                buffers["proprio"].append(observation["proprio"])
                buffers["state"].append(env.privileged_state())
                buffers["action"].append(action)
                buffers["phase"].append(phase)
                buffers["target_pixel"].append(target_pixel)
                buffers["goal_pixel"].append(goal_pixel)
                buffers["pixel_valid"].append([target_visible, goal_visible])

                observation, reward, terminated, truncated, info = env.step(action)
                buffers["reward"].append(reward)
                buffers["terminated"].append(terminated)
                buffers["truncated"].append(truncated)
                if terminated or truncated:
                    break

            attempted += 1
            if not info["success"]:
                print(
                    f"reject seed={episode_seed} phase={expert.phase.name} "
                    f"steps={info['step']} reason={info['termination_reason']}"
                )
                continue

            episode_path = episode_dir / f"episode_{accepted:06d}.npz"
            np.savez_compressed(
                episode_path,
                rgb=np.asarray(buffers["rgb"], dtype=np.uint8),
                proprio=np.asarray(buffers["proprio"], dtype=np.float32),
                state=np.asarray(buffers["state"], dtype=np.float32),
                language=observation["language"].astype(np.int64),
                language_mask=observation["language_mask"].astype(np.int8),
                action=np.asarray(buffers["action"], dtype=np.float32),
                phase=np.asarray(buffers["phase"], dtype=np.int64),
                reward=np.asarray(buffers["reward"], dtype=np.float32),
                terminated=np.asarray(buffers["terminated"], dtype=np.bool_),
                truncated=np.asarray(buffers["truncated"], dtype=np.bool_),
                target_pixel=np.asarray(buffers["target_pixel"], dtype=np.float32),
                goal_pixel=np.asarray(buffers["goal_pixel"], dtype=np.float32),
                pixel_valid=np.asarray(buffers["pixel_valid"], dtype=np.bool_),
            )
            record = {
                "episode": accepted,
                "seed": episode_seed,
                "path": str(episode_path.relative_to(output_dir)).replace("\\", "/"),
                "length": len(buffers["action"]),
                "instruction": info["instruction"],
                "target_color": info["target_color"],
                "goal_side": info["goal_side"],
                "success": True,
            }
            records.append(record)
            lengths.append(record["length"])
            accepted += 1
            print(
                f"accept episode={accepted:04d}/{episodes} seed={episode_seed} "
                f"length={record['length']} task={record['target_color']}->{record['goal_side']}"
            )
    finally:
        env.close()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "episodes": accepted,
        "attempted": attempted,
        "seed": seed,
        "balanced_tasks": balanced_tasks,
        "domain_randomization": domain_randomization,
        "environment": asdict(config),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "task_counts": {
            f"{color}->{side}": sum(
                record["target_color"] == color and record["goal_side"] == side
                for record in records
            )
            for color in TASK_COLORS
            for side in TASK_SIDES
        },
        "records": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    if accepted < episodes:
        raise RuntimeError(f"collected only {accepted}/{episodes} episodes")
    return manifest


class ActionChunkDataset(Dataset[dict[str, torch.Tensor]]):
    """Lazy per-episode dataset with leakage-safe episode splits."""

    def __init__(
        self,
        root: Path,
        *,
        action_horizon: int = 8,
        split: str = "train",
        validation_fraction: float = 0.15,
        split_seed: int = 2026,
        episode_cache_size: int | None = None,
        proprio_dim: int = JOINT_PROPRIO_DIM,
    ) -> None:
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        if episode_cache_size is not None and episode_cache_size <= 0:
            raise ValueError("episode_cache_size must be positive or None")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported trajectory schema")
        records = list(manifest["records"])
        rng = np.random.default_rng(split_seed)
        train_indices, validation_indices, split_strategy = _episode_split_indices(
            records,
            validation_fraction=validation_fraction,
            rng=rng,
        )
        selected_indices = train_indices if split == "train" else validation_indices
        self.root = root
        self.action_horizon = action_horizon
        self.split_strategy = split_strategy
        self.records = [records[int(index)] for index in selected_indices]
        self.episode_cache_size = episode_cache_size
        self.proprio_dim = proprio_dim
        self._episode_cache: dict[int, dict[str, np.ndarray]] = {}
        self._indices: list[tuple[int, int]] = []
        for episode_index, record in enumerate(self.records):
            self._indices.extend(
                (episode_index, time_index) for time_index in range(record["length"])
            )

    def __len__(self) -> int:
        return len(self._indices)

    @property
    def sample_time_indices(self) -> list[int]:
        return [time_index for _, time_index in self._indices]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, time_index = self._indices[index]
        episode = self._load_episode(episode_index)
        action_count = episode["action"].shape[0]
        chunk_end = min(action_count, time_index + self.action_horizon)
        valid_count = chunk_end - time_index
        action_chunk = np.zeros(
            (self.action_horizon, episode["action"].shape[1]),
            dtype=np.float32,
        )
        action_chunk[:valid_count] = episode["action"][time_index:chunk_end]
        if valid_count < self.action_horizon:
            action_chunk[valid_count:] = episode["action"][chunk_end - 1]
        action_mask = np.zeros(self.action_horizon, dtype=np.bool_)
        action_mask[:valid_count] = True

        image = episode["rgb"][time_index].astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return {
            "rgb": torch.from_numpy(image),
            "proprio": torch.from_numpy(
                assemble_model_proprio(
                    episode["proprio"][time_index],
                    expected_dim=self.proprio_dim,
                    normalized_end_effector_position=episode["state"][
                        time_index,
                        END_EFFECTOR_STATE_SLICE,
                    ],
                )
            ),
            "language": torch.from_numpy(episode["language"]),
            "language_mask": torch.from_numpy(episode["language_mask"].astype(np.bool_)),
            "action_chunk": torch.from_numpy(action_chunk),
            "action_mask": torch.from_numpy(action_mask),
            "phase": torch.tensor(episode["phase"][time_index], dtype=torch.long),
            "target_pixel": torch.from_numpy(episode["target_pixel"][time_index]),
            "goal_pixel": torch.from_numpy(episode["goal_pixel"][time_index]),
            "pixel_valid": torch.from_numpy(episode["pixel_valid"][time_index]),
            "target_world": torch.from_numpy(
                episode["state"][time_index, TARGET_WORLD_STATE_SLICE]
            ),
            "goal_world": torch.from_numpy(
                episode["state"][time_index, GOAL_WORLD_STATE_SLICE]
            ),
            "privileged_state": torch.from_numpy(episode["state"][time_index]),
            "episode_index": torch.tensor(episode_index, dtype=torch.long),
            "time_index": torch.tensor(time_index, dtype=torch.long),
            "episode_length": torch.tensor(action_count, dtype=torch.long),
        }

    def _load_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        if episode_index not in self._episode_cache:
            path = self.root / self.records[episode_index]["path"]
            with np.load(path) as archive:
                self._episode_cache[episode_index] = {
                    key: archive[key] for key in archive.files
                }
            if (
                self.episode_cache_size is not None
                and len(self._episode_cache) > self.episode_cache_size
            ):
                oldest_key = next(iter(self._episode_cache))
                if oldest_key != episode_index:
                    self._episode_cache.pop(oldest_key)
        return self._episode_cache[episode_index]


def _episode_split_indices(
    records: list[dict[str, Any]],
    *,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[int], str]:
    if len(records) < 2:
        raise ValueError("at least two episodes are required for a train/validation split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie between zero and one")

    groups: dict[tuple[str, str], list[int]] = {}
    task_labels_available = all(
        "target_color" in record and "goal_side" in record for record in records
    )
    if task_labels_available:
        for index, record in enumerate(records):
            key = (str(record["target_color"]), str(record["goal_side"]))
            groups.setdefault(key, []).append(index)

    can_stratify = len(groups) > 1 and all(len(indices) >= 2 for indices in groups.values())
    if can_stratify:
        train_indices: list[int] = []
        validation_indices: list[int] = []
        for key in sorted(groups):
            permutation = rng.permutation(groups[key]).tolist()
            validation_count = min(
                len(permutation) - 1,
                max(1, round(len(permutation) * validation_fraction)),
            )
            validation_indices.extend(permutation[:validation_count])
            train_indices.extend(permutation[validation_count:])
        rng.shuffle(train_indices)
        rng.shuffle(validation_indices)
        return train_indices, validation_indices, "stratified_task_episode"

    permutation = rng.permutation(len(records)).tolist()
    validation_count = min(
        len(permutation) - 1,
        max(1, round(len(permutation) * validation_fraction)),
    )
    return (
        permutation[validation_count:],
        permutation[:validation_count],
        "random_episode",
    )
