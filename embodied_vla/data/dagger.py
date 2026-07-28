from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from embodied_vla.envs import SOArmEnvConfig, SOArmPickPlaceEnv
from embodied_vla.envs.so_arm_pick_place import COLORS, SIDES
from embodied_vla.experts import PickPlaceExpert
from embodied_vla.models import TinyVLA, TinyVLAConfig
from embodied_vla.proprioception import (
    END_EFFECTOR_STATE_SLICE,
    GOAL_WORLD_STATE_SLICE,
    JOINT_PROPRIO_DIM,
    TARGET_WORLD_STATE_SLICE,
    assemble_model_proprio,
    uses_end_effector_position,
)
from embodied_vla.reproducibility import runtime_metadata
from embodied_vla.training.run_guard import claim_run_directory

DAGGER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DAggerCollectConfig:
    episodes: int = 24
    seed: int = 30_000
    max_steps: int = 180
    expert_mixing_probability: float = 0.35
    balanced_tasks: bool = True
    domain_randomization: bool = True
    video_episodes: int = 1
    torch_threads: int = 1

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not 0.0 <= self.expert_mixing_probability <= 1.0:
            raise ValueError("expert_mixing_probability must lie in [0, 1]")
        if self.video_episodes < 0:
            raise ValueError("video_episodes cannot be negative")
        if self.torch_threads <= 0:
            raise ValueError("torch_threads must be positive")


def collect_dagger_corrections(
    checkpoint_path: Path,
    *,
    output_dir: Path,
    config: DAggerCollectConfig | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    if config is None:
        config = DAggerCollectConfig()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.torch_threads)
    try:
        with claim_run_directory(output_dir):
            return _collect_dagger_corrections_in_claimed_directory(
                checkpoint_path,
                output_dir=output_dir,
                config=config,
                device=device,
            )
    finally:
        torch.set_num_threads(previous_threads)


def _collect_dagger_corrections_in_claimed_directory(
    checkpoint_path: Path,
    *,
    output_dir: Path,
    config: DAggerCollectConfig,
    device: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model, model_config = _load_model(checkpoint_path, torch_device)
    env_config = SOArmEnvConfig(
        observation_mode="multimodal",
        task_level="pick_place",
        grasp_mode="contact_assisted",
        image_size=model_config.image_size,
        max_episode_steps=max(300, config.max_steps),
        domain_randomization=config.domain_randomization,
        include_end_effector_position_in_proprio=uses_end_effector_position(
            model_config.proprio_dim
        ),
    )
    env = SOArmPickPlaceEnv(env_config)
    expert = PickPlaceExpert(env_config)
    records: list[dict[str, Any]] = []
    total_phase_counts = np.zeros(model_config.phase_count, dtype=np.int64)
    all_errors: list[float] = []
    try:
        for episode_index in range(config.episodes):
            reset_options = None
            if config.balanced_tasks:
                task_index = episode_index % (len(COLORS) * len(SIDES))
                reset_options = {
                    "target_color": COLORS[task_index // len(SIDES)],
                    "goal_side": SIDES[task_index % len(SIDES)],
                }
            episode_seed = config.seed + episode_index
            observation, info = env.reset(seed=episode_seed, options=reset_options)
            expert.reset()
            buffers: dict[str, list[Any]] = {
                "rgb": [],
                "proprio": [],
                "state": [],
                "action": [],
                "model_action": [],
                "executed_action": [],
                "expert_executed": [],
                "phase": [],
                "reward": [],
                "terminated": [],
                "truncated": [],
                "target_pixel": [],
                "goal_pixel": [],
                "pixel_valid": [],
            }
            episode_errors: list[float] = []
            preview_frames: list[NDArray[np.uint8]] = []
            terminated = False
            truncated = False
            for _ in range(config.max_steps):
                predicted_chunk = _predict_action_chunk(model, observation, torch_device)
                model_action = predicted_chunk[0]
                expert_phase = int(expert.phase)
                expert_action = expert.act(info)
                use_expert = bool(rng.random() < config.expert_mixing_probability)
                executed_action = expert_action if use_expert else model_action
                target_pixel, target_visible = env.project_world_point(
                    info["target_position"]
                )
                goal_pixel, goal_visible = env.project_world_point(info["goal_position"])

                buffers["rgb"].append(observation["rgb"])
                buffers["proprio"].append(observation["proprio"])
                buffers["state"].append(env.privileged_state())
                buffers["action"].append(expert_action)
                buffers["model_action"].append(model_action)
                buffers["executed_action"].append(executed_action)
                buffers["expert_executed"].append(use_expert)
                buffers["phase"].append(expert_phase)
                buffers["target_pixel"].append(target_pixel)
                buffers["goal_pixel"].append(goal_pixel)
                buffers["pixel_valid"].append([target_visible, goal_visible])
                if episode_index < config.video_episodes:
                    frame = np.repeat(
                        np.repeat(observation["rgb"], 4, axis=0),
                        4,
                        axis=1,
                    )
                    preview_frames.append(frame)

                error = float(np.mean(np.abs(model_action - expert_action)))
                episode_errors.append(error)
                observation, reward, terminated, truncated, info = env.step(
                    executed_action
                )
                buffers["reward"].append(reward)
                buffers["terminated"].append(terminated)
                buffers["truncated"].append(truncated)
                if terminated or truncated:
                    break

            episode_path = episode_dir / f"episode_{episode_index:06d}.npz"
            np.savez_compressed(
                episode_path,
                rgb=np.asarray(buffers["rgb"], dtype=np.uint8),
                proprio=np.asarray(buffers["proprio"], dtype=np.float32),
                state=np.asarray(buffers["state"], dtype=np.float32),
                language=observation["language"].astype(np.int64),
                language_mask=observation["language_mask"].astype(np.int8),
                action=np.asarray(buffers["action"], dtype=np.float32),
                model_action=np.asarray(buffers["model_action"], dtype=np.float32),
                executed_action=np.asarray(
                    buffers["executed_action"],
                    dtype=np.float32,
                ),
                expert_executed=np.asarray(buffers["expert_executed"], dtype=np.bool_),
                phase=np.asarray(buffers["phase"], dtype=np.int64),
                reward=np.asarray(buffers["reward"], dtype=np.float32),
                terminated=np.asarray(buffers["terminated"], dtype=np.bool_),
                truncated=np.asarray(buffers["truncated"], dtype=np.bool_),
                target_pixel=np.asarray(buffers["target_pixel"], dtype=np.float32),
                goal_pixel=np.asarray(buffers["goal_pixel"], dtype=np.float32),
                pixel_valid=np.asarray(buffers["pixel_valid"], dtype=np.bool_),
            )
            phases = np.asarray(buffers["phase"], dtype=np.int64)
            phase_counts = np.bincount(phases, minlength=model_config.phase_count)
            total_phase_counts += phase_counts[: model_config.phase_count]
            all_errors.extend(episode_errors)
            episode_sha256 = _sha256_file(episode_path)
            record = {
                "episode": episode_index,
                "seed": episode_seed,
                "path": str(episode_path.relative_to(output_dir)).replace("\\", "/"),
                "sha256": episode_sha256,
                "length": len(buffers["action"]),
                "instruction": info["instruction"],
                "target_color": info["target_color"],
                "goal_side": info["goal_side"],
                "success": bool(info["success"]),
                "termination_reason": info["termination_reason"],
                "expert_execution_fraction": float(
                    np.mean(buffers["expert_executed"])
                ),
                "mean_model_expert_mae": float(np.mean(episode_errors)),
            }
            records.append(record)
            if preview_frames:
                imageio.mimsave(
                    output_dir / f"episode_{episode_index:03d}_preview.gif",
                    preview_frames[::2],
                    duration=0.10,
                    loop=0,
                )
            print(
                f"dagger episode={episode_index + 1:03d}/{config.episodes} "
                f"steps={record['length']} success={record['success']} "
                f"model_expert_mae={record['mean_model_expert_mae']:.3f}"
            )
    finally:
        env.close()

    fingerprint_payload = {
        "schema_version": DAGGER_SCHEMA_VERSION,
        "checkpoint": str(checkpoint_path.resolve()),
        "collection": asdict(config),
        "episode_hashes": [record["sha256"] for record in records],
    }
    dataset_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": DAGGER_SCHEMA_VERSION,
        "dataset_type": "dagger_corrections",
        "checkpoint": str(checkpoint_path.resolve()),
        "model": asdict(model_config),
        "collection": asdict(config),
        "episodes": len(records),
        "total_steps": sum(record["length"] for record in records),
        "successes": sum(record["success"] for record in records),
        "phase_counts": total_phase_counts.tolist(),
        "mean_model_expert_mae": float(np.mean(all_errors)),
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "runtime": runtime_metadata(device=torch_device),
        "records": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return manifest


class DAggerCorrectionDataset(Dataset[dict[str, torch.Tensor]]):
    """Independent expert labels queried on learner-visited states.

    Only the first action in each chunk is valid. Future oracle actions were
    never observed counterfactually, so marking them valid would fabricate
    supervision.
    """

    def __init__(
        self,
        root: Path,
        *,
        action_horizon: int = 8,
        episode_cache_size: int | None = None,
        proprio_dim: int = JOINT_PROPRIO_DIM,
    ) -> None:
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if episode_cache_size is not None and episode_cache_size <= 0:
            raise ValueError("episode_cache_size must be positive or None")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != DAGGER_SCHEMA_VERSION:
            raise ValueError("unsupported DAgger correction schema")
        if manifest.get("dataset_type") != "dagger_corrections":
            raise ValueError("manifest is not a DAgger correction dataset")
        self.root = root
        self.action_horizon = action_horizon
        self.records = list(manifest["records"])
        self.metadata = manifest
        self.episode_cache_size = episode_cache_size
        self.proprio_dim = proprio_dim
        self._episode_cache: dict[int, dict[str, np.ndarray]] = {}
        self._indices = [
            (episode_index, time_index)
            for episode_index, record in enumerate(self.records)
            for time_index in range(int(record["length"]))
        ]

    def __len__(self) -> int:
        return len(self._indices)

    @property
    def sample_time_indices(self) -> list[int]:
        return [time_index for _, time_index in self._indices]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, time_index = self._indices[index]
        episode = self._load_episode(episode_index)
        action = episode["action"][time_index].astype(np.float32)
        action_chunk = np.repeat(action[None, :], self.action_horizon, axis=0)
        action_mask = np.zeros(self.action_horizon, dtype=np.bool_)
        action_mask[0] = True
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
            "episode_length": torch.tensor(
                episode["action"].shape[0],
                dtype=torch.long,
            ),
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


def _load_model(
    checkpoint_path: Path,
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


def _predict_action_chunk(
    model: TinyVLA,
    observation: dict[str, NDArray[Any]],
    device: torch.device,
) -> NDArray[np.float32]:
    rgb = torch.as_tensor(
        observation["rgb"],
        dtype=torch.float32,
        device=device,
    ).permute(2, 0, 1).unsqueeze(0) / 255.0
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
        output = model(rgb, proprio, language, language_mask)
    return output.action_chunk.squeeze(0).cpu().numpy().astype(np.float32)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
