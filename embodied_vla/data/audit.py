from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from embodied_vla.data.trajectory import SCHEMA_VERSION


@dataclass
class _RunningVectorMoments:
    dimension: int

    def __post_init__(self) -> None:
        self.count = 0
        self.total = np.zeros(self.dimension, dtype=np.float64)
        self.total_squared = np.zeros(self.dimension, dtype=np.float64)
        self.minimum = np.full(self.dimension, np.inf, dtype=np.float64)
        self.maximum = np.full(self.dimension, -np.inf, dtype=np.float64)

    def update(self, values: NDArray[np.floating]) -> None:
        flattened = np.asarray(values, dtype=np.float64).reshape(-1, self.dimension)
        self.count += flattened.shape[0]
        self.total += flattened.sum(axis=0)
        self.total_squared += np.square(flattened).sum(axis=0)
        self.minimum = np.minimum(self.minimum, flattened.min(axis=0))
        self.maximum = np.maximum(self.maximum, flattened.max(axis=0))

    def summary(self) -> dict[str, Any]:
        mean = self.total / max(1, self.count)
        variance = self.total_squared / max(1, self.count) - np.square(mean)
        return {
            "count": self.count,
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(variance, 0.0)).tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
        }


def audit_expert_dataset(
    root: Path,
    *,
    write_statistics: bool = True,
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported trajectory schema")
    records = list(manifest.get("records", []))
    if len(records) < 2:
        raise ValueError("dataset must contain at least two episodes")
    if len({record["seed"] for record in records}) != len(records):
        raise ValueError("episode seeds must be unique")
    if len({record["path"] for record in records}) != len(records):
        raise ValueError("episode paths must be unique")

    fingerprint = hashlib.sha256()
    fingerprint.update(manifest_bytes)
    action_moments = _RunningVectorMoments(5)
    proprio_moments = _RunningVectorMoments(12)
    state_moments = _RunningVectorMoments(37)
    image_moments = _RunningVectorMoments(3)
    lengths: list[int] = []
    phase_counts = np.zeros(9, dtype=np.int64)
    expected_time_fields = (
        "rgb",
        "proprio",
        "state",
        "action",
        "phase",
        "reward",
        "terminated",
        "truncated",
        "target_pixel",
        "goal_pixel",
        "pixel_valid",
    )

    for record in records:
        episode_path = root / record["path"]
        relative_path = episode_path.relative_to(root).as_posix()
        fingerprint.update(relative_path.encode("utf-8"))
        with episode_path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                fingerprint.update(block)

        with np.load(episode_path) as episode:
            length = int(record["length"])
            if length < 10:
                raise ValueError(f"implausibly short successful episode: {relative_path}")
            for field in expected_time_fields:
                if field not in episode:
                    raise ValueError(f"{relative_path} is missing field {field}")
                if episode[field].shape[0] != length:
                    raise ValueError(
                        f"{relative_path}:{field} length {episode[field].shape[0]} != {length}"
                    )
            if episode["rgb"].ndim != 4 or episode["rgb"].shape[-1] != 3:
                raise ValueError(f"{relative_path} has an invalid RGB shape")
            if episode["proprio"].shape[1:] != (12,):
                raise ValueError(f"{relative_path} has an invalid proprio shape")
            if episode["state"].shape[1:] != (37,):
                raise ValueError(f"{relative_path} has an invalid state shape")
            if episode["action"].shape[1:] != (5,):
                raise ValueError(f"{relative_path} has an invalid action shape")
            if not bool(episode["terminated"][-1]):
                raise ValueError(f"{relative_path} does not end in a true success terminal")
            if bool(episode["truncated"][-1]):
                raise ValueError(f"{relative_path} is marked both successful and truncated")
            if not np.isfinite(episode["action"]).all():
                raise ValueError(f"{relative_path} contains non-finite actions")
            if np.abs(episode["action"]).max() > 1.0001:
                raise ValueError(f"{relative_path} contains an out-of-range action")
            phases = episode["phase"].astype(np.int64)
            if phases.min() < 0 or phases.max() >= len(phase_counts):
                raise ValueError(f"{relative_path} contains an invalid expert phase")

            lengths.append(length)
            action_moments.update(episode["action"])
            proprio_moments.update(episode["proprio"])
            state_moments.update(episode["state"])
            image_moments.update(episode["rgb"].astype(np.float32) / 255.0)
            phase_counts += np.bincount(phases, minlength=len(phase_counts))

    task_counts = {
        f"{color}->{side}": sum(
            record.get("target_color") == color and record.get("goal_side") == side
            for record in records
        )
        for color in ("red", "green", "blue")
        for side in ("left", "right")
    }
    if any(count == 0 for count in task_counts.values()):
        raise ValueError("dataset does not cover every color and goal-side task")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_fingerprint_sha256": fingerprint.hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "episodes": len(records),
        "attempted": int(manifest.get("attempted", len(records))),
        "rejected": int(manifest.get("attempted", len(records))) - len(records),
        "total_steps": int(sum(lengths)),
        "episode_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
        },
        "task_counts": task_counts,
        "phase_counts": phase_counts.tolist(),
        "normalization": {
            "rgb_0_1": image_moments.summary(),
            "proprio": proprio_moments.summary(),
            "action_normalized_interface": action_moments.summary(),
            "privileged_state": state_moments.summary(),
        },
    }
    if write_statistics:
        (root / "statistics.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    return summary
