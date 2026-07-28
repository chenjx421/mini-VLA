from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from embodied_vla.data.dagger import (
    DAGGER_SCHEMA_VERSION,
    DAggerCorrectionDataset,
)


def test_dagger_correction_dataset_only_supervises_queried_action() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        episode_root = root / "episodes"
        episode_root.mkdir()
        length = 3
        episode_path = episode_root / "episode_000000.npz"
        actions = np.arange(length * 5, dtype=np.float32).reshape(length, 5) / 20.0
        np.savez_compressed(
            episode_path,
            rgb=np.zeros((length, 32, 32, 3), dtype=np.uint8),
            proprio=np.zeros((length, 12), dtype=np.float32),
            state=np.zeros((length, 37), dtype=np.float32),
            language=np.zeros(16, dtype=np.int64),
            language_mask=np.ones(16, dtype=np.int8),
            action=actions,
            phase=np.arange(length, dtype=np.int64),
            target_pixel=np.zeros((length, 2), dtype=np.float32),
            goal_pixel=np.ones((length, 2), dtype=np.float32),
            pixel_valid=np.ones((length, 2), dtype=np.bool_),
        )
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": DAGGER_SCHEMA_VERSION,
                    "dataset_type": "dagger_corrections",
                    "records": [
                        {
                            "episode": 0,
                            "path": "episodes/episode_000000.npz",
                            "length": length,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        dataset = DAggerCorrectionDataset(root, action_horizon=4)
        sample = dataset[1]

        assert len(dataset) == length
        assert sample["action_chunk"].shape == (4, 5)
        assert sample["action_mask"].tolist() == [True, False, False, False]
        np.testing.assert_allclose(sample["action_chunk"][0].numpy(), actions[1])
        np.testing.assert_allclose(sample["action_chunk"][1].numpy(), actions[1])
        assert int(sample["time_index"]) == 1
        assert sample["target_world"].shape == (3,)
        assert sample["goal_world"].shape == (3,)
        assert dataset.sample_time_indices == [0, 1, 2]

        augmented_dataset = DAggerCorrectionDataset(
            root,
            action_horizon=4,
            proprio_dim=15,
        )
        augmented_sample = augmented_dataset[1]
        assert augmented_sample["proprio"].shape == (15,)
        np.testing.assert_allclose(
            augmented_sample["proprio"][-3:].numpy(),
            augmented_sample["privileged_state"][12:15].numpy(),
        )
