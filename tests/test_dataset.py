from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from embodied_vla.data.trajectory import SCHEMA_VERSION, ActionChunkDataset


class ActionChunkDatasetTest(unittest.TestCase):
    def test_action_chunk_padding_and_episode_split(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "episodes").mkdir()
            records = []
            for episode_index in range(4):
                path = root / "episodes" / f"episode_{episode_index:06d}.npz"
                length = 3 + episode_index
                np.savez_compressed(
                    path,
                    rgb=np.zeros((length, 32, 32, 3), dtype=np.uint8),
                    proprio=np.zeros((length, 12), dtype=np.float32),
                    state=np.zeros((length, 37), dtype=np.float32),
                    language=np.zeros(16, dtype=np.int64),
                    language_mask=np.ones(16, dtype=np.int8),
                    action=np.ones((length, 5), dtype=np.float32),
                    phase=np.zeros(length, dtype=np.int64),
                    reward=np.zeros(length, dtype=np.float32),
                    terminated=np.zeros(length, dtype=np.bool_),
                    truncated=np.zeros(length, dtype=np.bool_),
                    target_pixel=np.zeros((length, 2), dtype=np.float32),
                    goal_pixel=np.zeros((length, 2), dtype=np.float32),
                    pixel_valid=np.ones((length, 2), dtype=np.bool_),
                )
                records.append(
                    {
                        "episode": episode_index,
                        "path": str(path.relative_to(root)).replace("\\", "/"),
                        "length": length,
                    }
                )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "records": records,
                    }
                ),
                encoding="utf-8",
            )
            train = ActionChunkDataset(
                root,
                action_horizon=5,
                split="train",
                validation_fraction=0.25,
            )
            validation = ActionChunkDataset(
                root,
                action_horizon=5,
                split="validation",
                validation_fraction=0.25,
            )
            train_paths = {record["path"] for record in train.records}
            validation_paths = {record["path"] for record in validation.records}
            self.assertFalse(train_paths & validation_paths)
            sample = train[len(train) - 1]
            self.assertEqual(sample["action_chunk"].shape, (5, 5))
            self.assertEqual(sample["action_mask"].shape, (5,))
            self.assertIn("episode_index", sample)
            self.assertIn("time_index", sample)
            self.assertIn("episode_length", sample)
            self.assertEqual(sample["target_world"].shape, (3,))
            self.assertEqual(sample["goal_world"].shape, (3,))
            self.assertEqual(len(train.sample_time_indices), len(train))
            self.assertGreaterEqual(int(sample["action_mask"].sum()), 1)
            for index in range(len(train)):
                train[index]
            self.assertEqual(len(train._episode_cache), len(train.records))

            bounded = ActionChunkDataset(
                root,
                action_horizon=5,
                split="train",
                validation_fraction=0.25,
                episode_cache_size=1,
            )
            for index in range(len(bounded)):
                bounded[index]
            self.assertLessEqual(len(bounded._episode_cache), 1)

            augmented = ActionChunkDataset(
                root,
                action_horizon=5,
                split="train",
                validation_fraction=0.25,
                proprio_dim=15,
            )
            augmented_sample = augmented[0]
            self.assertEqual(augmented_sample["proprio"].shape, (15,))
            np.testing.assert_allclose(
                augmented_sample["proprio"][-3:].numpy(),
                augmented_sample["privileged_state"][12:15].numpy(),
            )

    def test_task_stratified_episode_split(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            records = []
            episode_index = 0
            for color in ("red", "green", "blue"):
                for side in ("left", "right"):
                    for _ in range(2):
                        records.append(
                            {
                                "episode": episode_index,
                                "path": f"episodes/episode_{episode_index:06d}.npz",
                                "length": 3,
                                "target_color": color,
                                "goal_side": side,
                            }
                        )
                        episode_index += 1
            (root / "manifest.json").write_text(
                json.dumps({"schema_version": SCHEMA_VERSION, "records": records}),
                encoding="utf-8",
            )
            train = ActionChunkDataset(root, split="train", validation_fraction=0.5)
            validation = ActionChunkDataset(
                root,
                split="validation",
                validation_fraction=0.5,
            )
            self.assertEqual(train.split_strategy, "stratified_task_episode")
            self.assertEqual(validation.split_strategy, "stratified_task_episode")
            self.assertEqual(len(train.records), 6)
            self.assertEqual(len(validation.records), 6)
            validation_tasks = {
                (record["target_color"], record["goal_side"])
                for record in validation.records
            }
            self.assertEqual(len(validation_tasks), 6)


if __name__ == "__main__":
    unittest.main()
