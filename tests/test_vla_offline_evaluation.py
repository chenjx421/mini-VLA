from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from embodied_vla.data.trajectory import SCHEMA_VERSION
from embodied_vla.evaluation import OfflineVLAEvalConfig, evaluate_tiny_vla_offline
from embodied_vla.models import TinyVLA, TinyVLAConfig


def test_tiny_vla_offline_evaluation_writes_grouped_metrics() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        dataset_root = root / "dataset"
        episode_root = dataset_root / "episodes"
        episode_root.mkdir(parents=True)
        records = []
        for episode_index in range(4):
            length = 3
            episode_path = episode_root / f"episode_{episode_index:06d}.npz"
            np.savez_compressed(
                episode_path,
                rgb=np.zeros((length, 32, 32, 3), dtype=np.uint8),
                proprio=np.zeros((length, 12), dtype=np.float32),
                state=np.zeros((length, 37), dtype=np.float32),
                language=np.zeros(16, dtype=np.int64),
                language_mask=np.ones(16, dtype=np.int8),
                action=np.zeros((length, 5), dtype=np.float32),
                phase=np.arange(length, dtype=np.int64),
                reward=np.zeros(length, dtype=np.float32),
                terminated=np.zeros(length, dtype=np.bool_),
                truncated=np.zeros(length, dtype=np.bool_),
                target_pixel=np.full((length, 2), 0.25, dtype=np.float32),
                goal_pixel=np.full((length, 2), 0.75, dtype=np.float32),
                pixel_valid=np.ones((length, 2), dtype=np.bool_),
            )
            records.append(
                {
                    "episode": episode_index,
                    "path": str(episode_path.relative_to(dataset_root)).replace("\\", "/"),
                    "length": length,
                    "target_color": "red" if episode_index < 2 else "blue",
                    "goal_side": "left",
                }
            )
        (dataset_root / "manifest.json").write_text(
            json.dumps({"schema_version": SCHEMA_VERSION, "records": records}),
            encoding="utf-8",
        )

        model_config = TinyVLAConfig(
            image_size=32,
            patch_size=8,
            model_dim=32,
            attention_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dim=64,
            action_horizon=2,
            dropout=0.0,
        )
        model = TinyVLA(model_config)
        checkpoint_path = root / "model.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "model_config": asdict(model_config),
            },
            checkpoint_path,
        )

        output_dir = root / "offline"
        summary = evaluate_tiny_vla_offline(
            checkpoint_path,
            dataset_root,
            output_dir=output_dir,
            eval_config=OfflineVLAEvalConfig(
                split="validation",
                batch_size=2,
                progress_bins=3,
            ),
        )

        assert summary["samples"] == 6
        assert summary["episodes"] == 2
        assert summary["initial_state_metrics"]["samples"] == 2
        assert "predicted_first_action_std" in summary["initial_state_metrics"]
        assert "first_action_correlation" in summary["initial_state_metrics"]
        assert "runtime" in summary
        assert set(summary["by_true_phase"]) == {
            "approach",
            "descend_grasp",
            "close_gripper",
        }
        assert len(summary["phase_confusion_matrix"]) == model_config.phase_count
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "samples.jsonl").exists()
        assert (output_dir / "offline_diagnostics.png").exists()
