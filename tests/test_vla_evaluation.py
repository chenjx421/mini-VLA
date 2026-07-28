from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from embodied_vla.evaluation import VLAEvalConfig, evaluate_tiny_vla
from embodied_vla.models import TinyVLA, TinyVLAConfig


def test_tiny_vla_closed_loop_evaluation_smoke() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
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
        summary = evaluate_tiny_vla(
            checkpoint_path,
            output_dir=root / "evaluation",
            eval_config=VLAEvalConfig(
                episodes=1,
                seed=123,
                execution_horizon=1,
                video_episodes=0,
                max_episode_steps=2,
                cartesian_action_gain=2.0,
            ),
        )
        assert summary["episodes"] == 1
        assert 0.0 <= summary["success_rate"] <= 1.0
        assert summary["mean_minimum_approach_waypoint_distance"] >= 0.0
        assert summary["mean_minimum_approach_waypoint_xy_error"] >= 0.0
        assert summary["mean_minimum_approach_waypoint_z_error"] >= 0.0
        assert (root / "evaluation" / "summary.json").exists()
        assert not (root / "evaluation" / ".run.lock").exists()
        trace = [
            json.loads(line)
            for line in (
                root / "evaluation" / "episode_000_trace.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        raw_action = torch.tensor(trace[0]["model_action"])
        executed_action = torch.tensor(trace[0]["executed_action"])
        torch.testing.assert_close(
            executed_action[:3],
            torch.clamp(raw_action[:3] * 2.0, -1.0, 1.0),
        )
        torch.testing.assert_close(executed_action[3:], raw_action[3:])
