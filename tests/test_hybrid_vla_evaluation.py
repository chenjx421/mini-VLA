from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from embodied_vla.evaluation.hybrid_vla import (
    HybridVLAEvalConfig,
    evaluate_hybrid_vla,
)
from embodied_vla.models import TinyVLA, TinyVLAConfig


def test_hybrid_vla_evaluation_smoke() -> None:
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
        summary = evaluate_hybrid_vla(
            checkpoint_path,
            output_dir=root / "hybrid",
            eval_config=HybridVLAEvalConfig(
                episodes=1,
                seed=123,
                video_episodes=0,
                max_episode_steps=2,
            ),
        )

        assert summary["episodes"] == 1
        assert (
            summary["policy_boundary"][
                "action_uses_privileged_target_or_goal_coordinates"
            ]
            is False
        )
        assert summary["mean_pregrasp_target_world_xy_error_m"] is not None
        assert summary["mean_raw_pixel_grounding_l2"] >= 0.0
        assert (root / "hybrid" / "summary.json").exists()
        assert (root / "hybrid" / "episode_000_trace.jsonl").exists()
