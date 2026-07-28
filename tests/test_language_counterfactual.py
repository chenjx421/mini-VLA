from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from embodied_vla.evaluation.counterfactual import (
    CounterfactualEvalConfig,
    evaluate_language_counterfactuals,
)
from embodied_vla.models import TinyVLA, TinyVLAConfig


def test_counterfactual_language_evaluation_controls_scene_state() -> None:
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

        output_dir = root / "counterfactual"
        summary = evaluate_language_counterfactuals(
            checkpoint_path,
            output_dir=output_dir,
            eval_config=CounterfactualEvalConfig(
                scenes=1,
                seed=321,
                visualized_scenes=1,
            ),
        )

        assert summary["scenes"] == 1
        assert summary["task_variants_per_scene"] == 6
        assert 0.0 <= summary["target_grounding_accuracy"] <= 1.0
        assert 0.0 <= summary["goal_grounding_accuracy"] <= 1.0
        assert summary["mean_color_action_rms"] >= 0.0
        assert summary["mean_side_action_rms"] >= 0.0
        assert (output_dir / "counterfactuals.jsonl").exists()
        assert (output_dir / "scene_000_counterfactual.png").exists()
