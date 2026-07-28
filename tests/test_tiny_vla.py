from __future__ import annotations

import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch

from embodied_vla.models import TinyVLA, TinyVLAConfig
from embodied_vla.training.vla_trainer import (
    TinyVLATrainConfig,
    _initialize_vla_weights,
    tiny_vla_loss,
)


class TinyVLATest(unittest.TestCase):
    def test_forward_and_loss_contract(self) -> None:
        config = TinyVLAConfig(
            image_size=32,
            patch_size=8,
            model_dim=64,
            attention_heads=4,
            encoder_layers=2,
            decoder_layers=1,
            feedforward_dim=128,
            action_horizon=4,
        )
        model = TinyVLA(config)
        batch_size = 3
        output = model(
            torch.rand(batch_size, 3, 32, 32),
            torch.rand(batch_size, 12),
            torch.randint(0, config.vocabulary_size, (batch_size, 16)),
            torch.ones(batch_size, 16, dtype=torch.bool),
        )
        self.assertEqual(output.action_chunk.shape, (batch_size, 4, 5))
        self.assertEqual(output.phase_logits.shape, (batch_size, 7))
        self.assertEqual(output.grounding_coordinates.shape, (batch_size, 2, 2))
        self.assertEqual(output.grounding_heatmaps.shape, (batch_size, 2, 4, 4))
        torch.testing.assert_close(
            output.grounding_heatmaps.sum(dim=(-1, -2)),
            torch.ones(batch_size, 2),
            atol=1e-5,
            rtol=1e-5,
        )

        batch = {
            "action_chunk": torch.rand(batch_size, 4, 5) * 2.0 - 1.0,
            "action_mask": torch.ones(batch_size, 4, dtype=torch.bool),
            "phase": torch.randint(0, 7, (batch_size,)),
            "target_pixel": torch.rand(batch_size, 2),
            "goal_pixel": torch.rand(batch_size, 2),
            "pixel_valid": torch.ones(batch_size, 2, dtype=torch.bool),
        }
        loss, losses = tiny_vla_loss(
            output,
            batch,
            model_config=config,
            train_config=TinyVLATrainConfig(),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(losses), {
            "total",
            "action",
            "phase",
            "grounding_coordinate",
            "grounding_heatmap",
            "grounding_world",
        })
        loss.backward()
        self.assertIsNotNone(model.action_head[-1].weight.grad)

    def test_flow_matching_head_trains_and_samples(self) -> None:
        config = TinyVLAConfig(
            image_size=32,
            patch_size=8,
            model_dim=32,
            attention_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dim=64,
            action_horizon=3,
            action_head="flow_matching",
            flow_matching_steps=2,
            dropout=0.0,
        )
        model = TinyVLA(config)
        batch_size = 2
        inputs = (
            torch.rand(batch_size, 3, 32, 32),
            torch.rand(batch_size, 12),
            torch.randint(0, config.vocabulary_size, (batch_size, 16)),
            torch.ones(batch_size, 16, dtype=torch.bool),
        )
        action_targets = torch.rand(batch_size, 3, 5) * 2.0 - 1.0
        output = model(*inputs, action_targets=action_targets)
        self.assertEqual(output.flow_velocity.shape, action_targets.shape)
        self.assertEqual(output.flow_target.shape, action_targets.shape)
        loss = torch.nn.functional.mse_loss(output.flow_velocity, output.flow_target)
        loss.backward()
        self.assertIsNotNone(model.flow_velocity_head[-1].weight.grad)

        model.eval()
        with torch.inference_mode():
            sampled = model(*inputs).action_chunk
        self.assertEqual(sampled.shape, action_targets.shape)
        self.assertTrue(torch.all(sampled >= -1.0))
        self.assertTrue(torch.all(sampled <= 1.0))

    def test_grounding_conditioned_action_path_receives_gradients(self) -> None:
        config = TinyVLAConfig(
            image_size=32,
            patch_size=8,
            model_dim=32,
            attention_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dim=64,
            action_horizon=2,
            grounding_action_conditioning=True,
            grounding_coordinate_refinement=True,
        )
        model = TinyVLA(config)
        output = model(
            torch.rand(2, 3, 32, 32),
            torch.rand(2, 12),
            torch.randint(0, config.vocabulary_size, (2, 16)),
            torch.ones(2, 16, dtype=torch.bool),
        )
        (
            output.action_chunk.square().mean()
            + output.grounding_coordinates.square().mean()
        ).backward()

        self.assertIsNotNone(model.grounding_action_projection)
        self.assertIsNotNone(model.grounding_action_projection[0].weight.grad)
        self.assertIsNotNone(model.grounding_attention.in_proj_weight.grad)
        self.assertIsNotNone(model.grounding_coordinate_refiner)
        self.assertIsNotNone(model.grounding_coordinate_refiner[0].weight.grad)
        torch.testing.assert_close(
            model.grounding_action_projection[-1].bias.detach(),
            torch.zeros(config.model_dim),
        )

    def test_grounding_upgrade_is_functionally_zero_initialized(self) -> None:
        base_config = TinyVLAConfig(
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
        upgraded_config = replace(
            base_config,
            grounding_action_conditioning=True,
            grounding_coordinate_refinement=True,
        )
        base_model = TinyVLA(base_config).eval()
        upgraded_model = TinyVLA(upgraded_config).eval()
        metadata = _initialize_vla_weights(
            upgraded_model,
            {
                "model": base_model.state_dict(),
                "model_config": asdict(base_config),
            },
            model_config=upgraded_config,
            checkpoint_path=Path("base.pt"),
        )
        inputs = (
            torch.rand(2, 3, 32, 32),
            torch.rand(2, 12),
            torch.randint(0, base_config.vocabulary_size, (2, 16)),
            torch.ones(2, 16, dtype=torch.bool),
        )
        with torch.inference_mode():
            base_output = base_model(*inputs)
            upgraded_output = upgraded_model(*inputs)

        torch.testing.assert_close(
            upgraded_output.action_chunk,
            base_output.action_chunk,
        )
        torch.testing.assert_close(
            upgraded_output.grounding_coordinates,
            base_output.grounding_coordinates,
        )
        self.assertEqual(
            set(metadata["changed_model_fields"]),
            {
                "grounding_action_conditioning",
                "grounding_coordinate_refinement",
            },
        )

    def test_high_resolution_grounding_upgrade_preserves_old_outputs(self) -> None:
        base_config = TinyVLAConfig(
            image_size=32,
            patch_size=8,
            model_dim=32,
            attention_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dim=64,
            action_horizon=2,
            dropout=0.0,
            grounding_action_conditioning=True,
            grounding_coordinate_refinement=True,
        )
        upgraded_config = replace(
            base_config,
            high_resolution_grounding=True,
        )
        base_model = TinyVLA(base_config).eval()
        upgraded_model = TinyVLA(upgraded_config).eval()
        metadata = _initialize_vla_weights(
            upgraded_model,
            {
                "model": base_model.state_dict(),
                "model_config": asdict(base_config),
            },
            model_config=upgraded_config,
            checkpoint_path=Path("base.pt"),
        )
        inputs = (
            torch.rand(2, 3, 32, 32),
            torch.rand(2, 12),
            torch.randint(0, base_config.vocabulary_size, (2, 16)),
            torch.ones(2, 16, dtype=torch.bool),
        )
        with torch.inference_mode():
            base_output = base_model(*inputs)
            upgraded_output = upgraded_model(*inputs)

        torch.testing.assert_close(
            upgraded_output.action_chunk,
            base_output.action_chunk,
        )
        torch.testing.assert_close(
            upgraded_output.grounding_coordinates,
            base_output.grounding_coordinates,
        )
        self.assertEqual(upgraded_output.grounding_heatmaps.shape, (2, 2, 8, 8))
        self.assertEqual(
            metadata["changed_model_fields"],
            {"high_resolution_grounding": {"from": False, "to": True}},
        )

    def test_high_resolution_grounding_receives_heatmap_gradients(self) -> None:
        config = TinyVLAConfig(
            image_size=32,
            patch_size=8,
            model_dim=32,
            attention_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dim=64,
            action_horizon=2,
            high_resolution_grounding=True,
        )
        model = TinyVLA(config)
        batch_size = 2
        output = model(
            torch.rand(batch_size, 3, 32, 32),
            torch.rand(batch_size, 12),
            torch.randint(0, config.vocabulary_size, (batch_size, 16)),
            torch.ones(batch_size, 16, dtype=torch.bool),
        )
        batch = {
            "action_chunk": torch.rand(batch_size, 2, 5) * 2.0 - 1.0,
            "action_mask": torch.ones(batch_size, 2, dtype=torch.bool),
            "phase": torch.randint(0, 7, (batch_size,)),
            "target_pixel": torch.rand(batch_size, 2),
            "goal_pixel": torch.rand(batch_size, 2),
            "pixel_valid": torch.ones(batch_size, 2, dtype=torch.bool),
        }
        loss, _ = tiny_vla_loss(
            output,
            batch,
            model_config=config,
            train_config=TinyVLATrainConfig(),
        )
        loss.backward()

        self.assertIsNotNone(model.high_resolution_grounding_stem)
        self.assertIsNotNone(
            model.high_resolution_grounding_stem[0].weight.grad
        )
        self.assertIsNotNone(model.high_resolution_grounding_gate)
        self.assertIsNotNone(model.high_resolution_grounding_gate.grad)

    def test_cartesian_proprio_upgrade_preserves_old_outputs(self) -> None:
        base_config = TinyVLAConfig(
            image_size=32,
            patch_size=8,
            model_dim=32,
            attention_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dim=64,
            action_horizon=2,
            dropout=0.0,
            grounding_action_conditioning=True,
            grounding_coordinate_refinement=True,
        )
        upgraded_config = replace(base_config, proprio_dim=15)
        base_model = TinyVLA(base_config).eval()
        upgraded_model = TinyVLA(upgraded_config).eval()
        metadata = _initialize_vla_weights(
            upgraded_model,
            {
                "model": base_model.state_dict(),
                "model_config": asdict(base_config),
            },
            model_config=upgraded_config,
            checkpoint_path=Path("base.pt"),
        )
        rgb = torch.rand(2, 3, 32, 32)
        joint_proprio = torch.rand(2, 12)
        cartesian_proprio = torch.cat(
            (joint_proprio, torch.rand(2, 3)),
            dim=-1,
        )
        language = torch.randint(0, base_config.vocabulary_size, (2, 16))
        language_mask = torch.ones(2, 16, dtype=torch.bool)
        with torch.inference_mode():
            base_output = base_model(
                rgb,
                joint_proprio,
                language,
                language_mask,
            )
            upgraded_output = upgraded_model(
                rgb,
                cartesian_proprio,
                language,
                language_mask,
            )

        torch.testing.assert_close(
            upgraded_output.action_chunk,
            base_output.action_chunk,
        )
        torch.testing.assert_close(
            upgraded_output.grounding_coordinates,
            base_output.grounding_coordinates,
        )
        self.assertEqual(
            set(metadata["shape_adapted_parameters"]),
            {
                "proprio_projection.0.weight",
                "grounding_action_projection.0.weight",
            },
        )
        self.assertEqual(
            metadata["changed_model_fields"]["proprio_dim"],
            {"from": 12, "to": 15},
        )

    def test_world_grounding_upgrade_preserves_action_outputs(self) -> None:
        base_config = TinyVLAConfig(
            image_size=32,
            patch_size=8,
            proprio_dim=15,
            model_dim=32,
            attention_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dim=64,
            action_horizon=2,
            dropout=0.0,
            grounding_action_conditioning=True,
            grounding_coordinate_refinement=True,
        )
        upgraded_config = replace(
            base_config,
            world_grounding=True,
            world_grounding_action_conditioning=True,
            phase_action_conditioning=True,
        )
        base_model = TinyVLA(base_config).eval()
        upgraded_model = TinyVLA(upgraded_config).eval()
        metadata = _initialize_vla_weights(
            upgraded_model,
            {
                "model": base_model.state_dict(),
                "model_config": asdict(base_config),
            },
            model_config=upgraded_config,
            checkpoint_path=Path("base.pt"),
        )
        inputs = (
            torch.rand(2, 3, 32, 32),
            torch.rand(2, 15),
            torch.randint(0, base_config.vocabulary_size, (2, 16)),
            torch.ones(2, 16, dtype=torch.bool),
        )
        with torch.inference_mode():
            base_output = base_model(*inputs)
            upgraded_output = upgraded_model(*inputs)

        torch.testing.assert_close(
            upgraded_output.action_chunk,
            base_output.action_chunk,
        )
        torch.testing.assert_close(
            upgraded_output.grounding_coordinates,
            base_output.grounding_coordinates,
        )
        self.assertIsNotNone(upgraded_output.grounding_world_positions)
        self.assertEqual(
            upgraded_output.grounding_world_positions.shape,
            (2, 2, 3),
        )
        self.assertEqual(
            set(metadata["changed_model_fields"]),
            {
                "world_grounding",
                "world_grounding_action_conditioning",
                "phase_action_conditioning",
            },
        )


if __name__ == "__main__":
    unittest.main()
