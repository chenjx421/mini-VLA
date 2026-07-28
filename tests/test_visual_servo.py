from __future__ import annotations

import numpy as np

from embodied_vla.control import GroundedVisualServoController
from embodied_vla.envs import SOArmEnvConfig, SOArmPickPlaceEnv


def test_calibrated_visual_servo_solves_perfect_grounding_rollout() -> None:
    config = SOArmEnvConfig(
        observation_mode="multimodal",
        image_size=32,
        max_episode_steps=300,
        grasp_mode="contact_assisted",
    )
    env = SOArmPickPlaceEnv(config)
    controller = GroundedVisualServoController(config, smoothing_alpha=1.0)
    successes = 0
    try:
        for seed in range(910, 916):
            _, info = env.reset(seed=seed)
            controller.reset()
            for _ in range(config.max_episode_steps):
                target_pixel, _ = env.project_world_point(info["target_position"])
                goal_pixel, _ = env.project_world_point(info["goal_position"])
                target_estimate = env.unproject_normalized_pixel_to_plane(
                    target_pixel,
                    world_z=config.cube_half_size,
                )
                goal_estimate = env.unproject_normalized_pixel_to_plane(
                    goal_pixel,
                    world_z=0.004,
                )
                action = controller.act(
                    {
                        "end_effector_position": info["end_effector_position"],
                        "bilateral_contact": info["bilateral_contact"],
                        "assisted_grasp_active": info["assisted_grasp_active"],
                    },
                    target_position_estimate=target_estimate,
                    goal_position_estimate=goal_estimate,
                )
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            successes += int(info["success"])
    finally:
        env.close()

    assert successes >= 5


def test_visual_servo_recovery_search_changes_failed_grasp_location() -> None:
    config = SOArmEnvConfig()
    controller = GroundedVisualServoController(
        config,
        recovery_search_radius_m=0.02,
    )

    np.testing.assert_allclose(controller.search_offset, np.zeros(3))
    controller.retries = 1
    np.testing.assert_allclose(controller.search_offset, [-0.02, 0.0, 0.0])
    controller.retries = 2
    np.testing.assert_allclose(controller.search_offset, [0.02, 0.0, 0.0])
    controller.retries = 3
    np.testing.assert_allclose(controller.search_offset, [0.0, -0.02, 0.0])
