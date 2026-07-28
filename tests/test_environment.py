from __future__ import annotations

import unittest

import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

from embodied_vla.envs import SOArmEnvConfig, SOArmPickPlaceEnv
from embodied_vla.experts import PickPlaceExpert


class SOArmEnvironmentTest(unittest.TestCase):
    def test_gymnasium_state_contract(self) -> None:
        env = SOArmPickPlaceEnv(SOArmEnvConfig(observation_mode="state"))
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()

    def test_seed_reproduces_task_and_state(self) -> None:
        env = SOArmPickPlaceEnv(SOArmEnvConfig(observation_mode="state"))
        try:
            observation_a, info_a = env.reset(seed=19)
            observation_b, info_b = env.reset(seed=19)
            np.testing.assert_allclose(observation_a, observation_b, atol=1e-6)
            self.assertEqual(info_a["instruction"], info_b["instruction"])
            np.testing.assert_allclose(
                info_a["target_position"],
                info_b["target_position"],
                atol=1e-6,
            )
        finally:
            env.close()

    def test_multimodal_observation_contract(self) -> None:
        config = SOArmEnvConfig(observation_mode="multimodal", image_size=48)
        env = SOArmPickPlaceEnv(config)
        try:
            observation, info = env.reset(seed=23)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertEqual(observation["rgb"].shape, (48, 48, 3))
            self.assertEqual(observation["proprio"].shape, (12,))
            self.assertEqual(observation["language"].shape, (16,))
            depth = env.render_depth()
            self.assertEqual(depth.shape, (48, 48))
            self.assertTrue(np.isfinite(depth).all())
            self.assertGreater(depth.min(), 0.0)
            self.assertEqual(env.camera_intrinsics().shape, (3, 3))
            camera_position, camera_rotation = env.camera_pose()
            self.assertEqual(camera_position.shape, (3,))
            self.assertEqual(camera_rotation.shape, (3, 3))
            np.testing.assert_allclose(
                camera_rotation.T @ camera_rotation,
                np.eye(3),
                atol=1e-6,
            )
            self.assertEqual(env.joint_positions.shape, (6,))
            target_position = np.asarray(info["target_position"])
            target_pixel, target_visible = env.project_world_point(target_position)
            self.assertTrue(target_visible)
            reconstructed = env.unproject_normalized_pixel_to_plane(
                target_pixel,
                world_z=float(target_position[2]),
            )
            np.testing.assert_allclose(reconstructed, target_position, atol=1e-6)
        finally:
            env.close()

    def test_cartesian_proprioception_appends_normalized_end_effector(self) -> None:
        config = SOArmEnvConfig(
            observation_mode="multimodal",
            image_size=32,
            include_end_effector_position_in_proprio=True,
        )
        env = SOArmPickPlaceEnv(config)
        try:
            observation, info = env.reset(seed=29)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertEqual(observation["proprio"].shape, (15,))
            self.assertEqual(env.privileged_state().shape, (37,))
            np.testing.assert_allclose(
                observation["proprio"][-3:],
                np.asarray(info["end_effector_position"]) / 0.5,
                atol=1e-6,
            )
        finally:
            env.close()

    def test_strict_contact_mode_never_activates_assist(self) -> None:
        config = SOArmEnvConfig(observation_mode="state", grasp_mode="contact")
        env = SOArmPickPlaceEnv(config)
        try:
            _, info = env.reset(seed=31)
            for _ in range(20):
                _, _, terminated, truncated, info = env.step(
                    np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
                )
                self.assertFalse(info["assisted_grasp_active"])
                self.assertFalse(env.data.eq_active.any())
                if terminated or truncated:
                    break
        finally:
            env.close()

    def test_pick_place_requires_a_real_lift(self) -> None:
        config = SOArmEnvConfig(observation_mode="state", task_level="pick_place")
        env = SOArmPickPlaceEnv(config)
        try:
            _, info = env.reset(
                seed=37,
                options={"target_color": "red", "goal_side": "left"},
            )
            target_qpos = env._cube_qpos_adrs[env._target_index]
            env.data.qpos[target_qpos : target_qpos + 2] = info["goal_position"][:2]
            env.data.qpos[target_qpos + 2] = config.table_height + config.cube_half_size
            mujoco.mj_forward(env.model, env.data)
            env._joint_targets[5] = config.open_jaw
            env._last_contact = env._contact_state()
            self.assertFalse(env._has_lifted)
            self.assertFalse(env._is_success())
        finally:
            env.close()

    def test_target_never_initializes_inside_goal(self) -> None:
        config = SOArmEnvConfig(observation_mode="state", task_level="pick_place")
        env = SOArmPickPlaceEnv(config)
        try:
            for seed in range(20):
                _, info = env.reset(seed=seed)
                self.assertGreater(
                    info["goal_distance"],
                    config.success_radius + 0.015,
                )
        finally:
            env.close()

    def test_assisted_expert_solves_held_out_seeds(self) -> None:
        config = SOArmEnvConfig(
            observation_mode="state",
            grasp_mode="contact_assisted",
            max_episode_steps=300,
        )
        successes = 0
        env = SOArmPickPlaceEnv(config)
        expert = PickPlaceExpert(config)
        try:
            for seed in range(700, 710):
                _, info = env.reset(seed=seed)
                expert.reset()
                for _ in range(config.max_episode_steps):
                    _, _, terminated, truncated, info = env.step(expert.act(info))
                    if terminated or truncated:
                        break
                successes += int(info["success"])
        finally:
            env.close()
        self.assertGreaterEqual(successes, 9)


if __name__ == "__main__":
    unittest.main()
