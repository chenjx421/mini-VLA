from __future__ import annotations

import numpy as np

from embodied_vla.envs import MobileSlamConfig, MobileSlamEnv


def test_mobile_slam_sensor_contract_and_motion() -> None:
    config = MobileSlamConfig(
        lidar_beams=36,
        lidar_noise_std=0.0,
        odometry_linear_noise_std=0.0,
        odometry_angular_noise_std=0.0,
    )
    env = MobileSlamEnv(config)
    try:
        observation = env.reset(seed=5)
        assert observation["scan"].shape == (36,)
        assert np.all(observation["scan"] >= config.lidar_min_range)
        assert np.all(observation["scan"] <= config.lidar_max_range)
        initial_pose = observation["true_pose"].copy()
        for _ in range(10):
            observation = env.step(np.array([0.25, 0.0], dtype=np.float32))
        assert observation["true_pose"][0] > initial_pose[0] + 0.03
        assert np.linalg.norm(
            observation["odometry_pose"][:2] - observation["true_pose"][:2]
        ) < 0.03
    finally:
        env.close()


def test_mobile_slam_seed_reproduces_scan() -> None:
    env = MobileSlamEnv(MobileSlamConfig(lidar_beams=24))
    try:
        scan_a = env.reset(seed=11)["scan"]
        scan_b = env.reset(seed=11)["scan"]
        np.testing.assert_allclose(scan_a, scan_b)
    finally:
        env.close()
