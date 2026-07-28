from __future__ import annotations

import unittest

import numpy as np
import torch

from embodied_vla.algorithms import compute_gae
from embodied_vla.models import StateActorCritic


class PPOTest(unittest.TestCase):
    def test_gae_stops_at_episode_boundary(self) -> None:
        rewards = np.array([[1.0], [1.0], [10.0]], dtype=np.float32)
        values = np.zeros_like(rewards)
        bootstrap = np.zeros_like(rewards)
        episode_ends = np.array([[False], [True], [True]])
        advantages, returns = compute_gae(
            rewards,
            values,
            bootstrap,
            episode_ends,
            gamma=1.0,
            gae_lambda=1.0,
        )
        np.testing.assert_allclose(advantages[:, 0], [2.0, 1.0, 10.0])
        np.testing.assert_allclose(returns, advantages)

    def test_truncation_bootstraps_without_leaking_next_episode(self) -> None:
        rewards = np.array([[1.0]], dtype=np.float32)
        values = np.array([[0.5]], dtype=np.float32)
        bootstrap = np.array([[2.0]], dtype=np.float32)
        episode_ends = np.array([[True]])
        advantages, _ = compute_gae(
            rewards,
            values,
            bootstrap,
            episode_ends,
            gamma=0.9,
            gae_lambda=0.95,
        )
        self.assertAlmostEqual(float(advantages[0, 0]), 1.0 + 0.9 * 2.0 - 0.5)

    def test_squashed_policy_contract(self) -> None:
        policy = StateActorCritic(37, 5)
        observation = torch.randn(6, 37)
        action, raw_action, log_probability, value = policy.sample(observation)
        self.assertEqual(action.shape, (6, 5))
        self.assertEqual(raw_action.shape, (6, 5))
        self.assertEqual(log_probability.shape, (6,))
        self.assertEqual(value.shape, (6,))
        self.assertTrue(torch.isfinite(log_probability).all())
        self.assertTrue((action.abs() <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
