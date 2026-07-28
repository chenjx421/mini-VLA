from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from embodied_vla.models import StateActorCritic


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 100_000
    num_envs: int = 8
    rollout_steps: int = 256
    update_epochs: int = 8
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.005
    max_grad_norm: float = 0.5
    target_kl: float = 0.03

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.num_envs <= 0 or self.rollout_steps <= 0:
            raise ValueError("num_envs and rollout_steps must be positive")
        batch_size = self.num_envs * self.rollout_steps
        if self.minibatch_size > batch_size:
            raise ValueError("minibatch_size cannot exceed rollout batch size")


def compute_gae(
    rewards: NDArray[np.floating],
    values: NDArray[np.floating],
    bootstrap_values: NDArray[np.floating],
    episode_ends: NDArray[np.bool_],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute generalized advantage estimates.

    ``bootstrap_values[t]`` is zero for a true terminal transition, but is the
    critic value of the final observation for a time-limit truncation.
    ``episode_ends`` prevents GAE from leaking across environment resets.
    """

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    bootstrap_values = np.asarray(bootstrap_values, dtype=np.float32)
    episode_ends = np.asarray(episode_ends, dtype=np.bool_)
    if not (rewards.shape == values.shape == bootstrap_values.shape == episode_ends.shape):
        raise ValueError("all GAE inputs must have the same [time, env] shape")

    advantages = np.zeros_like(rewards, dtype=np.float32)
    running_advantage = np.zeros(rewards.shape[1], dtype=np.float32)
    for time_index in range(rewards.shape[0] - 1, -1, -1):
        delta = rewards[time_index] + gamma * bootstrap_values[time_index] - values[time_index]
        continuation = 1.0 - episode_ends[time_index].astype(np.float32)
        running_advantage = delta + gamma * gae_lambda * continuation * running_advantage
        advantages[time_index] = running_advantage
    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy: StateActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    observations: Tensor,
    raw_actions: Tensor,
    old_log_probabilities: Tensor,
    old_values: Tensor,
    advantages: Tensor,
    returns: Tensor,
    config: PPOConfig,
    generator: torch.Generator,
) -> dict[str, float]:
    batch_size = observations.shape[0]
    normalized_advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    metric_sums = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "gradient_norm": 0.0,
    }
    update_count = 0
    stop_early = False

    for _ in range(config.update_epochs):
        permutation = torch.randperm(batch_size, generator=generator)
        for start in range(0, batch_size, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            new_log_probability, entropy, new_value = policy.evaluate_raw_action(
                observations[indices],
                raw_actions[indices],
            )
            log_ratio = new_log_probability - old_log_probabilities[indices]
            ratio = log_ratio.exp()
            unclipped_objective = ratio * normalized_advantages[indices]
            clipped_objective = (
                ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
                * normalized_advantages[indices]
            )
            policy_loss = -torch.minimum(
                unclipped_objective,
                clipped_objective,
            ).mean()

            value_delta = new_value - old_values[indices]
            clipped_value = old_values[indices] + value_delta.clamp(
                -config.value_clip_ratio,
                config.value_clip_ratio,
            )
            value_loss_unclipped = (new_value - returns[indices]).square()
            value_loss_clipped = (clipped_value - returns[indices]).square()
            value_loss = (
                0.5
                * torch.maximum(
                    value_loss_unclipped,
                    value_loss_clipped,
                ).mean()
            )
            entropy_mean = entropy.mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy_mean
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                config.max_grad_norm,
            )
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > config.clip_ratio).float().mean()
            values_to_add = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy_mean,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "gradient_norm": gradient_norm,
            }
            for name, value in values_to_add.items():
                metric_sums[name] += float(value.detach().cpu())
            update_count += 1
            if float(approx_kl) > config.target_kl:
                stop_early = True
                break
        if stop_early:
            break

    metrics = {name: value / max(1, update_count) for name, value in metric_sums.items()}
    metrics["epochs_stopped_early"] = float(stop_early)
    return metrics
