from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from embodied_vla.algorithms import PPOConfig, compute_gae, ppo_update
from embodied_vla.envs import SOArmEnvConfig, SOArmPickPlaceEnv
from embodied_vla.metrics import wilson_score_interval
from embodied_vla.models import StateActorCritic
from embodied_vla.training.run_guard import claim_run_directory


def evaluate_state_policy(
    policy: StateActorCritic,
    env_config: SOArmEnvConfig,
    *,
    episodes: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    env = SOArmPickPlaceEnv(env_config)
    successes = 0
    returns: list[float] = []
    lengths: list[int] = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            for _ in range(env_config.max_episode_steps):
                observation_tensor = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                action = (
                    policy.act(observation_tensor, deterministic=True)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            successes += int(info["success"])
            returns.append(episode_return)
            lengths.append(info["step"])
    finally:
        env.close()
    confidence_low, confidence_high = wilson_score_interval(successes, episodes)
    return {
        "success_rate": successes / episodes,
        "successes": float(successes),
        "success_wilson_95_low": confidence_low,
        "success_wilson_95_high": confidence_high,
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "episodes": float(episodes),
    }


def train_ppo(
    env_config: SOArmEnvConfig,
    ppo_config: PPOConfig,
    *,
    output_dir: Path,
    seed: int,
    eval_episodes: int = 20,
    eval_interval_updates: int = 5,
    device: str = "cpu",
) -> dict[str, Any]:
    with claim_run_directory(output_dir):
        return _train_ppo_in_claimed_directory(
            env_config,
            ppo_config,
            output_dir=output_dir,
            seed=seed,
            eval_episodes=eval_episodes,
            eval_interval_updates=eval_interval_updates,
            device=device,
        )


def _train_ppo_in_claimed_directory(
    env_config: SOArmEnvConfig,
    ppo_config: PPOConfig,
    *,
    output_dir: Path,
    seed: int,
    eval_episodes: int,
    eval_interval_updates: int,
    device: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    envs = [SOArmPickPlaceEnv(env_config) for _ in range(ppo_config.num_envs)]
    observations = np.stack(
        [env.reset(seed=seed + index)[0] for index, env in enumerate(envs)]
    )
    policy = StateActorCritic(
        observation_dim=SOArmPickPlaceEnv.STATE_DIM,
        action_dim=envs[0].action_space.shape[0],
    ).to(torch_device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo_config.learning_rate, eps=1e-5)
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(seed)

    batch_shape = (ppo_config.rollout_steps, ppo_config.num_envs)
    observation_buffer = np.zeros(
        (*batch_shape, SOArmPickPlaceEnv.STATE_DIM),
        dtype=np.float32,
    )
    raw_action_buffer = np.zeros((*batch_shape, 5), dtype=np.float32)
    log_probability_buffer = np.zeros(batch_shape, dtype=np.float32)
    value_buffer = np.zeros(batch_shape, dtype=np.float32)
    reward_buffer = np.zeros(batch_shape, dtype=np.float32)
    bootstrap_value_buffer = np.zeros(batch_shape, dtype=np.float32)
    episode_end_buffer = np.zeros(batch_shape, dtype=np.bool_)

    completed_returns: list[float] = []
    completed_successes: list[float] = []
    running_returns = np.zeros(ppo_config.num_envs, dtype=np.float64)
    best_success_rate = -1.0
    global_step = 0
    update_index = 0
    start_time = time.perf_counter()
    metrics_path = output_dir / "metrics.jsonl"

    try:
        while global_step < ppo_config.total_steps:
            policy.eval()
            for time_index in range(ppo_config.rollout_steps):
                observation_buffer[time_index] = observations
                observation_tensor = torch.as_tensor(
                    observations,
                    dtype=torch.float32,
                    device=torch_device,
                )
                with torch.no_grad():
                    actions, raw_actions, log_probabilities, values = policy.sample(
                        observation_tensor
                    )
                action_array = actions.cpu().numpy()
                raw_action_buffer[time_index] = raw_actions.cpu().numpy()
                log_probability_buffer[time_index] = log_probabilities.cpu().numpy()
                value_buffer[time_index] = values.cpu().numpy()

                next_observations = []
                final_observations: list[NDArray[np.float32]] = []
                terminated_flags = np.zeros(ppo_config.num_envs, dtype=np.bool_)
                truncated_flags = np.zeros(ppo_config.num_envs, dtype=np.bool_)
                for env_index, env in enumerate(envs):
                    next_observation, reward, terminated, truncated, info = env.step(
                        action_array[env_index]
                    )
                    reward_buffer[time_index, env_index] = reward
                    running_returns[env_index] += reward
                    terminated_flags[env_index] = terminated
                    truncated_flags[env_index] = truncated
                    final_observations.append(next_observation)
                    if terminated or truncated:
                        completed_returns.append(float(running_returns[env_index]))
                        completed_successes.append(float(info["success"]))
                        running_returns[env_index] = 0.0
                        next_observation, _ = env.reset()
                    next_observations.append(next_observation)

                final_tensor = torch.as_tensor(
                    np.stack(final_observations),
                    dtype=torch.float32,
                    device=torch_device,
                )
                with torch.no_grad():
                    _, final_values = policy.distribution_and_value(final_tensor)
                final_value_array = final_values.cpu().numpy()
                final_value_array[terminated_flags] = 0.0
                bootstrap_value_buffer[time_index] = final_value_array
                episode_end_buffer[time_index] = terminated_flags | truncated_flags
                observations = np.stack(next_observations)
                global_step += ppo_config.num_envs

            advantages, returns = compute_gae(
                reward_buffer,
                value_buffer,
                bootstrap_value_buffer,
                episode_end_buffer,
                gamma=ppo_config.gamma,
                gae_lambda=ppo_config.gae_lambda,
            )
            policy.train()
            flat_observations = torch.as_tensor(
                observation_buffer.reshape(-1, SOArmPickPlaceEnv.STATE_DIM),
                dtype=torch.float32,
                device=torch_device,
            )
            flat_raw_actions = torch.as_tensor(
                raw_action_buffer.reshape(-1, 5),
                dtype=torch.float32,
                device=torch_device,
            )
            update_metrics = ppo_update(
                policy,
                optimizer,
                observations=flat_observations,
                raw_actions=flat_raw_actions,
                old_log_probabilities=torch.as_tensor(
                    log_probability_buffer.reshape(-1),
                    dtype=torch.float32,
                    device=torch_device,
                ),
                old_values=torch.as_tensor(
                    value_buffer.reshape(-1),
                    dtype=torch.float32,
                    device=torch_device,
                ),
                advantages=torch.as_tensor(
                    advantages.reshape(-1),
                    dtype=torch.float32,
                    device=torch_device,
                ),
                returns=torch.as_tensor(
                    returns.reshape(-1),
                    dtype=torch.float32,
                    device=torch_device,
                ),
                config=ppo_config,
                generator=torch_generator,
            )
            update_index += 1
            elapsed = time.perf_counter() - start_time
            recent_return = float(np.mean(completed_returns[-100:])) if completed_returns else 0.0
            recent_success = (
                float(np.mean(completed_successes[-100:])) if completed_successes else 0.0
            )
            record: dict[str, Any] = {
                "update": update_index,
                "global_step": global_step,
                "steps_per_second": global_step / max(elapsed, 1e-6),
                "train_recent_return": recent_return,
                "train_recent_success_rate": recent_success,
                **update_metrics,
            }

            if update_index % eval_interval_updates == 0 or global_step >= ppo_config.total_steps:
                policy.eval()
                evaluation = evaluate_state_policy(
                    policy,
                    env_config,
                    episodes=eval_episodes,
                    seed=seed + 100_000,
                    device=torch_device,
                )
                record.update({f"eval_{name}": value for name, value in evaluation.items()})
                if evaluation["success_rate"] > best_success_rate:
                    best_success_rate = evaluation["success_rate"]
                    _save_checkpoint(
                        checkpoint_dir / "best.pt",
                        policy,
                        optimizer,
                        env_config,
                        ppo_config,
                        seed,
                        global_step,
                        evaluation,
                    )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            print(
                f"update={update_index:03d} step={global_step:07d} "
                f"train_success={recent_success:.1%} "
                f"eval_success={record.get('eval_success_rate', float('nan')):.1%} "
                f"return={recent_return:.3f} sps={record['steps_per_second']:.0f}"
            )
    finally:
        for env in envs:
            env.close()

    final_evaluation = evaluate_state_policy(
        policy,
        env_config,
        episodes=eval_episodes,
        seed=seed + 200_000,
        device=torch_device,
    )
    _save_checkpoint(
        checkpoint_dir / "last.pt",
        policy,
        optimizer,
        env_config,
        ppo_config,
        seed,
        global_step,
        final_evaluation,
    )
    summary = {
        "seed": seed,
        "global_step": global_step,
        "best_eval_success_rate": best_success_rate,
        "final_evaluation": final_evaluation,
        "elapsed_seconds": time.perf_counter() - start_time,
        "environment": asdict(env_config),
        "ppo": asdict(ppo_config),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def _save_checkpoint(
    path: Path,
    policy: StateActorCritic,
    optimizer: torch.optim.Optimizer,
    env_config: SOArmEnvConfig,
    ppo_config: PPOConfig,
    seed: int,
    global_step: int,
    evaluation: dict[str, float],
) -> None:
    torch.save(
        {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "environment": asdict(env_config),
            "ppo": asdict(ppo_config),
            "seed": seed,
            "global_step": global_step,
            "evaluation": evaluation,
        },
        path,
    )
