from __future__ import annotations

import argparse
from pathlib import Path

from embodied_vla.algorithms import PPOConfig
from embodied_vla.envs import SOArmEnvConfig
from embodied_vla.training.ppo_trainer import train_ppo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the from-scratch PPO baseline.")
    parser.add_argument("--task", choices=("reach", "pick", "pick_place"), default="reach")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ppo_reach"))
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    max_episode_steps = {"reach": 100, "pick": 180, "pick_place": 300}[args.task]
    env_config = SOArmEnvConfig(
        observation_mode="state",
        task_level=args.task,
        grasp_mode="contact_assisted",
        max_episode_steps=max_episode_steps,
    )
    ppo_config = PPOConfig(
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
    )
    train_ppo(
        env_config,
        ppo_config,
        output_dir=args.output_dir,
        seed=args.seed,
        eval_episodes=args.eval_episodes,
        device=args.device,
    )


if __name__ == "__main__":
    main()
