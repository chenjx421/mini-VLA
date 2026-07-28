from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio

from embodied_vla.envs import SOArmEnvConfig, SOArmPickPlaceEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the SO-ARM100 environment.")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--save-frame", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SOArmEnvConfig(
        observation_mode="multimodal",
        image_size=args.image_size,
    )
    env = SOArmPickPlaceEnv(config)
    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            if episode == 0 and args.save_frame is not None:
                args.save_frame.parent.mkdir(parents=True, exist_ok=True)
                iio.imwrite(args.save_frame, observation["rgb"])
            total_reward = 0.0
            for _ in range(args.steps):
                observation, reward, terminated, truncated, info = env.step(
                    env.action_space.sample()
                )
                total_reward += reward
                if terminated or truncated:
                    break
            print(
                f"episode={episode} instruction={info['instruction']!r} "
                f"steps={info['step']} reward={total_reward:.3f} "
                f"success={info['success']}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
