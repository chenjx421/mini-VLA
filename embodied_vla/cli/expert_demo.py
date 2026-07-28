from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio

from embodied_vla.envs import SOArmEnvConfig, SOArmPickPlaceEnv
from embodied_vla.experts import PickPlaceExpert


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the privileged pick-place expert.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--save-gif", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SOArmEnvConfig(
        observation_mode="multimodal",
        image_size=args.image_size,
        max_episode_steps=350,
    )
    env = SOArmPickPlaceEnv(config)
    expert = PickPlaceExpert(config)
    successes = 0
    saved_frames = []
    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            expert.reset()
            total_reward = 0.0
            for step in range(config.max_episode_steps):
                action = expert.act(info)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if episode == 0 and args.save_gif is not None and step % 2 == 0:
                    saved_frames.append(observation["rgb"])
                if terminated or truncated:
                    break
            successes += int(info["success"])
            print(
                f"episode={episode:03d} seed={args.seed + episode} "
                f"task={info['target_color']}->{info['goal_side']} "
                f"phase={expert.phase.name} steps={info['step']} retries={expert.retries} "
                f"reward={total_reward:.3f} success={info['success']} "
                f"reason={info['termination_reason']}"
            )
    finally:
        env.close()

    if args.save_gif is not None and saved_frames:
        args.save_gif.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(args.save_gif, saved_frames, duration=40, loop=0)
    print(f"success_rate={successes}/{args.episodes}={successes / args.episodes:.1%}")


if __name__ == "__main__":
    main()
