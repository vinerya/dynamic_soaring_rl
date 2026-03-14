#!/usr/bin/env python
"""Train multi-agent soaring with parameter-sharing IPPO."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dynamic_soaring.config import Config
from dynamic_soaring.envs.multi_agent_env import MultiAgentSoaringEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-agent soaring")
    parser.add_argument("--config", type=str, default="configs/multi_agent.yaml")
    parser.add_argument("--n-agents", type=int, default=None)
    parser.add_argument("--comm", type=str, default=None,
                        choices=["no_comm", "shared_obs", "full_comm"])
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints/multi_agent/")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.n_agents:
        config.multi_agent.n_agents = args.n_agents
    if args.comm:
        config.multi_agent.comm_variant = args.comm
    timesteps = args.timesteps or config.training.total_timesteps

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {config.multi_agent.n_agents} agents, comm={config.multi_agent.comm_variant}")

    def make_env():
        return MultiAgentSoaringEnv(config)

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        n_steps=config.training.n_steps,
        n_epochs=config.training.n_epochs,
        clip_range=config.training.clip_range,
        seed=config.training.seed,
        verbose=1,
        tensorboard_log=str(save_dir / "tb_logs"),
    )

    model.learn(total_timesteps=timesteps)

    model_path = save_dir / f"multi_agent_{config.multi_agent.n_agents}_{config.multi_agent.comm_variant}"
    model.save(str(model_path))
    env.save(str(model_path) + "_vecnorm.pkl")
    config.to_yaml(str(model_path) + "_config.yaml")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
