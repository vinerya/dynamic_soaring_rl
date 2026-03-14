"""SB3 PPO training pipeline for dynamic soaring."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from dynamic_soaring.config import Config
from dynamic_soaring.envs.soaring_env import DynamicSoaringEnv
from dynamic_soaring.training.callbacks import SoaringMetricsCallback


def make_env_fn(config: Config, seed: int):
    """Create a factory function for environment creation."""
    def _init():
        env = DynamicSoaringEnv(config)
        env.reset(seed=seed)
        return env
    return _init


def train(config: Config) -> PPO:
    """Run PPO training with the given configuration."""
    tc = config.training

    # Create directories
    Path(tc.log_dir).mkdir(parents=True, exist_ok=True)
    Path(tc.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Save config used for this run
    config.to_yaml(os.path.join(tc.log_dir, "config.yaml"))

    # Vectorized training environments
    env_fns = [make_env_fn(config, tc.seed + i) for i in range(tc.n_envs)]
    vec_env = SubprocVecEnv(env_fns)

    if tc.normalize_obs or tc.normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=tc.normalize_obs,
            norm_reward=tc.normalize_reward,
            clip_obs=10.0,
            clip_reward=10.0,
        )

    # Evaluation environment (wrapped consistently with training env)
    eval_env = DummyVecEnv([make_env_fn(config, tc.seed + 1000)])
    if tc.normalize_obs or tc.normalize_reward:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=tc.normalize_obs,
            norm_reward=False,  # don't normalize eval rewards
            clip_obs=10.0,
        )

    # PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=tc.learning_rate,
        n_steps=tc.n_steps,
        batch_size=tc.batch_size,
        n_epochs=tc.n_epochs,
        gamma=tc.gamma,
        gae_lambda=tc.gae_lambda,
        clip_range=tc.clip_range,
        policy_kwargs={
            "net_arch": dict(pi=tc.net_arch, vf=tc.net_arch),
            "activation_fn": torch.nn.ReLU,
        },
        verbose=1,
        tensorboard_log=tc.log_dir,
        seed=tc.seed,
    )

    # Callbacks
    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            eval_freq=max(tc.eval_freq // tc.n_envs, 1),
            n_eval_episodes=tc.eval_episodes,
            best_model_save_path=tc.checkpoint_dir,
            log_path=tc.log_dir,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=max(tc.checkpoint_freq // tc.n_envs, 1),
            save_path=tc.checkpoint_dir,
            name_prefix="ppo_soaring",
        ),
        SoaringMetricsCallback(),
    ])

    # Train
    model.learn(total_timesteps=tc.total_timesteps, callback=callbacks)

    # Save final model and normalization stats
    model.save(os.path.join(tc.checkpoint_dir, "final_model"))
    if isinstance(vec_env, VecNormalize):
        vec_env.save(os.path.join(tc.checkpoint_dir, "vec_normalize.pkl"))

    vec_env.close()
    return model
