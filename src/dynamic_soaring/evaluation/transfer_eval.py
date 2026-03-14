"""Transfer and generalization evaluation across wind conditions."""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from dynamic_soaring.config import Config
from dynamic_soaring.envs.soaring_env import DynamicSoaringEnv
from dynamic_soaring.evaluation.metrics import compute_episode_stats


def evaluate_transfer(
    model_path: str,
    test_configs: list[tuple[str, Config]],
    n_episodes: int = 20,
    seed: int = 0,
) -> dict[str, dict]:
    """Evaluate a trained model across multiple wind conditions.

    Args:
        model_path: path to trained model .zip
        test_configs: list of (name, config) pairs
        n_episodes: episodes per condition
        seed: base seed

    Returns dict mapping condition name -> performance metrics.
    """
    model = PPO.load(model_path)
    results = {}

    for name, config in test_configs:
        env = DynamicSoaringEnv(config)
        rewards = []
        durations = []
        energy_changes = []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            total_reward = 0.0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            traj = np.array(env.trajectory)
            stats = compute_episode_stats(traj, config.sim.dt)
            rewards.append(total_reward)
            durations.append(stats["duration_s"])
            energy_changes.append(stats["energy_change"])

        results[name] = {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "duration_mean": float(np.mean(durations)),
            "energy_change_mean": float(np.mean(energy_changes)),
            "survival_rate": float(np.mean([d >= config.sim.max_episode_steps * config.sim.dt * 0.95 for d in durations])),
        }

    return results


def compute_robustness_curve(
    model_path: str,
    base_config: Config,
    param_path: str,
    param_range: np.ndarray,
    n_episodes: int = 10,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Sweep one parameter and measure performance degradation.

    Args:
        param_path: dot-separated path like "wind.reference_speed"
        param_range: array of values to test

    Returns dict with param_values, reward_means, reward_stds.
    """
    model = PPO.load(model_path)
    section, field = param_path.split(".")

    reward_means = []
    reward_stds = []

    for val in param_range:
        config = base_config.merge_overrides({section: {field: float(val)}})
        env = DynamicSoaringEnv(config)
        rewards = []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            total_reward = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)

        reward_means.append(float(np.mean(rewards)))
        reward_stds.append(float(np.std(rewards)))

    return {
        "param_values": param_range,
        "reward_means": np.array(reward_means),
        "reward_stds": np.array(reward_stds),
    }
