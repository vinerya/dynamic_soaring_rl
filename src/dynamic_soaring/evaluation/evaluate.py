"""Evaluate trained dynamic soaring policies."""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from dynamic_soaring.config import Config
from dynamic_soaring.envs.soaring_env import DynamicSoaringEnv
from dynamic_soaring.evaluation.metrics import compute_episode_stats


def evaluate_policy(
    model_path: str,
    config: Config | None = None,
    n_episodes: int = 20,
    deterministic: bool = True,
    seed: int = 0,
    return_trajectories: bool = False,
) -> dict:
    """Evaluate a trained model over multiple episodes.

    Returns aggregate statistics across episodes.
    """
    config = config or Config()
    env = DynamicSoaringEnv(config)
    model = PPO.load(model_path)

    all_stats = []
    all_rewards = []
    all_trajectories = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        trajectory = np.array(env.trajectory)
        if return_trajectories:
            all_trajectories.append(trajectory)
        stats = compute_episode_stats(trajectory, config.sim.dt)
        stats["total_reward"] = total_reward
        stats["termination_reason"] = info.get("termination_reason", "truncated")
        all_stats.append(stats)
        all_rewards.append(total_reward)

    # Aggregate
    agg = {
        "n_episodes": n_episodes,
        "reward_mean": float(np.mean(all_rewards)),
        "reward_std": float(np.std(all_rewards)),
        "duration_mean": float(np.mean([s["duration_s"] for s in all_stats])),
        "duration_std": float(np.std([s["duration_s"] for s in all_stats])),
        "altitude_max_mean": float(np.mean([s["altitude_max"] for s in all_stats])),
        "energy_change_mean": float(np.mean([s["energy_change"] for s in all_stats])),
        "energy_change_std": float(np.std([s["energy_change"] for s in all_stats])),
        "soaring_cycles_mean": float(np.mean([s["n_soaring_cycles"] for s in all_stats])),
        "success_rate": float(
            sum(1 for s in all_stats if s["termination_reason"] == "truncated") / n_episodes
        ),
        "termination_reasons": {
            reason: sum(1 for s in all_stats if s["termination_reason"] == reason)
            for reason in set(s["termination_reason"] for s in all_stats)
        },
        "episodes": all_stats,
    }

    if return_trajectories:
        agg["trajectories"] = all_trajectories

    return agg


def print_evaluation_report(results: dict) -> None:
    """Print formatted evaluation results."""
    print("\n" + "=" * 60)
    print("DYNAMIC SOARING EVALUATION REPORT")
    print("=" * 60)
    print(f"Episodes: {results['n_episodes']}")
    print(f"Reward:   {results['reward_mean']:.2f} +/- {results['reward_std']:.2f}")
    print(f"Duration: {results['duration_mean']:.1f}s +/- {results['duration_std']:.1f}s")
    print(f"Success:  {results['success_rate'] * 100:.1f}% (survived full episode)")
    print(f"Max Alt:  {results['altitude_max_mean']:.1f}m (mean)")
    print(f"Energy:   {results['energy_change_mean']:.1f} +/- {results['energy_change_std']:.1f}")
    print(f"Cycles:   {results['soaring_cycles_mean']:.1f} (mean soaring cycles)")
    print(f"\nTermination reasons:")
    for reason, count in results["termination_reasons"].items():
        print(f"  {reason}: {count}")
    print("=" * 60)
