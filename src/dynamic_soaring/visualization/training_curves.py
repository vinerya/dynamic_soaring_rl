"""Plot training metrics from TensorBoard logs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(log_dir: str, save_path: str | None = None) -> None:
    """Plot training curves from SB3 monitor logs.

    Reads the evaluations.npz file produced by EvalCallback.
    """
    eval_path = Path(log_dir) / "evaluations.npz"
    if not eval_path.exists():
        print(f"No evaluations found at {eval_path}")
        return

    data = np.load(eval_path)
    timesteps = data["timesteps"]
    results = data["results"]  # (n_evals, n_episodes)

    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reward curve
    ax = axes[0]
    ax.plot(timesteps, mean_rewards, "b-", linewidth=1.5)
    ax.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
    )
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Evaluation Reward")
    ax.set_title("Training Progress")
    ax.grid(True, alpha=0.3)

    # Episode lengths if available
    if "ep_lengths" in data:
        ep_lengths = data["ep_lengths"]
        mean_lengths = np.mean(ep_lengths, axis=1)
        ax2 = axes[1]
        ax2.plot(timesteps, mean_lengths, "g-", linewidth=1.5)
        ax2.set_xlabel("Timesteps")
        ax2.set_ylabel("Mean Episode Length")
        ax2.set_title("Episode Duration")
        ax2.grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
