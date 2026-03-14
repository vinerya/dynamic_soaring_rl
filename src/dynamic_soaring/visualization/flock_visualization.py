"""Multi-agent flock trajectory visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_multi_agent_trajectory(
    trajectories: list[np.ndarray],
    save_path: str | None = None,
) -> None:
    """3D plot of multiple agent trajectories."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.Set1(np.linspace(0, 1, len(trajectories)))

    for i, traj in enumerate(trajectories):
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=colors[i], linewidth=0.8, label=f"Agent {i}")
        ax.scatter(*traj[0, :3], color=colors[i], s=50, marker="o")
        ax.scatter(*traj[-1, :3], color=colors[i], s=50, marker="x")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title("Multi-Agent Soaring Trajectories")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_formation_snapshots(
    trajectories: list[np.ndarray],
    times: list[float],
    dt: float,
    save_path: str | None = None,
) -> None:
    """Top-down snapshots of flock formation at specified times."""
    n_times = len(times)
    fig, axes = plt.subplots(1, n_times, figsize=(4 * n_times, 4))
    if n_times == 1:
        axes = [axes]

    colors = plt.cm.Set1(np.linspace(0, 1, len(trajectories)))

    for ax, t in zip(axes, times):
        step = int(t / dt)
        for i, traj in enumerate(trajectories):
            if step < len(traj):
                ax.scatter(traj[step, 0], traj[step, 1],
                           c=[colors[i]], s=100, zorder=5, label=f"Agent {i}")
                # Trail (last 50 steps)
                s = max(0, step - 50)
                ax.plot(traj[s:step + 1, 0], traj[s:step + 1, 1],
                        color=colors[i], alpha=0.3, linewidth=1)

        ax.set_title(f"t = {t:.0f}s")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    axes[0].legend(fontsize=8)
    plt.suptitle("Formation Evolution")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_inter_agent_distances(
    trajectories: list[np.ndarray],
    dt: float,
    save_path: str | None = None,
) -> None:
    """Time series of pairwise inter-agent distances."""
    from dynamic_soaring.analysis.flock_analysis import compute_inter_agent_distances

    distances = compute_inter_agent_distances(trajectories)
    min_len = distances.shape[0]
    t = np.arange(min_len) * dt

    fig, ax = plt.subplots(figsize=(10, 5))

    n = len(trajectories)
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            ax.plot(t, distances[:, pair_idx], linewidth=0.8,
                    label=f"Agents {i}-{j}", alpha=0.7)
            pair_idx += 1

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Inter-Agent Distances Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
