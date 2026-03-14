"""Phase space and strategy comparison visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_phase_portrait(
    data: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str | None = None,
    color_by: np.ndarray | None = None,
) -> None:
    """Plot a 2D phase portrait."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if color_by is not None:
        scatter = ax.scatter(data[:, 0], data[:, 1], c=color_by, cmap="viridis", s=1, alpha=0.5)
        fig.colorbar(scatter, ax=ax)
    else:
        ax.plot(data[:, 0], data[:, 1], "b-", linewidth=0.5, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_trajectory_comparison(
    traj_agent: np.ndarray,
    traj_baseline: np.ndarray,
    labels: tuple[str, str] = ("RL Agent", "Rayleigh Cycle"),
    save_path: str | None = None,
) -> None:
    """Side-by-side 3D trajectory + phase portrait comparison."""
    fig = plt.figure(figsize=(16, 10))

    # 3D trajectories
    ax1 = fig.add_subplot(231, projection="3d")
    ax1.plot(*traj_agent[:, :3].T, "b-", linewidth=0.8, label=labels[0])
    ax1.set_title(labels[0])
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(232, projection="3d")
    ax2.plot(*traj_baseline[:, :3].T, "r-", linewidth=0.8, label=labels[1])
    ax2.set_title(labels[1])
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

    # Top-down overlay
    ax3 = fig.add_subplot(233)
    ax3.plot(traj_agent[:, 0], traj_agent[:, 1], "b-", alpha=0.7, label=labels[0])
    ax3.plot(traj_baseline[:, 0], traj_baseline[:, 1], "r-", alpha=0.7, label=labels[1])
    ax3.set_xlabel("X (m)"); ax3.set_ylabel("Y (m)")
    ax3.set_title("Top-down View"); ax3.legend(); ax3.grid(True, alpha=0.3)

    # Altitude vs time
    ax4 = fig.add_subplot(234)
    t_a = np.arange(len(traj_agent)) * 0.02
    t_b = np.arange(len(traj_baseline)) * 0.02
    ax4.plot(t_a, traj_agent[:, 2], "b-", label=labels[0])
    ax4.plot(t_b, traj_baseline[:, 2], "r-", label=labels[1])
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Altitude (m)")
    ax4.legend(); ax4.grid(True, alpha=0.3)

    # Speed comparison
    ax5 = fig.add_subplot(235)
    speed_a = np.linalg.norm(traj_agent[:, 3:], axis=1)
    speed_b = np.linalg.norm(traj_baseline[:, 3:], axis=1)
    ax5.plot(t_a, speed_a, "b-", label=labels[0])
    ax5.plot(t_b, speed_b, "r-", label=labels[1])
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Ground Speed (m/s)")
    ax5.legend(); ax5.grid(True, alpha=0.3)

    # Energy
    ax6 = fig.add_subplot(236)
    g = 9.81
    e_a = 0.5 * speed_a**2 + g * traj_agent[:, 2]
    e_b = 0.5 * speed_b**2 + g * traj_baseline[:, 2]
    ax6.plot(t_a, e_a, "b-", label=labels[0])
    ax6.plot(t_b, e_b, "r-", label=labels[1])
    ax6.set_xlabel("Time (s)"); ax6.set_ylabel("Specific Energy (J/kg)")
    ax6.legend(); ax6.grid(True, alpha=0.3)

    plt.suptitle("Strategy Comparison", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_energy_budget(budget: dict, save_path: str | None = None) -> None:
    """Bar chart of energy budget decomposition."""
    fig, ax = plt.subplots(figsize=(8, 5))
    components = ["wind_extraction", "drag_loss", "kinetic_change", "potential_change", "total_change"]
    labels = ["Wind\nExtraction", "Drag\nLoss", "KE\nChange", "PE\nChange", "Total\nChange"]
    values = [budget.get(c, 0) for c in components]
    colors = ["green", "red", "blue", "orange", "purple"]

    bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Energy (J/kg)")
    ax.set_title("Energy Budget Decomposition")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
