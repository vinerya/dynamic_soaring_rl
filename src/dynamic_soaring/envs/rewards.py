"""Reward functions for dynamic soaring."""

from __future__ import annotations

import numpy as np

from dynamic_soaring.config import RewardConfig


def compute_reward(
    old_energy: float,
    new_energy: float,
    action: np.ndarray,
    prev_action: np.ndarray | None,
    terminated: bool,
    termination_reason: str | None,
    config: RewardConfig,
) -> float:
    """Compute step reward for dynamic soaring.

    Components:
        - Energy change (primary): encourages total energy gain
        - Survival bonus: small per-step reward for staying aloft
        - Smoothness penalty: discourages jerky control inputs
        - Terminal penalties: large negative for crash/stall
    """
    reward = 0.0

    # Energy change (normalized)
    delta_e = new_energy - old_energy
    reward += config.energy_weight * delta_e / config.energy_scale

    # Survival bonus
    reward += config.survival_bonus

    # Smoothness penalty
    if prev_action is not None:
        delta_action = action - prev_action
        reward -= config.smoothness_weight * float(np.sum(delta_action ** 2))

    # Terminal penalties
    if terminated:
        if termination_reason == "crash":
            reward += config.crash_penalty
        elif termination_reason == "stall":
            reward += config.stall_penalty
        elif termination_reason == "altitude":
            reward += config.crash_penalty * 0.5

    return reward
