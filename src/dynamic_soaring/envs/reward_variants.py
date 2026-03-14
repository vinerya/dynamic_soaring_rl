"""Alternative reward functions for ablation studies."""

from __future__ import annotations

import math

import numpy as np

from dynamic_soaring.config import RewardConfig


def compute_reward_energy_only(
    old_energy: float, new_energy: float,
    action: np.ndarray, prev_action: np.ndarray | None,
    terminated: bool, termination_reason: str | None,
    config: RewardConfig,
) -> float:
    """Energy change only -- no survival, smoothness, or terminal penalties."""
    return config.energy_weight * (new_energy - old_energy) / config.energy_scale


def compute_reward_survival_heavy(
    old_energy: float, new_energy: float,
    action: np.ndarray, prev_action: np.ndarray | None,
    terminated: bool, termination_reason: str | None,
    config: RewardConfig,
) -> float:
    """Heavy survival bonus (10x default), reduced energy weight."""
    reward = 0.3 * (new_energy - old_energy) / config.energy_scale
    reward += 0.1  # 10x default survival bonus
    if terminated:
        reward += config.crash_penalty
    return reward


def compute_reward_altitude_band(
    old_energy: float, new_energy: float,
    action: np.ndarray, prev_action: np.ndarray | None,
    terminated: bool, termination_reason: str | None,
    config: RewardConfig,
    altitude: float = 0.0,
) -> float:
    """Bonus for staying in the productive altitude band (5-50m)."""
    from dynamic_soaring.envs.rewards import compute_reward
    reward = compute_reward(old_energy, new_energy, action, prev_action,
                            terminated, termination_reason, config)
    # Gaussian bonus centered at 20m
    reward += 0.02 * math.exp(-((altitude - 20.0) / 30.0) ** 2)
    return reward


def compute_reward_cycle_bonus(
    old_energy: float, new_energy: float,
    action: np.ndarray, prev_action: np.ndarray | None,
    terminated: bool, termination_reason: str | None,
    config: RewardConfig,
    altitude_history: list[float] | None = None,
) -> float:
    """Bonus for completing soaring cycles (detected from altitude oscillations)."""
    from dynamic_soaring.envs.rewards import compute_reward
    reward = compute_reward(old_energy, new_energy, action, prev_action,
                            terminated, termination_reason, config)

    # Detect cycle completion from altitude history
    if altitude_history and len(altitude_history) > 20:
        recent = np.array(altitude_history[-50:])
        dz = np.diff(recent)
        sign_changes = np.sum(np.diff(np.sign(dz)) != 0)
        if sign_changes >= 2:  # at least one complete oscillation
            amp = np.max(recent) - np.min(recent)
            if amp > 5.0:  # minimum 5m amplitude
                reward += 0.05  # cycle completion bonus

    return reward


# Registry for selecting reward variants by name
REWARD_VARIANTS = {
    "default": None,  # uses the standard compute_reward
    "energy_only": compute_reward_energy_only,
    "survival_heavy": compute_reward_survival_heavy,
    "altitude_band": compute_reward_altitude_band,
    "cycle_bonus": compute_reward_cycle_bonus,
}
