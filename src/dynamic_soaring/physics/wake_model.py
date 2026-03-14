"""Simplified wake/vortex interaction model between birds."""

from __future__ import annotations

import math

import numpy as np

from dynamic_soaring.config import BirdConfig


def compute_circulation(airspeed: float, lift_coeff: float, bird: BirdConfig) -> float:
    """Compute vortex circulation from Kutta-Joukowski: Gamma = L / (rho * V * b/2).

    Simplified: Gamma ~ Cl * V * S / b.
    """
    return lift_coeff * airspeed * bird.wing_area / bird.wing_span


def compute_wake_velocity(
    leader_pos: np.ndarray,
    leader_vel: np.ndarray,
    leader_circulation: float,
    follower_pos: np.ndarray,
    bird: BirdConfig,
    decay_rate: float = 0.1,
) -> np.ndarray:
    """Compute wake-induced velocity at follower position due to leader.

    Uses a simplified horseshoe vortex model:
    - Two trailing vortices behind the leader
    - Upwash to the sides, downwash directly behind
    - Exponential decay with downstream distance

    Returns 3D velocity perturbation at follower position.
    """
    # Displacement from leader to follower
    dx = follower_pos - leader_pos

    # Leader flight direction
    speed = np.linalg.norm(leader_vel[:2])
    if speed < 1e-6:
        return np.zeros(3)
    flight_dir = leader_vel[:2] / speed

    # Transform to leader's frame: along-track and cross-track
    along = np.dot(dx[:2], flight_dir)  # positive = behind leader
    cross = dx[0] * (-flight_dir[1]) + dx[1] * flight_dir[0]  # perpendicular
    dz = dx[2]

    # Wake only affects positions behind the leader
    if along < 0:
        return np.zeros(3)

    # Downstream decay
    downstream_factor = math.exp(-decay_rate * along)

    # Horseshoe vortex: two point vortices at +/- b/2
    b_half = bird.wing_span / 2.0
    gamma = leader_circulation

    # Induced velocity from two trailing vortices (Biot-Savart simplified)
    w_z = 0.0
    for sign in [1.0, -1.0]:
        y_vortex = sign * b_half
        r_sq = (cross - y_vortex) ** 2 + dz ** 2
        r_sq = max(r_sq, 0.5 ** 2)  # core radius limit
        # Vertical velocity from this vortex line
        w_z += sign * gamma / (2 * math.pi) * (cross - y_vortex) / r_sq

    w_z *= downstream_factor

    return np.array([0.0, 0.0, w_z])


def compute_all_wake_effects(
    states: np.ndarray,
    circulations: np.ndarray,
    bird: BirdConfig,
    decay_rate: float = 0.1,
) -> np.ndarray:
    """Compute wake-induced velocities for all agents.

    Args:
        states: (N, 6) array of agent states
        circulations: (N,) array of current circulations
        bird: bird configuration
        decay_rate: wake decay rate

    Returns (N, 3) array of wake-induced velocity perturbations.
    """
    n = len(states)
    wake_velocities = np.zeros((n, 3))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = compute_wake_velocity(
                states[j, :3], states[j, 3:],
                circulations[j],
                states[i, :3],
                bird, decay_rate,
            )
            wake_velocities[i] += w

    return wake_velocities
