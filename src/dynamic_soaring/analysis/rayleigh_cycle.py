"""Classical Rayleigh cycle trajectory generation for baseline comparison."""

from __future__ import annotations

import math

import numpy as np

from dynamic_soaring.config import Config
from dynamic_soaring.physics.dynamics import rk4_step
from dynamic_soaring.physics.wind import create_wind_profile


def generate_rayleigh_cycle(
    config: Config,
    n_cycles: int = 5,
    z_low: float = 5.0,
    z_high: float = 30.0,
    bank_magnitude: float = 0.7,  # ~40 deg
    alpha_cruise: float = 0.08,  # ~4.5 deg
) -> np.ndarray:
    """Generate an idealized Rayleigh cycle trajectory.

    The Rayleigh cycle consists of 4 phases per cycle:
    1. Climb upwind into stronger wind (gaining airspeed)
    2. Turn downwind at high altitude
    3. Descend downwind in weaker wind
    4. Turn upwind at low altitude

    Returns (N, 6) trajectory array.
    """
    wind_profile = create_wind_profile(config.wind)
    wind_dir = config.wind.direction

    # Start conditions
    v0 = 14.0  # m/s initial airspeed
    heading = wind_dir + math.pi  # start heading into the wind
    z0 = (z_low + z_high) / 2

    state = np.array([
        0.0, 0.0, z0,
        v0 * math.cos(heading),
        v0 * math.sin(heading),
        0.0,
    ])

    trajectory = [state.copy()]
    dt = config.sim.dt

    for cycle in range(n_cycles):
        # Phase 1: Climb upwind (positive alpha for lift, shallow bank)
        state = _fly_phase(state, alpha_cruise + 0.03, 0.0, wind_profile, config,
                           lambda s: s[2] < z_high, trajectory)

        # Phase 2: Turn downwind at altitude (banked turn)
        target_heading = wind_dir
        state = _fly_phase(state, alpha_cruise, bank_magnitude, wind_profile, config,
                           lambda s: _heading_delta(s, target_heading) > 0.3, trajectory)

        # Phase 3: Descend downwind (reduced alpha)
        state = _fly_phase(state, alpha_cruise - 0.02, 0.0, wind_profile, config,
                           lambda s: s[2] > z_low, trajectory)

        # Phase 4: Turn upwind at low altitude
        target_heading = wind_dir + math.pi
        state = _fly_phase(state, alpha_cruise, -bank_magnitude, wind_profile, config,
                           lambda s: _heading_delta(s, target_heading) > 0.3, trajectory)

    return np.array(trajectory)


def _fly_phase(
    state: np.ndarray,
    alpha: float,
    bank: float,
    wind_profile,
    config: Config,
    continue_condition,
    trajectory: list,
    max_steps: int = 2000,
) -> np.ndarray:
    """Fly a single phase until condition is met."""
    for _ in range(max_steps):
        state, _ = rk4_step(state, alpha, bank, wind_profile, config.bird, config.sim)
        trajectory.append(state.copy())
        if state[2] <= 0.1:  # ground safety
            break
        if not continue_condition(state):
            break
    return state


def _heading_delta(state: np.ndarray, target: float) -> float:
    """Absolute heading difference (wrapped to [0, pi])."""
    heading = math.atan2(state[4], state[3])
    delta = abs(heading - target)
    if delta > math.pi:
        delta = 2 * math.pi - delta
    return delta
