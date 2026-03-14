"""Flight dynamics with RK4 integration."""

from __future__ import annotations

import numpy as np

from dynamic_soaring.config import BirdConfig, SimConfig
from dynamic_soaring.physics.aerodynamics import compute_aero_forces
from dynamic_soaring.physics.wind import WindProfile


def derivatives(
    state: np.ndarray,
    alpha: float,
    bank: float,
    wind_profile: WindProfile,
    bird: BirdConfig,
    sim: SimConfig,
    time: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Compute state derivatives [dx,dy,dz, dvx,dvy,dvz].

    Returns (d_state, info_dict) where info_dict contains forces and airspeed.
    """
    pos = state[:3]
    vel = state[3:]

    wind = wind_profile.get_wind(pos, time)
    F_lift, F_drag, airspeed, cl, cd = compute_aero_forces(
        vel, wind, alpha, bank, bird, sim.rho
    )

    # Gravity
    F_gravity = np.array([0.0, 0.0, -bird.mass * sim.g])

    # Total force and acceleration
    F_total = F_lift + F_drag + F_gravity
    acc = F_total / bird.mass

    d_state = np.concatenate([vel, acc])

    info = {
        "airspeed": airspeed,
        "cl": cl,
        "cd": cd,
        "lift_mag": np.linalg.norm(F_lift),
        "drag_mag": np.linalg.norm(F_drag),
        "wind": wind,
    }
    return d_state, info


def rk4_step(
    state: np.ndarray,
    alpha: float,
    bank: float,
    wind_profile: WindProfile,
    bird: BirdConfig,
    sim: SimConfig,
    time: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Fourth-order Runge-Kutta integration step.

    Returns (new_state, info) where info is from the final derivative evaluation.
    """
    dt = sim.dt

    k1, _ = derivatives(state, alpha, bank, wind_profile, bird, sim, time)
    k2, _ = derivatives(state + 0.5 * dt * k1, alpha, bank, wind_profile, bird, sim, time + 0.5 * dt)
    k3, _ = derivatives(state + 0.5 * dt * k2, alpha, bank, wind_profile, bird, sim, time + 0.5 * dt)
    k4, info = derivatives(state + dt * k3, alpha, bank, wind_profile, bird, sim, time + dt)

    new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Ground clamp
    if new_state[2] < 0.0:
        new_state[2] = 0.0
        new_state[5] = max(new_state[5], 0.0)  # no downward velocity at ground

    return new_state, info


def compute_total_energy(state: np.ndarray, wind: np.ndarray, mass: float, g: float) -> float:
    """Compute total specific energy: 0.5*V_air^2 + g*z."""
    v_air = state[3:] - wind
    airspeed = np.linalg.norm(v_air)
    return 0.5 * airspeed * airspeed + g * state[2]


def compute_load_factor(lift_mag: float, mass: float, g: float) -> float:
    """Load factor n = L / (m*g)."""
    return lift_mag / (mass * g)
