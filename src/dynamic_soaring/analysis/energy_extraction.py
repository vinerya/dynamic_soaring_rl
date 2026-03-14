"""Energy extraction analysis for dynamic soaring."""

from __future__ import annotations

import numpy as np

from dynamic_soaring.config import Config
from dynamic_soaring.physics.aerodynamics import compute_aero_forces
from dynamic_soaring.physics.wind import create_wind_profile, WindProfile


def compute_energy_extraction_rate(
    trajectory: np.ndarray,
    config: Config,
    wind_profile: WindProfile | None = None,
) -> np.ndarray:
    """Compute per-step energy extraction rate from wind (J/kg/s).

    Energy extraction = d(total_specific_energy)/dt.
    """
    wp = wind_profile or create_wind_profile(config.wind)
    g = config.sim.g
    dt = config.sim.dt

    n = len(trajectory)
    rates = np.zeros(n - 1)

    for i in range(n - 1):
        pos_old, vel_old = trajectory[i, :3], trajectory[i, 3:]
        pos_new, vel_new = trajectory[i + 1, :3], trajectory[i + 1, 3:]

        wind_old = wp.get_wind(pos_old)
        wind_new = wp.get_wind(pos_new)

        v_air_old = np.linalg.norm(vel_old - wind_old)
        v_air_new = np.linalg.norm(vel_new - wind_new)

        E_old = 0.5 * v_air_old ** 2 + g * pos_old[2]
        E_new = 0.5 * v_air_new ** 2 + g * pos_new[2]

        rates[i] = (E_new - E_old) / dt

    return rates


def compute_energy_efficiency(
    trajectory: np.ndarray,
    config: Config,
) -> float:
    """Compute energy efficiency: net energy gained / total drag work.

    Values > 0 mean the bird gains more from wind than it loses to drag.
    Returns ratio (can be negative if bird is losing energy).
    """
    wp = create_wind_profile(config.wind)
    g = config.sim.g
    dt = config.sim.dt
    bird = config.bird
    rho = config.sim.rho

    total_drag_work = 0.0
    n = len(trajectory)

    for i in range(n):
        vel = trajectory[i, 3:]
        pos = trajectory[i, :3]
        wind = wp.get_wind(pos)
        _, F_drag, airspeed, _, _ = compute_aero_forces(vel, wind, 0.05, 0.0, bird, rho)
        total_drag_work += np.linalg.norm(F_drag) * airspeed * dt / bird.mass

    # Net energy change
    wind_0 = wp.get_wind(trajectory[0, :3])
    wind_n = wp.get_wind(trajectory[-1, :3])
    v_air_0 = np.linalg.norm(trajectory[0, 3:] - wind_0)
    v_air_n = np.linalg.norm(trajectory[-1, 3:] - wind_n)
    E_0 = 0.5 * v_air_0 ** 2 + g * trajectory[0, 2]
    E_n = 0.5 * v_air_n ** 2 + g * trajectory[-1, 2]
    net_energy = E_n - E_0

    if total_drag_work == 0:
        return 0.0
    return float(net_energy / total_drag_work)


def decompose_energy_budget(
    trajectory: np.ndarray,
    config: Config,
) -> dict[str, float]:
    """Break down the total energy change into components.

    Returns dict with:
    - kinetic_change: change in airspeed kinetic energy
    - potential_change: change in gravitational PE
    - total_change: sum of above (= wind extraction - drag loss)
    - drag_loss: total energy lost to drag (positive number)
    - wind_extraction: total energy extracted from wind gradient (inferred)
    """
    wp = create_wind_profile(config.wind)
    g = config.sim.g
    dt = config.sim.dt

    wind_0 = wp.get_wind(trajectory[0, :3])
    wind_n = wp.get_wind(trajectory[-1, :3])
    v_air_0 = np.linalg.norm(trajectory[0, 3:] - wind_0)
    v_air_n = np.linalg.norm(trajectory[-1, 3:] - wind_n)

    ke_change = 0.5 * (v_air_n ** 2 - v_air_0 ** 2)
    pe_change = g * (trajectory[-1, 2] - trajectory[0, 2])
    total_change = ke_change + pe_change

    # Estimate drag loss
    drag_loss = 0.0
    for i in range(len(trajectory)):
        vel = trajectory[i, 3:]
        pos = trajectory[i, :3]
        wind = wp.get_wind(pos)
        _, F_drag, airspeed, _, _ = compute_aero_forces(
            vel, wind, 0.05, 0.0, config.bird, config.sim.rho
        )
        drag_loss += np.linalg.norm(F_drag) * airspeed * dt / config.bird.mass

    wind_extraction = total_change + drag_loss

    return {
        "kinetic_change": float(ke_change),
        "potential_change": float(pe_change),
        "total_change": float(total_change),
        "drag_loss": float(drag_loss),
        "wind_extraction": float(wind_extraction),
    }
