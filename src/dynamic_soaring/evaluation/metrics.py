"""Soaring performance metrics and cycle detection."""

from __future__ import annotations

import numpy as np


def detect_soaring_cycles(altitudes: np.ndarray, min_amplitude: float = 5.0) -> list[dict]:
    """Detect dynamic soaring cycles from altitude time series.

    A cycle is defined as a full oscillation (peak -> valley -> peak)
    with minimum amplitude.

    Returns list of cycle dicts with: start_idx, end_idx, period, amplitude.
    """
    if len(altitudes) < 10:
        return []

    # Find peaks and valleys using simple sign change of derivative
    dz = np.diff(altitudes)
    sign_changes = np.where(np.diff(np.sign(dz)))[0] + 1

    if len(sign_changes) < 2:
        return []

    cycles = []
    for i in range(0, len(sign_changes) - 1, 2):
        start = sign_changes[i]
        end = sign_changes[i + 1] if i + 1 < len(sign_changes) else len(altitudes) - 1
        amplitude = abs(altitudes[start] - altitudes[(start + end) // 2])
        if amplitude >= min_amplitude:
            cycles.append({
                "start_idx": int(start),
                "end_idx": int(end),
                "period": int(end - start),
                "amplitude": float(amplitude),
            })

    return cycles


def compute_episode_stats(trajectory: np.ndarray, dt: float) -> dict:
    """Compute comprehensive statistics for an episode trajectory.

    Args:
        trajectory: (N, 6) array of [x, y, z, vx, vy, vz]
        dt: time step in seconds

    Returns dict with flight statistics.
    """
    positions = trajectory[:, :3]
    velocities = trajectory[:, 3:]
    altitudes = positions[:, 2]
    speeds = np.linalg.norm(velocities, axis=1)

    duration = len(trajectory) * dt

    # Energy (kinetic + potential, per unit mass)
    g = 9.81
    ke = 0.5 * speeds ** 2
    pe = g * altitudes
    total_energy = ke + pe

    cycles = detect_soaring_cycles(altitudes)

    return {
        "duration_s": duration,
        "n_steps": len(trajectory),
        "altitude_mean": float(np.mean(altitudes)),
        "altitude_max": float(np.max(altitudes)),
        "altitude_min": float(np.min(altitudes)),
        "altitude_std": float(np.std(altitudes)),
        "speed_mean": float(np.mean(speeds)),
        "speed_max": float(np.max(speeds)),
        "speed_min": float(np.min(speeds)),
        "energy_initial": float(total_energy[0]),
        "energy_final": float(total_energy[-1]),
        "energy_change": float(total_energy[-1] - total_energy[0]),
        "energy_change_pct": float(
            (total_energy[-1] - total_energy[0]) / max(total_energy[0], 1e-6) * 100
        ),
        "n_soaring_cycles": len(cycles),
        "cycles": cycles,
        "distance_traveled": float(
            np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        ),
    }
