"""Phase space analysis for soaring trajectories."""

from __future__ import annotations

import math

import numpy as np

from dynamic_soaring.config import Config
from dynamic_soaring.physics.wind import create_wind_profile


def compute_phase_portraits(trajectory: np.ndarray, config: Config) -> dict[str, np.ndarray]:
    """Compute phase portrait data for a trajectory.

    Returns dict of 2D arrays for various phase space projections.
    """
    positions = trajectory[:, :3]
    velocities = trajectory[:, 3:]
    altitudes = positions[:, 2]

    wind_profile = create_wind_profile(config.wind)
    winds = np.array([wind_profile.get_wind(p) for p in positions])
    v_air = velocities - winds
    airspeeds = np.linalg.norm(v_air, axis=1)
    ground_speeds = np.linalg.norm(velocities[:, :2], axis=1)

    headings = np.arctan2(velocities[:, 1], velocities[:, 0])
    climb_rates = velocities[:, 2]

    # Specific energy
    g = config.sim.g
    specific_energy = 0.5 * airspeeds ** 2 + g * altitudes

    return {
        "altitude_airspeed": np.column_stack([altitudes, airspeeds]),
        "heading_climb_rate": np.column_stack([headings, climb_rates]),
        "energy_altitude": np.column_stack([specific_energy, altitudes]),
        "airspeed_climb_rate": np.column_stack([airspeeds, climb_rates]),
        "ground_speed_altitude": np.column_stack([ground_speeds, altitudes]),
    }


def frechet_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute discrete Frechet distance between two 2D curves.

    P, Q: (N, 2) and (M, 2) arrays.
    """
    n, m = len(P), len(Q)
    ca = np.full((n, m), -1.0)

    def _c(i, j):
        if ca[i, j] > -0.5:
            return ca[i, j]
        d = np.linalg.norm(P[i] - Q[j])
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i == 0:
            ca[i, j] = max(_c(0, j - 1), d)
        elif j == 0:
            ca[i, j] = max(_c(i - 1, 0), d)
        else:
            ca[i, j] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), d)
        return ca[i, j]

    # Use iterative approach to avoid recursion limit
    for i in range(n):
        for j in range(m):
            d = np.linalg.norm(P[i] - Q[j])
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(ca[0, j - 1], d)
            elif j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
            else:
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d)

    return float(ca[n - 1, m - 1])


def compare_phase_portraits(
    portraits_a: dict[str, np.ndarray],
    portraits_b: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compare two sets of phase portraits using Frechet distance."""
    distances = {}
    for key in portraits_a:
        if key in portraits_b:
            # Normalize both to [0,1] range for fair comparison
            pa = portraits_a[key].copy()
            pb = portraits_b[key].copy()
            combined = np.vstack([pa, pb])
            mins = combined.min(axis=0)
            ranges = combined.max(axis=0) - mins
            ranges[ranges == 0] = 1.0
            pa = (pa - mins) / ranges
            pb = (pb - mins) / ranges
            # Subsample for performance
            step_a = max(1, len(pa) // 500)
            step_b = max(1, len(pb) // 500)
            distances[key] = frechet_distance(pa[::step_a], pb[::step_b])
    return distances
