"""Biological comparison: RL agent vs real albatross trajectories."""

from __future__ import annotations

import math

import numpy as np

from dynamic_soaring.config import Config
from dynamic_soaring.physics.wind import create_wind_profile


def compute_biological_metrics(
    trajectory: np.ndarray,
    config: Config,
) -> dict[str, float]:
    """Compute biologically-relevant flight metrics.

    Returns: wing_loading, glide_ratio, mean_turn_radius, mean_bank_estimate,
             mean_airspeed, altitude_range, cycle_period.
    """
    positions = trajectory[:, :3]
    velocities = trajectory[:, 3:]
    altitudes = positions[:, 2]
    dt = config.sim.dt

    wp = create_wind_profile(config.wind)
    winds = np.array([wp.get_wind(p) for p in positions])
    v_air = velocities - winds
    airspeeds = np.linalg.norm(v_air, axis=1)

    # Wing loading (N/m²)
    wing_loading = config.bird.mass * config.sim.g / config.bird.wing_area

    # Glide ratio (L/D) estimated from horizontal distance / altitude loss
    horiz_dist = np.sum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1))
    alt_change = altitudes[-1] - altitudes[0]
    glide_ratio = abs(horiz_dist / max(abs(alt_change), 0.1)) if alt_change < 0 else float("inf")

    # Turn radius estimation
    headings = np.arctan2(velocities[:, 1], velocities[:, 0])
    heading_rates = np.diff(np.unwrap(headings)) / dt
    ground_speeds = np.linalg.norm(velocities[:, :2], axis=1)
    # r = v / omega
    valid_mask = np.abs(heading_rates) > 0.01
    if np.any(valid_mask):
        turn_radii = ground_speeds[:-1][valid_mask] / np.abs(heading_rates[valid_mask])
        mean_turn_radius = float(np.median(turn_radii))
    else:
        mean_turn_radius = float("inf")

    # Bank angle estimation from turn: tan(bank) = v²/(r*g)
    if mean_turn_radius < 1e6:
        v_mean = float(np.mean(ground_speeds))
        estimated_bank = math.atan2(v_mean ** 2, mean_turn_radius * config.sim.g)
    else:
        estimated_bank = 0.0

    # Altitude oscillation period
    dz = np.diff(altitudes)
    sign_changes = np.where(np.diff(np.sign(dz)) != 0)[0]
    if len(sign_changes) >= 2:
        avg_half_period = np.mean(np.diff(sign_changes)) * dt
        cycle_period = 2 * avg_half_period
    else:
        cycle_period = 0.0

    return {
        "wing_loading": float(wing_loading),
        "glide_ratio": float(min(glide_ratio, 100.0)),
        "mean_turn_radius": float(mean_turn_radius),
        "estimated_bank_angle": float(estimated_bank),
        "mean_airspeed": float(np.mean(airspeeds)),
        "altitude_range": float(np.max(altitudes) - np.min(altitudes)),
        "cycle_period": float(cycle_period),
        "mean_climb_rate": float(np.mean(np.abs(velocities[:, 2]))),
    }


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Dynamic Time Warping distance between two sequences.

    Args:
        seq_a, seq_b: (N, D) and (M, D) arrays.

    Returns the DTW distance (float).
    """
    n, m = len(seq_a), len(seq_b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq_a[i - 1] - seq_b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(dtw[n, m])


def compare_trajectories(
    traj_agent: np.ndarray,
    traj_bird: np.ndarray,
    config: Config,
) -> dict:
    """Quantitative comparison of RL agent and real bird trajectories.

    Returns similarity metrics and biological metric comparison.
    """
    metrics_agent = compute_biological_metrics(traj_agent, config)
    metrics_bird = compute_biological_metrics(traj_bird, config)

    # Subsample for DTW efficiency
    step_a = max(1, len(traj_agent) // 200)
    step_b = max(1, len(traj_bird) // 200)

    # Normalize positions for fair comparison
    combined = np.vstack([traj_agent[::step_a, :3], traj_bird[::step_b, :3]])
    pos_scale = np.std(combined, axis=0)
    pos_scale[pos_scale == 0] = 1.0

    dtw_dist = dtw_distance(
        traj_agent[::step_a, :3] / pos_scale,
        traj_bird[::step_b, :3] / pos_scale,
    )

    # Altitude pattern similarity (DTW on altitude only)
    alt_a = traj_agent[::step_a, 2:3] / max(np.std(traj_agent[:, 2]), 1.0)
    alt_b = traj_bird[::step_b, 2:3] / max(np.std(traj_bird[:, 2]), 1.0)
    dtw_alt = dtw_distance(alt_a, alt_b)

    return {
        "metrics_agent": metrics_agent,
        "metrics_bird": metrics_bird,
        "dtw_3d": float(dtw_dist),
        "dtw_altitude": float(dtw_alt),
        "metric_differences": {
            k: metrics_agent[k] - metrics_bird[k]
            for k in metrics_agent
            if isinstance(metrics_agent[k], (int, float)) and k in metrics_bird
        },
    }
