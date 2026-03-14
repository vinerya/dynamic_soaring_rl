"""Analyze flock behavior in multi-agent soaring."""

from __future__ import annotations

import numpy as np


def compute_inter_agent_distances(trajectories: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise distances over time.

    Args:
        trajectories: list of (T, 6) arrays per agent

    Returns (T, N*(N-1)/2) array of pairwise distances.
    """
    n = len(trajectories)
    min_len = min(len(t) for t in trajectories)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            dists = np.linalg.norm(
                trajectories[i][:min_len, :3] - trajectories[j][:min_len, :3],
                axis=1,
            )
            pairs.append(dists)

    return np.column_stack(pairs) if pairs else np.zeros((min_len, 0))


def detect_formation(positions: np.ndarray) -> dict:
    """Classify flock formation at a single timestep.

    Args:
        positions: (N, 3) array of agent positions

    Returns dict with formation type and metrics.
    """
    n = len(positions)
    if n < 2:
        return {"type": "single", "spread": 0.0}

    # Centroid
    centroid = np.mean(positions, axis=0)
    spread = float(np.mean(np.linalg.norm(positions - centroid, axis=1)))

    # PCA to find dominant axis
    centered = positions[:, :2] - centroid[:2]  # 2D projection
    if n >= 2:
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Ratio of eigenvalues indicates formation shape
        if eigvals[0] > 0:
            elongation = eigvals[1] / max(eigvals[0], 1e-6)
        else:
            elongation = 1.0
    else:
        elongation = 1.0

    # Check for V-formation: project onto dominant axis and check symmetry
    if elongation > 3.0:
        formation_type = "line"
    elif elongation > 1.5:
        # Could be V-formation or echelon
        # Check altitude variation
        alt_range = np.max(positions[:, 2]) - np.min(positions[:, 2])
        if alt_range < 5.0:
            formation_type = "v_formation"
        else:
            formation_type = "echelon"
    else:
        formation_type = "cluster"

    return {
        "type": formation_type,
        "spread": spread,
        "elongation": float(elongation),
        "centroid": centroid.tolist(),
    }


def compute_flock_metrics(trajectories: list[np.ndarray], dt: float) -> dict:
    """Compute comprehensive flock behavior metrics.

    Returns: mean_distance, distance_std, formation_stability,
             collective_energy_efficiency, coordination_index.
    """
    n = len(trajectories)
    min_len = min(len(t) for t in trajectories)

    distances = compute_inter_agent_distances(trajectories)
    mean_dist = float(np.mean(distances))
    dist_std = float(np.std(distances))

    # Formation stability: how consistent is the formation over time?
    formation_types = []
    for t in range(0, min_len, max(1, min_len // 50)):
        positions = np.array([traj[t, :3] for traj in trajectories])
        f = detect_formation(positions)
        formation_types.append(f["type"])

    if formation_types:
        from collections import Counter
        type_counts = Counter(formation_types)
        dominant_type = type_counts.most_common(1)[0]
        formation_stability = dominant_type[1] / len(formation_types)
        dominant_formation = dominant_type[0]
    else:
        formation_stability = 0.0
        dominant_formation = "unknown"

    # Coordination: how correlated are altitude oscillations?
    alt_signals = [traj[:min_len, 2] for traj in trajectories]
    correlations = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.std(alt_signals[i]) > 0 and np.std(alt_signals[j]) > 0:
                c = np.corrcoef(alt_signals[i], alt_signals[j])[0, 1]
                correlations.append(c)
    coordination_index = float(np.mean(correlations)) if correlations else 0.0

    return {
        "n_agents": n,
        "mean_distance": mean_dist,
        "distance_std": dist_std,
        "formation_stability": float(formation_stability),
        "dominant_formation": dominant_formation,
        "coordination_index": coordination_index,
        "flight_duration": float(min_len * dt),
    }


def compute_cooperation_benefit(
    multi_agent_energies: list[float],
    single_agent_energy: float,
) -> dict:
    """Quantify energy advantage of cooperative flight.

    Returns: mean_benefit, benefit_std, benefit_pct.
    """
    benefits = [e - single_agent_energy for e in multi_agent_energies]
    mean_benefit = float(np.mean(benefits))
    return {
        "mean_benefit": mean_benefit,
        "benefit_std": float(np.std(benefits)),
        "benefit_pct": float(mean_benefit / max(abs(single_agent_energy), 1e-6) * 100),
    }
