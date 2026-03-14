"""Classify dynamic soaring trajectory patterns."""

from __future__ import annotations

import math

import numpy as np


def classify_trajectory_pattern(trajectory: np.ndarray, dt: float) -> dict:
    """Classify the dominant flight pattern in a trajectory.

    Analyzes heading and altitude time series to identify:
    - circular: monotonic heading change (constant-turn soaring)
    - figure_8: heading oscillates with ~2 reversals per altitude cycle
    - s_turns: heading oscillates with ~1 reversal per altitude cycle
    - rayleigh: 4-phase pattern with upwind climb and downwind descent
    - erratic: no clear pattern

    Returns dict with: pattern, confidence, dominant_period, heading_rate_mean.
    """
    if len(trajectory) < 50:
        return {"pattern": "too_short", "confidence": 0.0}

    positions = trajectory[:, :3]
    velocities = trajectory[:, 3:]
    altitudes = positions[:, 2]

    # Heading time series
    headings = np.arctan2(velocities[:, 1], velocities[:, 0])
    heading_rate = np.diff(np.unwrap(headings)) / dt

    # Altitude oscillation analysis via FFT
    alt_detrended = altitudes - np.mean(altitudes)
    n = len(alt_detrended)
    if n < 20:
        return {"pattern": "too_short", "confidence": 0.0}

    fft_alt = np.fft.rfft(alt_detrended)
    freqs = np.fft.rfftfreq(n, d=dt)
    power = np.abs(fft_alt[1:]) ** 2  # skip DC
    freqs = freqs[1:]

    if len(power) == 0 or np.max(power) == 0:
        return {"pattern": "erratic", "confidence": 0.0, "dominant_period": 0.0}

    dominant_idx = np.argmax(power)
    dominant_freq = freqs[dominant_idx]
    dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else 0.0

    # Heading analysis
    mean_heading_rate = float(np.mean(np.abs(heading_rate)))
    heading_reversals = np.sum(np.diff(np.sign(heading_rate)) != 0)

    # Altitude cycles
    alt_range = np.max(altitudes) - np.min(altitudes)
    alt_std = np.std(altitudes)

    # Classification logic
    total_heading_change = np.abs(np.sum(np.diff(np.unwrap(headings))))
    n_full_turns = total_heading_change / (2 * math.pi)
    duration = len(trajectory) * dt

    # Altitude cycles count
    dz = np.diff(altitudes)
    alt_zero_crossings = np.sum(np.diff(np.sign(dz)) != 0) // 2

    if alt_range < 3.0:
        pattern = "level_flight"
        confidence = 0.8
    elif n_full_turns > 0.8 * duration / max(dominant_period, 1.0):
        # Heading continuously rotates -> circular
        pattern = "circular"
        confidence = min(1.0, n_full_turns / max(alt_zero_crossings, 1))
    elif alt_zero_crossings > 0 and heading_reversals > 1.5 * alt_zero_crossings:
        # More heading reversals than altitude cycles -> figure-8
        pattern = "figure_8"
        confidence = min(1.0, heading_reversals / (2 * max(alt_zero_crossings, 1)))
    elif alt_zero_crossings > 0 and heading_reversals > 0:
        pattern = "s_turns"
        confidence = 0.6
    else:
        pattern = "erratic"
        confidence = 0.3

    return {
        "pattern": pattern,
        "confidence": float(confidence),
        "dominant_period": float(dominant_period),
        "heading_rate_mean": float(mean_heading_rate),
        "altitude_range": float(alt_range),
        "altitude_cycles": int(alt_zero_crossings),
        "heading_reversals": int(heading_reversals),
        "n_full_turns": float(n_full_turns),
    }
