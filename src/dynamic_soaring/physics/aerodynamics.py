"""Aerodynamic force computation with proper stall modeling."""

from __future__ import annotations

import math

import numpy as np

from dynamic_soaring.config import BirdConfig


def lift_coefficient(alpha: float, bird: BirdConfig) -> float:
    """Compute lift coefficient with post-stall dropoff.

    Linear regime: Cl = Cl_alpha * alpha
    Post-stall: exponential decay from Cl_max toward flat-plate Cl.
    """
    if abs(alpha) <= bird.alpha_stall:
        return bird.cl_alpha * alpha

    sign = 1.0 if alpha > 0 else -1.0
    excess = abs(alpha) - bird.alpha_stall
    # Exponential decay: Cl drops from Cl_max with increasing excess angle
    decay_rate = 4.0  # controls how quickly lift drops post-stall
    cl = bird.cl_max * math.exp(-decay_rate * excess)
    return sign * cl


def drag_coefficient(cl: float, bird: BirdConfig, alpha: float) -> float:
    """Compute drag coefficient: parabolic polar + post-stall increase."""
    k = 1.0 / (math.pi * bird.oswald_efficiency * bird.aspect_ratio)
    cd = bird.cd0 + k * cl * cl

    # Extra drag in post-stall regime
    if abs(alpha) > bird.alpha_stall:
        excess = abs(alpha) - bird.alpha_stall
        cd += 2.0 * excess * excess  # quadratic post-stall drag rise

    return cd


def compute_aero_forces(
    velocity: np.ndarray,
    wind: np.ndarray,
    alpha: float,
    bank_angle: float,
    bird: BirdConfig,
    rho: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Compute aerodynamic lift and drag forces.

    Uses Rodrigues' rotation for proper lift direction with bank angle.

    Returns:
        (F_lift, F_drag, airspeed, cl, cd)
    """
    v_air = velocity - wind
    airspeed = np.linalg.norm(v_air)

    if airspeed < 1e-6:
        zero = np.zeros(3)
        return zero, zero, 0.0, 0.0, 0.0

    # Dynamic pressure
    q = 0.5 * rho * airspeed * airspeed
    S = bird.wing_area

    # Coefficients
    cl = lift_coefficient(alpha, bird)
    cd = drag_coefficient(cl, bird, alpha)

    # Drag direction: opposite to airspeed
    e_drag = v_air / airspeed
    F_drag = -cd * q * S * e_drag

    # Lift direction: perpendicular to velocity, rotated by bank angle
    # Step 1: world-up component perpendicular to flight direction
    e_up = np.array([0.0, 0.0, 1.0])
    e_perp = e_up - np.dot(e_up, e_drag) * e_drag
    perp_norm = np.linalg.norm(e_perp)

    if perp_norm < 1e-6:
        # Flying straight up/down - use arbitrary perpendicular
        e_perp = np.array([1.0, 0.0, 0.0])
        e_perp = e_perp - np.dot(e_perp, e_drag) * e_drag
        perp_norm = np.linalg.norm(e_perp)

    e_perp = e_perp / perp_norm

    # Step 2: Rodrigues' rotation around velocity axis for bank
    cos_b = math.cos(bank_angle)
    sin_b = math.sin(bank_angle)
    e_lift = e_perp * cos_b + np.cross(e_drag, e_perp) * sin_b

    F_lift = cl * q * S * e_lift

    return F_lift, F_drag, airspeed, cl, cd
