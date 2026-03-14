"""Unit tests for the physics engine."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dynamic_soaring.config import BirdConfig, Config, SimConfig, WindConfig
from dynamic_soaring.physics.aerodynamics import (
    compute_aero_forces,
    drag_coefficient,
    lift_coefficient,
)
from dynamic_soaring.physics.dynamics import compute_total_energy, rk4_step
from dynamic_soaring.physics.wind import (
    LogarithmicWindProfile,
    PowerLawWindProfile,
    create_wind_profile,
)


class TestWindProfile:
    def test_log_profile_zero_at_ground(self):
        config = WindConfig(reference_speed=15.0, reference_height=10.0, surface_roughness=0.001)
        wp = LogarithmicWindProfile(config)
        wind = wp.get_wind(np.array([0.0, 0.0, 0.0]))
        assert np.allclose(wind, 0.0)

    def test_log_profile_at_reference_height(self):
        config = WindConfig(reference_speed=15.0, reference_height=10.0, surface_roughness=0.001)
        wp = LogarithmicWindProfile(config)
        wind = wp.get_wind(np.array([0.0, 0.0, 10.0]))
        speed = np.linalg.norm(wind)
        assert abs(speed - 15.0) < 0.01

    def test_log_profile_increases_with_altitude(self):
        config = WindConfig(reference_speed=15.0, reference_height=10.0)
        wp = LogarithmicWindProfile(config)
        s1 = np.linalg.norm(wp.get_wind(np.array([0.0, 0.0, 5.0])))
        s2 = np.linalg.norm(wp.get_wind(np.array([0.0, 0.0, 20.0])))
        s3 = np.linalg.norm(wp.get_wind(np.array([0.0, 0.0, 50.0])))
        assert s1 < s2 < s3

    def test_log_profile_gradient_positive(self):
        config = WindConfig(reference_speed=15.0, reference_height=10.0)
        wp = LogarithmicWindProfile(config)
        grad = wp.get_gradient(np.array([0.0, 0.0, 10.0]))
        assert grad > 0

    def test_log_profile_gradient_decreases_with_altitude(self):
        config = WindConfig(reference_speed=15.0, reference_height=10.0)
        wp = LogarithmicWindProfile(config)
        g1 = wp.get_gradient(np.array([0.0, 0.0, 5.0]))
        g2 = wp.get_gradient(np.array([0.0, 0.0, 50.0]))
        assert g1 > g2

    def test_power_law_at_reference(self):
        config = WindConfig(profile_type="power_law", reference_speed=15.0, reference_height=10.0)
        wp = PowerLawWindProfile(config)
        speed = np.linalg.norm(wp.get_wind(np.array([0.0, 0.0, 10.0])))
        assert abs(speed - 15.0) < 0.01

    def test_factory(self):
        config = WindConfig(profile_type="logarithmic")
        wp = create_wind_profile(config)
        assert isinstance(wp, LogarithmicWindProfile)

    def test_wind_direction(self):
        config = WindConfig(direction=math.pi / 2, reference_speed=15.0, reference_height=10.0)
        wp = LogarithmicWindProfile(config)
        wind = wp.get_wind(np.array([0.0, 0.0, 10.0]))
        assert abs(wind[0]) < 0.01  # cos(pi/2) ~ 0
        assert abs(wind[1] - 15.0) < 0.01  # sin(pi/2) ~ 1


class TestAerodynamics:
    def test_lift_linear_regime(self):
        bird = BirdConfig()
        alpha = 0.1  # ~5.7 deg
        cl = lift_coefficient(alpha, bird)
        assert abs(cl - bird.cl_alpha * alpha) < 1e-6

    def test_lift_symmetric(self):
        bird = BirdConfig()
        cl_pos = lift_coefficient(0.1, bird)
        cl_neg = lift_coefficient(-0.1, bird)
        assert abs(cl_pos + cl_neg) < 1e-6

    def test_lift_post_stall_drops(self):
        bird = BirdConfig()
        cl_stall = lift_coefficient(bird.alpha_stall, bird)
        # Well past stall (45 deg), lift should be significantly reduced
        cl_deep_stall = lift_coefficient(bird.alpha_stall + 0.5, bird)
        assert abs(cl_deep_stall) < abs(cl_stall)

    def test_drag_minimum_at_zero_lift(self):
        bird = BirdConfig()
        cd_zero = drag_coefficient(0.0, bird, 0.0)
        cd_lift = drag_coefficient(1.0, bird, 0.1)
        assert cd_zero < cd_lift

    def test_drag_increases_post_stall(self):
        bird = BirdConfig()
        cl_pre = lift_coefficient(bird.alpha_stall * 0.9, bird)
        cd_pre = drag_coefficient(cl_pre, bird, bird.alpha_stall * 0.9)
        cl_post = lift_coefficient(bird.alpha_stall * 1.5, bird)
        cd_post = drag_coefficient(cl_post, bird, bird.alpha_stall * 1.5)
        assert cd_post > cd_pre

    def test_forces_zero_airspeed(self):
        bird = BirdConfig()
        vel = np.array([10.0, 0.0, 0.0])
        wind = np.array([10.0, 0.0, 0.0])  # same as bird velocity
        F_lift, F_drag, airspeed, _, _ = compute_aero_forces(vel, wind, 0.1, 0.0, bird, 1.225)
        assert airspeed < 1e-5
        assert np.allclose(F_lift, 0.0)
        assert np.allclose(F_drag, 0.0)

    def test_lift_perpendicular_to_velocity(self):
        bird = BirdConfig()
        vel = np.array([15.0, 0.0, 0.0])
        wind = np.zeros(3)
        F_lift, F_drag, _, _, _ = compute_aero_forces(vel, wind, 0.1, 0.0, bird, 1.225)
        # Lift should be perpendicular to velocity (dot product ~ 0)
        dot = np.dot(F_lift, vel)
        assert abs(dot) < 1e-6

    def test_drag_opposes_velocity(self):
        bird = BirdConfig()
        vel = np.array([15.0, 0.0, 0.0])
        wind = np.zeros(3)
        F_lift, F_drag, _, _, _ = compute_aero_forces(vel, wind, 0.1, 0.0, bird, 1.225)
        # Drag should oppose velocity (negative x-component)
        assert F_drag[0] < 0


class TestDynamics:
    def test_rk4_free_fall(self):
        """Object in free fall should follow parabolic trajectory."""
        config = Config()
        wind_config = WindConfig(reference_speed=0.0)  # no wind
        wp = create_wind_profile(wind_config)

        # Start at 100m with no velocity
        state = np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0])

        # Zero alpha/bank -> zero lift, just gravity + drag
        # With very high altitude and no wind, after 1 second vz should be ~-g
        for _ in range(50):  # 50 * 0.02 = 1 second
            state, _ = rk4_step(state, 0.0, 0.0, wp, config.bird, config.sim)

        # Should have fallen (z decreased)
        assert state[2] < 100.0
        # Vertical velocity should be negative
        assert state[5] < 0.0

    def test_rk4_ground_clamp(self):
        """State should be clamped at ground level."""
        config = Config()
        wind_config = WindConfig(reference_speed=0.0)
        wp = create_wind_profile(wind_config)

        state = np.array([0.0, 0.0, 1.0, 0.0, 0.0, -50.0])  # fast downward
        state, _ = rk4_step(state, 0.0, 0.0, wp, config.bird, config.sim)
        assert state[2] >= 0.0
        assert state[5] >= 0.0

    def test_total_energy(self):
        state = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0])
        wind = np.zeros(3)
        e = compute_total_energy(state, wind, 8.5, 9.81)
        expected = 0.5 * 100 + 9.81 * 100  # 0.5*v^2 + g*z
        assert abs(e - expected) < 1e-6
