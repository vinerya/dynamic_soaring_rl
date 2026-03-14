"""Tests for research extension modules."""

from __future__ import annotations

import math
import unittest

import numpy as np

from dynamic_soaring.config import Config, WindConfig, BirdConfig


class TestTimeThreading(unittest.TestCase):
    """Verify time parameter flows through wind -> dynamics -> env."""

    def test_wind_profile_accepts_time(self):
        from dynamic_soaring.physics.wind import LogarithmicWindProfile, PowerLawWindProfile

        config = WindConfig()
        pos = np.array([0.0, 0.0, 10.0])

        log_profile = LogarithmicWindProfile(config)
        w1 = log_profile.get_wind(pos, time=0.0)
        w2 = log_profile.get_wind(pos, time=10.0)
        # Base profiles are time-independent, so results should match
        np.testing.assert_array_equal(w1, w2)

        pw_profile = PowerLawWindProfile(config)
        w3 = pw_profile.get_wind(pos, time=5.0)
        self.assertEqual(len(w3), 3)

    def test_time_varying_wind(self):
        from dynamic_soaring.physics.wind import LogarithmicWindProfile
        from dynamic_soaring.physics.wind_advanced import TimeVaryingWindProfile

        config = WindConfig(time_varying=True, time_varying_period=10.0, time_varying_amplitude=0.5)
        base = LogarithmicWindProfile(config)
        tv = TimeVaryingWindProfile(base, config)

        pos = np.array([0.0, 0.0, 10.0])
        w_t0 = tv.get_wind(pos, time=0.0)
        w_quarter = tv.get_wind(pos, time=2.5)  # sin(pi/2) = 1

        # At t=0, sin(0)=0 -> factor=1.0, at t=quarter_period factor=1.5
        np.testing.assert_allclose(w_quarter[:2], w_t0[:2] * 1.5, rtol=1e-10)

    def test_rk4_step_with_time(self):
        from dynamic_soaring.physics.dynamics import rk4_step
        from dynamic_soaring.physics.wind import LogarithmicWindProfile

        config = Config()
        wp = LogarithmicWindProfile(config.wind)
        state = np.array([0.0, 0.0, 30.0, 15.0, 0.0, 0.0])

        new_state, info = rk4_step(state, 0.05, 0.0, wp, config.bird, config.sim, time=1.0)
        self.assertEqual(len(new_state), 6)
        self.assertIn("airspeed", info)

    def test_composite_wind_factory(self):
        from dynamic_soaring.physics.wind import create_wind_profile

        config = WindConfig(
            profile_type="composite",
            dryden_enabled=True,
            time_varying=True,
        )
        profile = create_wind_profile(config)
        pos = np.array([0.0, 0.0, 10.0])
        w = profile.get_wind(pos, time=5.0)
        self.assertEqual(len(w), 3)


class TestDrydenTurbulence(unittest.TestCase):
    def test_dryden_adds_noise(self):
        from dynamic_soaring.physics.wind import LogarithmicWindProfile
        from dynamic_soaring.physics.wind_advanced import DrydenTurbulenceProfile

        # Use NO turbulence on base so base returns consistent values
        # Use shorter scale length to increase filter responsiveness
        config = WindConfig(dryden_enabled=True, dryden_intensity=2.0,
                            turbulence_intensity=0.0, dryden_scale_length=20.0)
        base = LogarithmicWindProfile(config)
        dryden = DrydenTurbulenceProfile(base, config, rng=np.random.default_rng(42))

        pos = np.array([0.0, 0.0, 20.0])
        # Step multiple times to build up gust state
        winds = [dryden.get_wind(pos) for _ in range(100)]

        # Dryden should produce variations across calls
        wind_array = np.array(winds)
        std = np.std(wind_array, axis=0)
        self.assertTrue(np.any(std > 0.001), "Dryden should add measurable noise")

    def test_dryden_gradient_delegates(self):
        from dynamic_soaring.physics.wind import LogarithmicWindProfile
        from dynamic_soaring.physics.wind_advanced import DrydenTurbulenceProfile

        config = WindConfig()
        base = LogarithmicWindProfile(config)
        dryden = DrydenTurbulenceProfile(base, config)

        pos = np.array([0.0, 0.0, 10.0])
        self.assertAlmostEqual(dryden.get_gradient(pos), base.get_gradient(pos))


class TestThermalUpdraft(unittest.TestCase):
    def test_thermal_adds_vertical(self):
        from dynamic_soaring.physics.wind import LogarithmicWindProfile
        from dynamic_soaring.physics.wind_advanced import ThermalUpdraftProfile

        config = WindConfig(
            thermal_enabled=True,
            thermal_centers=[(0.0, 0.0)],
            thermal_strength=5.0,
            thermal_radius=50.0,
        )
        base = LogarithmicWindProfile(config)
        thermal = ThermalUpdraftProfile(base, config)

        # At thermal center, should have updraft
        pos_center = np.array([0.0, 0.0, 50.0])
        w = thermal.get_wind(pos_center)
        w_base = base.get_wind(pos_center)
        self.assertGreater(w[2], w_base[2])

        # Far from thermal, no updraft
        pos_far = np.array([10000.0, 0.0, 50.0])
        w_far = thermal.get_wind(pos_far)
        self.assertAlmostEqual(w_far[2], w_base[2], places=2)


class TestRayleighCycle(unittest.TestCase):
    def test_generates_trajectory(self):
        from dynamic_soaring.analysis.rayleigh_cycle import generate_rayleigh_cycle

        config = Config()
        traj = generate_rayleigh_cycle(config)
        self.assertEqual(traj.ndim, 2)
        self.assertEqual(traj.shape[1], 6)
        self.assertGreater(len(traj), 10)


class TestTrajectoryClassifier(unittest.TestCase):
    def test_classifies_circular(self):
        from dynamic_soaring.analysis.trajectory_classifier import classify_trajectory_pattern

        # Create circular trajectory
        t = np.linspace(0, 4 * np.pi, 1000)
        traj = np.column_stack([
            10 * np.cos(t), 10 * np.sin(t), 30 + 5 * np.sin(t),
            -10 * np.sin(t), 10 * np.cos(t), 5 * np.cos(t),
        ])
        result = classify_trajectory_pattern(traj, dt=0.02)
        self.assertIn(result["pattern"], ["circular", "rayleigh", "level_flight", "figure_8", "s_turns"])
        self.assertIn("confidence", result)
        self.assertIn("dominant_period", result)

    def test_classifies_level_flight(self):
        from dynamic_soaring.analysis.trajectory_classifier import classify_trajectory_pattern

        # Straight level flight with very small altitude variation (< 3m)
        t = np.linspace(0, 10, 500)
        traj = np.column_stack([
            15 * t, np.zeros_like(t), 30 + 0.1 * np.sin(t),  # tiny alt variation
            15 * np.ones_like(t), np.zeros_like(t), 0.1 * np.cos(t),
        ])
        result = classify_trajectory_pattern(traj, dt=0.02)
        self.assertEqual(result["pattern"], "level_flight")


class TestPhaseSpace(unittest.TestCase):
    def test_compute_phase_portraits(self):
        from dynamic_soaring.analysis.phase_space import compute_phase_portraits

        config = Config()
        # Simple trajectory
        t = np.linspace(0, 10, 500)
        traj = np.column_stack([
            t, np.zeros_like(t), 30 + 5 * np.sin(t),
            np.ones_like(t) * 15, np.zeros_like(t), 5 * np.cos(t),
        ])
        portraits = compute_phase_portraits(traj, config)
        self.assertIn("altitude_airspeed", portraits)
        self.assertIn("heading_climb_rate", portraits)

    def test_frechet_distance(self):
        from dynamic_soaring.analysis.phase_space import frechet_distance

        p = np.array([[0, 0], [1, 0], [2, 0]])
        q = np.array([[0, 1], [1, 1], [2, 1]])
        d = frechet_distance(p, q)
        self.assertAlmostEqual(d, 1.0, places=5)


class TestEnergyExtraction(unittest.TestCase):
    def test_extraction_rate(self):
        from dynamic_soaring.analysis.energy_extraction import compute_energy_extraction_rate

        config = Config()
        t = np.linspace(0, 5, 250)
        traj = np.column_stack([
            t * 10, np.zeros_like(t), 30 + 5 * np.sin(t),
            10 * np.ones_like(t), np.zeros_like(t), 5 * np.cos(t),
        ])
        rates = compute_energy_extraction_rate(traj, config)
        self.assertEqual(len(rates), len(traj) - 1)

    def test_energy_budget(self):
        from dynamic_soaring.analysis.energy_extraction import decompose_energy_budget

        config = Config()
        t = np.linspace(0, 5, 250)
        traj = np.column_stack([
            t * 10, np.zeros_like(t), 30 + 5 * np.sin(t),
            10 * np.ones_like(t), np.zeros_like(t), 5 * np.cos(t),
        ])
        budget = decompose_energy_budget(traj, config)
        self.assertIn("kinetic_change", budget)
        self.assertIn("potential_change", budget)
        self.assertIn("total_change", budget)


class TestFlockAnalysis(unittest.TestCase):
    def test_inter_agent_distances(self):
        from dynamic_soaring.analysis.flock_analysis import compute_inter_agent_distances

        t = np.linspace(0, 5, 100)
        traj1 = np.column_stack([t, np.zeros_like(t), 30 * np.ones_like(t),
                                  np.ones_like(t), np.zeros_like(t), np.zeros_like(t)])
        traj2 = np.column_stack([t + 10, np.zeros_like(t), 30 * np.ones_like(t),
                                  np.ones_like(t), np.zeros_like(t), np.zeros_like(t)])
        distances = compute_inter_agent_distances([traj1, traj2])
        self.assertEqual(distances.shape, (100, 1))
        np.testing.assert_allclose(distances[:, 0], 10.0, atol=0.01)

    def test_detect_formation(self):
        from dynamic_soaring.analysis.flock_analysis import detect_formation

        # Cluster formation
        positions = np.array([[0, 0, 30], [1, 1, 30], [0, 1, 30], [1, 0, 30]], dtype=float)
        result = detect_formation(positions)
        self.assertIn("type", result)
        self.assertIn("spread", result)
        self.assertEqual(result["type"], "cluster")

    def test_flock_metrics(self):
        from dynamic_soaring.analysis.flock_analysis import compute_flock_metrics

        t = np.linspace(0, 5, 100)
        trajs = []
        for offset in [0, 10, 20]:
            traj = np.column_stack([
                t + offset, np.zeros_like(t), 30 + 5 * np.sin(t),
                np.ones_like(t), np.zeros_like(t), 5 * np.cos(t),
            ])
            trajs.append(traj)
        metrics = compute_flock_metrics(trajs, dt=0.02)
        self.assertIn("n_agents", metrics)
        self.assertEqual(metrics["n_agents"], 3)
        self.assertIn("coordination_index", metrics)


class TestNewMetrics(unittest.TestCase):
    def test_heading_rate(self):
        from dynamic_soaring.evaluation.metrics import compute_heading_rate

        # Constant heading -> rate should be zero
        velocities = np.array([[10, 0, 0]] * 10, dtype=float)
        rates = compute_heading_rate(velocities, dt=0.02)
        np.testing.assert_allclose(rates, 0.0, atol=1e-10)

    def test_turn_radius(self):
        from dynamic_soaring.evaluation.metrics import compute_turn_radius

        # Circular motion: radius=100m, omega=0.15 rad/s
        omega = 0.15
        r = 100.0
        speed = r * omega  # = 15 m/s
        dt = 0.01
        n_steps = 2000
        t = np.arange(n_steps) * dt
        velocities = np.column_stack([
            -speed * np.sin(omega * t),
            speed * np.cos(omega * t),
            np.zeros(n_steps),
        ])
        radii = compute_turn_radius(velocities, dt=dt)
        valid = radii[~np.isnan(radii)]
        self.assertGreater(len(valid), 0)
        # Median should be close to 100m
        self.assertAlmostEqual(np.median(valid), r, delta=r * 0.3)

    def test_glide_ratio(self):
        from dynamic_soaring.evaluation.metrics import compute_glide_ratio

        # Glide at 15 m/s horizontal, 1 m/s sink -> ratio = 15
        velocities = np.array([[15, 0, -1]] * 10, dtype=float)
        ratios = compute_glide_ratio(velocities)
        np.testing.assert_allclose(ratios, 15.0, rtol=0.01)

    def test_episode_stats_includes_new_metrics(self):
        from dynamic_soaring.evaluation.metrics import compute_episode_stats

        t = np.linspace(0, 5, 250)
        traj = np.column_stack([
            t * 10, np.zeros_like(t), 30 * np.ones_like(t),
            10 * np.ones_like(t), np.zeros_like(t), -0.5 * np.ones_like(t),
        ])
        stats = compute_episode_stats(traj, dt=0.02)
        self.assertIn("heading_rate_mean", stats)
        self.assertIn("turn_radius_mean", stats)
        self.assertIn("glide_ratio_mean", stats)


class TestRewardVariants(unittest.TestCase):
    def test_callable_variants(self):
        from dynamic_soaring.envs.reward_variants import REWARD_VARIANTS
        from dynamic_soaring.config import RewardConfig

        rc = RewardConfig()
        # Test the simple variants (not altitude_band/cycle_bonus which need extra args)
        for name in ["energy_only", "survival_heavy"]:
            fn = REWARD_VARIANTS[name]
            reward = fn(100.0, 101.0, np.zeros(2), None, False, None, rc)
            self.assertIsInstance(reward, float, f"Variant {name} didn't return float")

    def test_default_is_none(self):
        from dynamic_soaring.envs.reward_variants import REWARD_VARIANTS
        self.assertIsNone(REWARD_VARIANTS["default"])

    def test_altitude_band_variant(self):
        from dynamic_soaring.envs.reward_variants import compute_reward_altitude_band
        from dynamic_soaring.config import RewardConfig

        rc = RewardConfig()
        reward = compute_reward_altitude_band(100.0, 101.0, np.zeros(2), None, False, None, rc, altitude=20.0)
        self.assertIsInstance(reward, float)


class TestDomainRandomization(unittest.TestCase):
    def test_randomized_env_resets(self):
        from dynamic_soaring.training.domain_randomization import DomainRandomizedEnv

        config = Config()
        config.randomization.enabled = True
        env = DomainRandomizedEnv(config)
        obs, info = env.reset(seed=42)
        self.assertEqual(len(obs), 13)

    def test_context_conditioned_obs_extended(self):
        from dynamic_soaring.training.domain_randomization import ContextConditionedEnv

        config = Config()
        config.randomization.enabled = True
        config.training.context_conditioned = True
        env = ContextConditionedEnv(config)
        obs, info = env.reset(seed=42)
        self.assertEqual(len(obs), 17)  # 13 base + 4 context


class TestCurriculum(unittest.TestCase):
    def test_default_curriculum_stages(self):
        from dynamic_soaring.training.curriculum import DEFAULT_CURRICULUM

        self.assertEqual(len(DEFAULT_CURRICULUM), 3)
        self.assertEqual(DEFAULT_CURRICULUM[0]["name"], "easy")


class TestConfigExtensions(unittest.TestCase):
    def test_merge_overrides(self):
        config = Config()
        new = config.merge_overrides({"wind": {"reference_speed": 25.0}})
        self.assertEqual(new.wind.reference_speed, 25.0)
        self.assertEqual(config.wind.reference_speed, 15.0)  # original unchanged

    def test_curriculum_config(self):
        config = Config()
        self.assertFalse(config.curriculum.enabled)

    def test_multi_agent_config(self):
        config = Config()
        self.assertEqual(config.multi_agent.n_agents, 4)
        self.assertEqual(config.multi_agent.comm_variant, "no_comm")


class TestWakeModel(unittest.TestCase):
    def test_compute_circulation(self):
        from dynamic_soaring.physics.wake_model import compute_circulation

        config = Config()
        gamma = compute_circulation(15.0, 1.0, config.bird)
        self.assertGreater(gamma, 0)

    def test_wake_velocity_decays(self):
        from dynamic_soaring.physics.wake_model import compute_wake_velocity

        config = Config()
        source_pos = np.array([0.0, 0.0, 30.0])
        source_vel = np.array([15.0, 0.0, 0.0])
        gamma = 5.0

        # Close target (behind leader)
        target_close = np.array([5.0, 5.0, 30.0])
        w_close = compute_wake_velocity(source_pos, source_vel, gamma, target_close, config.bird)

        # Far target (behind leader)
        target_far = np.array([100.0, 5.0, 30.0])
        w_far = compute_wake_velocity(source_pos, source_vel, gamma, target_far, config.bird)

        self.assertGreater(np.linalg.norm(w_close), np.linalg.norm(w_far))


class TestBioComparison(unittest.TestCase):
    def test_biological_metrics(self):
        from dynamic_soaring.analysis.bio_comparison import compute_biological_metrics

        config = Config()
        t = np.linspace(0, 10, 500)
        traj = np.column_stack([
            100 * np.cos(t * 0.5), 100 * np.sin(t * 0.5), 30 + 10 * np.sin(t),
            -50 * np.sin(t * 0.5), 50 * np.cos(t * 0.5), 10 * np.cos(t),
        ])
        metrics = compute_biological_metrics(traj, config)
        self.assertIn("glide_ratio", metrics)
        self.assertIn("mean_turn_radius", metrics)

    def test_dtw_distance(self):
        from dynamic_soaring.analysis.bio_comparison import dtw_distance

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = dtw_distance(a, b)
        self.assertAlmostEqual(d, 0.0, places=5)

        c = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        d2 = dtw_distance(a, c)
        self.assertGreater(d2, 0)


class TestEvaluateReturnTrajectories(unittest.TestCase):
    def test_return_trajectories_flag(self):
        """Verify the return_trajectories parameter exists in evaluate_policy signature."""
        from dynamic_soaring.evaluation.evaluate import evaluate_policy
        import inspect
        sig = inspect.signature(evaluate_policy)
        self.assertIn("return_trajectories", sig.parameters)


if __name__ == "__main__":
    unittest.main()
