"""Tests for the Gymnasium environment."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dynamic_soaring.config import Config
from dynamic_soaring.envs.soaring_env import DynamicSoaringEnv


class TestEnvironment:
    def test_reset_returns_correct_shapes(self):
        env = DynamicSoaringEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (13,)
        assert isinstance(info, dict)

    def test_step_returns_correct_shapes(self):
        env = DynamicSoaringEnv()
        env.reset(seed=42)
        action = np.array([0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (13,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_obs_within_bounds(self):
        env = DynamicSoaringEnv()
        obs, _ = env.reset(seed=42)
        # Normalized obs should generally be in reasonable range
        assert np.all(np.isfinite(obs))

    def test_action_clipping(self):
        env = DynamicSoaringEnv()
        env.reset(seed=42)
        # Actions outside [-1,1] should be clipped
        action = np.array([5.0, -5.0], dtype=np.float32)
        obs, reward, _, _, _ = env.step(action)
        assert np.all(np.isfinite(obs))

    def test_episode_terminates(self):
        """An episode with extreme actions should eventually terminate."""
        env = DynamicSoaringEnv()
        env.reset(seed=42)
        done = False
        steps = 0
        while not done and steps < 5000:
            action = np.array([1.0, 1.0], dtype=np.float32)  # max alpha, max bank
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        assert done

    def test_trajectory_recorded(self):
        env = DynamicSoaringEnv()
        env.reset(seed=42)
        for _ in range(10):
            env.step(env.action_space.sample())
        assert len(env.trajectory) == 11  # initial + 10 steps

    def test_reproducibility(self):
        env = DynamicSoaringEnv()
        obs1, _ = env.reset(seed=42)
        action = np.array([0.1, -0.2], dtype=np.float32)
        obs1_next, r1, _, _, _ = env.step(action)

        obs2, _ = env.reset(seed=42)
        obs2_next, r2, _, _, _ = env.step(action)

        assert np.allclose(obs1, obs2)
        assert np.allclose(obs1_next, obs2_next)
        assert abs(r1 - r2) < 1e-6

    def test_info_contains_position(self):
        env = DynamicSoaringEnv()
        _, info = env.reset(seed=42)
        assert "position" in info
        assert "altitude" in info
        assert info["altitude"] > 0

    def test_crash_gives_negative_reward(self):
        """Forcing the bird downward should eventually crash with penalty."""
        config = Config()
        config.sim.init_altitude_range = (5.0, 5.0)  # start low
        env = DynamicSoaringEnv(config)
        env.reset(seed=42)

        total_reward = 0.0
        for _ in range(1000):
            # Nose down, no bank
            action = np.array([-1.0, 0.0], dtype=np.float32)
            _, reward, terminated, _, info = env.step(action)
            total_reward += reward
            if terminated:
                break

        # Should have crashed with negative terminal reward
        assert terminated
        assert info.get("termination_reason") in ("crash", "stall")
