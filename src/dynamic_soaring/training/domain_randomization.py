"""Domain-randomized environment for generalization training."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from dynamic_soaring.config import Config
from dynamic_soaring.envs.soaring_env import DynamicSoaringEnv
from dynamic_soaring.physics.wind import create_wind_profile


class DomainRandomizedEnv(DynamicSoaringEnv):
    """Environment that randomizes wind parameters each episode.

    Trains a generalist agent robust to varying wind conditions.
    """

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # First call parent reset to set up RNG
        obs, info = super().reset(seed=seed, options=options)

        # Randomize wind parameters
        rand = self.config.randomization
        if rand.enabled:
            new_wind = self.config.wind.__class__(
                profile_type=self.config.wind.profile_type,
                reference_speed=float(self._rng.uniform(*rand.reference_speed_range)),
                reference_height=self.config.wind.reference_height,
                surface_roughness=float(self._rng.uniform(*rand.surface_roughness_range)),
                direction=float(self._rng.uniform(*rand.direction_range)),
                direction_shear=float(self._rng.uniform(*rand.direction_shear_range)),
                turbulence_intensity=float(self._rng.uniform(*rand.turbulence_range)),
            )
            self._wind_profile = create_wind_profile(new_wind, self._rng)
            info["wind_params"] = {
                "reference_speed": new_wind.reference_speed,
                "surface_roughness": new_wind.surface_roughness,
                "direction": new_wind.direction,
                "turbulence_intensity": new_wind.turbulence_intensity,
            }

        # Rebuild observation with new wind
        obs = self._build_observation()
        return obs, info


class ContextConditionedEnv(DomainRandomizedEnv):
    """Domain-randomized env that includes wind parameters in observation.

    Observation is extended from 13D to 17D with wind context:
    [base_obs(13), ref_speed_norm, roughness_norm, turbulence_norm, dir_shear_norm]
    """

    def __init__(self, config: Config | None = None, render_mode: str | None = None):
        super().__init__(config, render_mode)
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )
        self._wind_context = np.zeros(4, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)

        # Capture wind context for observation
        wind_params = info.get("wind_params", {})
        rand = self.config.randomization
        self._wind_context = np.array([
            wind_params.get("reference_speed", self.config.wind.reference_speed) / 20.0,
            math.log10(max(wind_params.get("surface_roughness", self.config.wind.surface_roughness), 1e-6)) / 4.0 + 1.0,
            wind_params.get("turbulence_intensity", self.config.wind.turbulence_intensity) / 0.2,
            wind_params.get("direction_shear", self.config.wind.direction_shear) / 0.005 if 0.005 > 0 else 0,
        ], dtype=np.float32)

        obs = np.concatenate([obs, self._wind_context])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs = np.concatenate([obs, self._wind_context])
        return obs, reward, terminated, truncated, info
