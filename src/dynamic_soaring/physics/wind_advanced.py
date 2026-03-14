"""Advanced wind models: Dryden turbulence, thermals, time-varying, composite."""

from __future__ import annotations

import math

import numpy as np

from dynamic_soaring.config import WindConfig
from dynamic_soaring.physics.wind import LogarithmicWindProfile, WindProfile


class DrydenTurbulenceProfile(WindProfile):
    """Dryden continuous turbulence model (MIL-F-8785C).

    Adds filtered white noise to a base wind profile using first-order
    lag filters for each axis. Produces spatially-correlated gusts.
    """

    def __init__(
        self,
        base_profile: WindProfile,
        config: WindConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.base = base_profile
        self.scale_length = config.dryden_scale_length
        self.intensity = config.dryden_intensity
        self._rng = rng or np.random.default_rng()
        # Filter state (first-order lag)
        self._gust_state = np.zeros(3)
        self._dt = 0.02  # assumed update rate

    def get_wind(self, position: np.ndarray, time: float = 0.0) -> np.ndarray:
        base_wind = self.base.get_wind(position)

        # Scale turbulence with altitude (weaker near surface)
        z = max(position[2], 0.5)
        sigma = self.intensity * (z / (z + 5.0))  # smooth ramp-up

        # First-order lag filter: tau * du/dt + u = sigma * noise
        tau = self.scale_length / max(np.linalg.norm(base_wind[:2]), 1.0)
        alpha_filter = self._dt / (tau + self._dt)

        noise = self._rng.normal(0, sigma, size=3)
        noise[2] *= 0.5  # reduced vertical turbulence
        self._gust_state = (1 - alpha_filter) * self._gust_state + alpha_filter * noise

        return base_wind + self._gust_state

    def get_gradient(self, position: np.ndarray) -> float:
        return self.base.get_gradient(position)


class ThermalUpdraftProfile(WindProfile):
    """Localized thermal updraft columns on top of a base profile.

    Each thermal is a Gaussian plume: w_z = w_max * exp(-r²/(2σ²)) * f(z)
    """

    def __init__(
        self,
        base_profile: WindProfile,
        config: WindConfig,
    ) -> None:
        self.base = base_profile
        self.centers = [(c[0], c[1]) for c in config.thermal_centers]
        self.strength = config.thermal_strength
        self.radius = config.thermal_radius

    def get_wind(self, position: np.ndarray, time: float = 0.0) -> np.ndarray:
        wind = self.base.get_wind(position).copy()

        z = position[2]
        # Altitude dependence: thermals strengthen with height, cap at ~300m
        z_factor = min(z / 50.0, 1.0) * math.exp(-max(z - 300, 0) / 200.0)

        for cx, cy in self.centers:
            r_sq = (position[0] - cx) ** 2 + (position[1] - cy) ** 2
            sigma_sq = self.radius ** 2
            updraft = self.strength * math.exp(-r_sq / (2 * sigma_sq)) * z_factor
            wind[2] += updraft

        return wind

    def get_gradient(self, position: np.ndarray) -> float:
        return self.base.get_gradient(position)


class TimeVaryingWindProfile(WindProfile):
    """Wind with sinusoidal speed and direction variation over time."""

    def __init__(self, base_profile: WindProfile, config: WindConfig) -> None:
        self.base = base_profile
        self.period = config.time_varying_period
        self.amplitude = config.time_varying_amplitude
        self.base_speed = config.reference_speed

    def get_wind(self, position: np.ndarray, time: float = 0.0) -> np.ndarray:
        wind = self.base.get_wind(position).copy()

        # Sinusoidal speed modulation
        phase = 2 * math.pi * time / self.period
        speed_factor = 1.0 + self.amplitude * math.sin(phase)

        wind[:2] *= speed_factor
        return wind

    def get_gradient(self, position: np.ndarray) -> float:
        return self.base.get_gradient(position)


class CompositeWindProfile(WindProfile):
    """Chains multiple wind effects: base + turbulence + thermals + time variation."""

    def __init__(self, config: WindConfig, rng: np.random.Generator | None = None) -> None:
        self.config = config
        base = LogarithmicWindProfile(config, rng)
        profile: WindProfile = base

        if config.dryden_enabled:
            profile = DrydenTurbulenceProfile(profile, config, rng)
        if config.thermal_enabled and config.thermal_centers:
            profile = ThermalUpdraftProfile(profile, config)
        if config.time_varying:
            profile = TimeVaryingWindProfile(profile, config)

        self._profile = profile

    def get_wind(self, position: np.ndarray, time: float = 0.0) -> np.ndarray:
        return self._profile.get_wind(position, time)

    def get_gradient(self, position: np.ndarray) -> float:
        return self._profile.get_gradient(position)
