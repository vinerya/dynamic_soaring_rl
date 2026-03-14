"""Wind profile models for atmospheric boundary layer."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from dynamic_soaring.config import WindConfig


class WindProfile(ABC):
    """Base class for wind profiles."""

    @abstractmethod
    def get_wind(self, position: np.ndarray) -> np.ndarray:
        """Return 3D wind velocity vector at the given position."""

    @abstractmethod
    def get_gradient(self, position: np.ndarray) -> float:
        """Return dU/dz (wind speed gradient) at the given position."""


class LogarithmicWindProfile(WindProfile):
    """Logarithmic wind profile: U(z) = U_ref * ln(z/z0) / ln(z_ref/z0).

    Standard model for neutral atmospheric boundary layer over ocean.
    """

    def __init__(self, config: WindConfig, rng: np.random.Generator | None = None) -> None:
        self.config = config
        self.z0 = config.surface_roughness
        self.u_ref = config.reference_speed
        self.z_ref = config.reference_height
        self._log_ratio = math.log(self.z_ref / self.z0)
        self._direction = config.direction
        self._direction_shear = config.direction_shear
        self._turbulence = config.turbulence_intensity
        self._rng = rng or np.random.default_rng()

    def _speed_at(self, z: float) -> float:
        """Wind speed magnitude at altitude z."""
        if z <= self.z0:
            return 0.0
        return self.u_ref * math.log(z / self.z0) / self._log_ratio

    def _direction_at(self, z: float) -> float:
        """Wind direction at altitude z (includes Ekman shear)."""
        return self._direction + self._direction_shear * z

    def get_wind(self, position: np.ndarray) -> np.ndarray:
        z = max(position[2], 0.0)
        speed = self._speed_at(z)
        direction = self._direction_at(z)

        wind = np.array([
            speed * math.cos(direction),
            speed * math.sin(direction),
            0.0,
        ])

        if self._turbulence > 0:
            noise = self._rng.normal(0, self._turbulence * speed, size=3)
            noise[2] *= 0.5  # reduced vertical turbulence
            wind += noise

        return wind

    def get_gradient(self, position: np.ndarray) -> float:
        """dU/dz = U_ref / (z * ln(z_ref/z0))."""
        z = max(position[2], self.z0 + 1e-6)
        return self.u_ref / (z * self._log_ratio)


class PowerLawWindProfile(WindProfile):
    """Power-law wind profile: U(z) = U_ref * (z/z_ref)^alpha."""

    def __init__(self, config: WindConfig, rng: np.random.Generator | None = None) -> None:
        self.config = config
        self.u_ref = config.reference_speed
        self.z_ref = config.reference_height
        self.exponent = config.power_law_exponent
        self._direction = config.direction
        self._rng = rng or np.random.default_rng()

    def _speed_at(self, z: float) -> float:
        if z <= 0:
            return 0.0
        return self.u_ref * (z / self.z_ref) ** self.exponent

    def get_wind(self, position: np.ndarray) -> np.ndarray:
        z = max(position[2], 0.0)
        speed = self._speed_at(z)
        return np.array([
            speed * math.cos(self._direction),
            speed * math.sin(self._direction),
            0.0,
        ])

    def get_gradient(self, position: np.ndarray) -> float:
        z = max(position[2], 1e-6)
        return self.u_ref * self.exponent * (z / self.z_ref) ** (self.exponent - 1) / self.z_ref


def create_wind_profile(config: WindConfig, rng: np.random.Generator | None = None) -> WindProfile:
    """Factory function to create a wind profile from config."""
    if config.profile_type == "logarithmic":
        return LogarithmicWindProfile(config, rng)
    elif config.profile_type == "power_law":
        return PowerLawWindProfile(config, rng)
    else:
        raise ValueError(f"Unknown wind profile type: {config.profile_type}")
