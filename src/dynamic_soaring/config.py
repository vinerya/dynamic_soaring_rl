"""Centralized configuration using dataclasses with YAML support."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BirdConfig:
    """Wandering albatross physical parameters."""

    mass: float = 8.5  # kg
    wing_area: float = 0.65  # m^2
    wing_span: float = 3.1  # m
    aspect_ratio: float = 15.0
    cd0: float = 0.033  # zero-lift drag coefficient
    oswald_efficiency: float = 0.9
    cl_max: float = 1.5  # max lift coefficient before stall
    alpha_stall: float = 0.2618  # ~15 deg in radians
    cl_alpha: float = 5.7  # lift curve slope (1/rad), finite wing corrected


@dataclass
class WindConfig:
    """Wind profile parameters."""

    profile_type: str = "logarithmic"  # "logarithmic" or "power_law"
    reference_speed: float = 15.0  # m/s at reference height
    reference_height: float = 10.0  # m (meteorological standard)
    surface_roughness: float = 0.001  # m (ocean surface z0)
    power_law_exponent: float = 0.143  # 1/7 for neutral stability
    direction: float = 0.0  # radians, wind blows in +x by default
    direction_shear: float = 0.0  # rad/m altitude (Ekman spiral)
    turbulence_intensity: float = 0.0  # fraction, 0 = none


@dataclass
class SimConfig:
    """Simulation parameters."""

    dt: float = 0.02  # seconds
    g: float = 9.81
    rho: float = 1.225  # kg/m^3 sea-level air density
    max_episode_steps: int = 3000  # 60s at dt=0.02
    altitude_limit: float = 500.0  # m
    min_airspeed: float = 8.0  # m/s stall speed
    max_bank_angle: float = 1.0472  # 60 deg
    min_alpha: float = -0.0873  # -5 deg
    max_alpha: float = 0.2618  # 15 deg
    max_load_factor: float = 3.0  # structural limit
    init_altitude_range: tuple[float, float] = (20.0, 50.0)
    init_airspeed_range: tuple[float, float] = (12.0, 18.0)


@dataclass
class RewardConfig:
    """Reward function weights."""

    energy_weight: float = 1.0
    energy_scale: float = 981.0  # g * 100m reference
    survival_bonus: float = 0.01
    smoothness_weight: float = 0.05
    crash_penalty: float = -10.0
    stall_penalty: float = -5.0


@dataclass
class TrainingConfig:
    """SB3 training hyperparameters."""

    algorithm: str = "PPO"
    total_timesteps: int = 2_000_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    n_epochs: int = 10
    n_steps: int = 2048
    n_envs: int = 8
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    seed: int = 42
    log_dir: str = "logs/"
    checkpoint_dir: str = "checkpoints/"
    checkpoint_freq: int = 50_000
    eval_freq: int = 10_000
    eval_episodes: int = 10
    normalize_obs: bool = True
    normalize_reward: bool = True


@dataclass
class Config:
    """Top-level configuration."""

    bird: BirdConfig = field(default_factory=BirdConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load config from YAML, merging with defaults."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Config:
        config = cls()
        section_map = {
            "bird": (BirdConfig, config.bird),
            "wind": (WindConfig, config.wind),
            "sim": (SimConfig, config.sim),
            "reward": (RewardConfig, config.reward),
            "training": (TrainingConfig, config.training),
        }
        for key, (dc_cls, default_inst) in section_map.items():
            if key in data:
                section_data = data[key]
                valid_fields = {f.name for f in fields(dc_cls)}
                filtered = {k: v for k, v in section_data.items() if k in valid_fields}
                # Handle tuple fields
                for f in fields(dc_cls):
                    if f.name in filtered and f.type.startswith("tuple"):
                        filtered[f.name] = tuple(filtered[f.name])
                setattr(config, key, dc_cls(**{**asdict(default_inst), **filtered}))
        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML."""
        data = {}
        for section_name in ("bird", "wind", "sim", "reward", "training"):
            section = getattr(self, section_name)
            d = asdict(section)
            # Convert tuples to lists for YAML
            for k, v in d.items():
                if isinstance(v, tuple):
                    d[k] = list(v)
            data[section_name] = d
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @property
    def induced_drag_factor(self) -> float:
        """k = 1 / (pi * e * AR)."""
        return 1.0 / (math.pi * self.bird.oswald_efficiency * self.bird.aspect_ratio)
