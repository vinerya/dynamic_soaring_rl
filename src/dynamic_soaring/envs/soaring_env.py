"""Gymnasium environment for dynamic soaring."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dynamic_soaring.config import Config
from dynamic_soaring.envs.rewards import compute_reward
from dynamic_soaring.physics.dynamics import compute_load_factor, compute_total_energy, rk4_step
from dynamic_soaring.physics.wind import create_wind_profile, WindProfile


class DynamicSoaringEnv(gym.Env):
    """Dynamic soaring environment where an agent learns albatross-style flight.

    Observation (13D): altitude, airspeed, ground_speed, climb_rate,
        flight_path_angle, heading, bank_angle, alpha, wind_speed,
        wind_gradient, relative_wind_dir, specific_energy, load_factor

    Action (2D): [angle_of_attack_cmd, bank_angle_cmd] in [-1, 1],
        mapped to physical ranges.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    # Normalization constants for observations
    _OBS_SCALES = np.array([
        100.0,   # altitude (m)
        20.0,    # airspeed (m/s)
        20.0,    # ground_speed (m/s)
        10.0,    # climb_rate (m/s)
        1.0,     # flight_path_angle (rad)
        math.pi, # heading (rad)
        1.0,     # bank_angle (rad)
        0.3,     # alpha (rad)
        20.0,    # wind_speed (m/s)
        2.0,     # wind_gradient (1/s)
        math.pi, # relative_wind_dir (rad)
        1.0,     # specific_energy (normalized)
        3.0,     # load_factor
    ], dtype=np.float32)

    def __init__(self, config: Config | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        self.config = config or Config()
        self.render_mode = render_mode

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Internal state
        self._state: np.ndarray = np.zeros(6)  # [x, y, z, vx, vy, vz]
        self._alpha: float = 0.0
        self._bank: float = 0.0
        self._prev_action: np.ndarray | None = None
        self._step_count: int = 0
        self._wind_profile: WindProfile | None = None
        self._rng: np.random.Generator = np.random.default_rng()

        # Trajectory recording
        self.trajectory: list[np.ndarray] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        sim = self.config.sim
        # Random initial conditions
        z0 = self._rng.uniform(*sim.init_altitude_range)
        v0 = self._rng.uniform(*sim.init_airspeed_range)
        heading = self._rng.uniform(0, 2 * math.pi)

        self._state = np.array([
            0.0, 0.0, z0,
            v0 * math.cos(heading),
            v0 * math.sin(heading),
            0.0,
        ])
        self._alpha = 0.0
        self._bank = 0.0
        self._prev_action = None
        self._step_count = 0
        self.trajectory = [self._state.copy()]

        # Create wind profile (deterministic from config, turbulence uses rng)
        self._wind_profile = create_wind_profile(self.config.wind, self._rng)

        obs = self._build_observation()
        info = self._build_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0)
        sim = self.config.sim

        # Map actions to physical range
        alpha_cmd = sim.min_alpha + (action[0] + 1.0) * 0.5 * (sim.max_alpha - sim.min_alpha)
        bank_cmd = -sim.max_bank_angle + (action[1] + 1.0) * 0.5 * (2 * sim.max_bank_angle)

        self._alpha = float(alpha_cmd)
        self._bank = float(bank_cmd)

        # Current simulation time
        current_time = self._step_count * sim.dt

        # Compute energy before step
        wind_before = self._wind_profile.get_wind(self._state[:3], current_time)
        energy_before = compute_total_energy(
            self._state, wind_before, self.config.bird.mass, sim.g
        )

        # Physics step (RK4)
        new_state, step_info = rk4_step(
            self._state, self._alpha, self._bank,
            self._wind_profile, self.config.bird, sim, current_time
        )

        self._state = new_state
        self._step_count += 1
        self.trajectory.append(self._state.copy())

        # Compute energy after step
        next_time = self._step_count * sim.dt
        wind_after = self._wind_profile.get_wind(self._state[:3], next_time)
        energy_after = compute_total_energy(
            self._state, wind_after, self.config.bird.mass, sim.g
        )

        # Check termination
        terminated, reason = self._check_terminated(step_info)
        truncated = self._step_count >= sim.max_episode_steps

        # Reward
        reward = compute_reward(
            energy_before, energy_after,
            action, self._prev_action,
            terminated, reason,
            self.config.reward,
        )

        self._prev_action = action.copy()

        obs = self._build_observation()
        info = self._build_info(step_info, reason)
        return obs, reward, terminated, truncated, info

    def _check_terminated(self, step_info: dict) -> tuple[bool, str | None]:
        """Check termination conditions."""
        sim = self.config.sim
        z = self._state[2]

        if z <= 0.0:
            return True, "crash"
        if step_info["airspeed"] < sim.min_airspeed:
            return True, "stall"
        if z > sim.altitude_limit:
            return True, "altitude"
        if abs(self._bank) > math.pi / 2:
            return True, "inverted"
        return False, None

    def _build_observation(self) -> np.ndarray:
        """Build 13D normalized observation vector."""
        pos = self._state[:3]
        vel = self._state[3:]
        current_time = self._step_count * self.config.sim.dt
        wind = self._wind_profile.get_wind(pos, current_time)
        v_air = vel - wind
        airspeed = np.linalg.norm(v_air)
        ground_speed = np.linalg.norm(vel[:2])
        horiz_speed = max(np.linalg.norm(v_air[:2]), 1e-6)

        # Flight angles
        fpa = math.atan2(vel[2], max(ground_speed, 1e-6))
        heading = math.atan2(vel[1], vel[0])

        # Wind info
        wind_speed = np.linalg.norm(wind[:2])
        wind_dir = math.atan2(wind[1], wind[0])
        relative_wind = wind_dir - heading
        # Wrap to [-pi, pi]
        relative_wind = (relative_wind + math.pi) % (2 * math.pi) - math.pi

        wind_gradient = self._wind_profile.get_gradient(pos)

        # Specific energy (normalized by reference)
        e_ref = self.config.sim.g * 100.0  # 100m reference
        specific_energy = (0.5 * airspeed ** 2 + self.config.sim.g * pos[2]) / e_ref

        # Load factor
        from dynamic_soaring.physics.aerodynamics import compute_aero_forces
        F_lift, _, _, _, _ = compute_aero_forces(
            vel, wind, self._alpha, self._bank, self.config.bird, self.config.sim.rho
        )
        load_factor = np.linalg.norm(F_lift) / (self.config.bird.mass * self.config.sim.g)

        raw_obs = np.array([
            pos[2],           # altitude
            airspeed,         # airspeed
            ground_speed,     # ground speed
            vel[2],           # climb rate
            fpa,              # flight path angle
            heading,          # heading
            self._bank,       # bank angle
            self._alpha,      # angle of attack
            wind_speed,       # wind speed at current alt
            wind_gradient,    # dU/dz
            relative_wind,    # relative wind direction
            specific_energy,  # normalized total energy
            load_factor,      # L/(mg)
        ], dtype=np.float32)

        return raw_obs / self._OBS_SCALES

    def _build_info(
        self,
        step_info: dict | None = None,
        termination_reason: str | None = None,
    ) -> dict[str, Any]:
        """Build info dict with raw state data for logging."""
        current_time = self._step_count * self.config.sim.dt
        wind = self._wind_profile.get_wind(self._state[:3], current_time)
        info: dict[str, Any] = {
            "position": self._state[:3].copy(),
            "velocity": self._state[3:].copy(),
            "altitude": float(self._state[2]),
            "alpha": self._alpha,
            "bank": self._bank,
            "wind": wind.copy(),
        }
        if step_info is not None:
            info["airspeed"] = step_info["airspeed"]
            info["cl"] = step_info["cl"]
            info["cd"] = step_info["cd"]
            info["lift"] = step_info["lift_mag"]
            info["drag"] = step_info["drag_mag"]
        if termination_reason is not None:
            info["termination_reason"] = termination_reason
        return info
