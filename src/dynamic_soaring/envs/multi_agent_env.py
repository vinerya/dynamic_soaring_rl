"""Multi-agent dynamic soaring environment."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dynamic_soaring.config import Config
from dynamic_soaring.envs.rewards import compute_reward
from dynamic_soaring.physics.aerodynamics import compute_aero_forces, lift_coefficient
from dynamic_soaring.physics.dynamics import compute_total_energy, rk4_step
from dynamic_soaring.physics.wake_model import compute_all_wake_effects, compute_circulation
from dynamic_soaring.physics.wind import create_wind_profile


class MultiAgentSoaringEnv(gym.Env):
    """Multi-agent environment for cooperative dynamic soaring.

    N agents fly simultaneously, optionally sharing observations and
    experiencing wake interactions from other agents.

    Observations per agent: base(13) + optional neighbor info.
    Actions per agent: [alpha_cmd, bank_cmd] in [-1, 1].

    For SB3 compatibility with parameter sharing, this env presents
    a single "super-agent" interface where obs/actions are concatenated.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Config | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or Config()
        self.render_mode = render_mode
        self.n_agents = self.config.multi_agent.n_agents
        self.comm_variant = self.config.multi_agent.comm_variant
        self.wake_enabled = self.config.multi_agent.wake_enabled

        # Per-agent obs size
        base_obs_size = 13
        if self.comm_variant == "shared_obs":
            k = min(self.config.multi_agent.n_neighbors, self.n_agents - 1)
            self._obs_size = base_obs_size + k * 6  # relative pos + vel of neighbors
        elif self.comm_variant == "full_comm":
            self._obs_size = base_obs_size + (self.n_agents - 1) * 6
        else:
            self._obs_size = base_obs_size

        # Super-agent spaces (all agents concatenated)
        total_obs = self.n_agents * self._obs_size
        total_act = self.n_agents * 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(total_obs,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(total_act,), dtype=np.float32)

        self._states = np.zeros((self.n_agents, 6))
        self._alphas = np.zeros(self.n_agents)
        self._banks = np.zeros(self.n_agents)
        self._prev_actions = None
        self._step_count = 0
        self._alive = np.ones(self.n_agents, dtype=bool)
        self._wind_profile = None
        self._rng = np.random.default_rng()
        self.trajectories: list[list[np.ndarray]] = [[] for _ in range(self.n_agents)]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        sim = self.config.sim

        for i in range(self.n_agents):
            z0 = self._rng.uniform(*sim.init_altitude_range)
            v0 = self._rng.uniform(*sim.init_airspeed_range)
            heading = self._rng.uniform(0, 2 * math.pi)
            # Spread agents out spatially
            offset_x = i * 20.0
            offset_y = self._rng.uniform(-10, 10)
            self._states[i] = [offset_x, offset_y, z0,
                               v0 * math.cos(heading), v0 * math.sin(heading), 0.0]

        self._alphas[:] = 0.0
        self._banks[:] = 0.0
        self._prev_actions = None
        self._step_count = 0
        self._alive[:] = True
        self.trajectories = [[s.copy()] for s in self._states]
        self._wind_profile = create_wind_profile(self.config.wind, self._rng)

        return self._build_all_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        sim = self.config.sim
        bird = self.config.bird

        agent_actions = action.reshape(self.n_agents, 2)

        # Compute wake interactions if enabled
        wake_velocities = np.zeros((self.n_agents, 3))
        if self.wake_enabled and np.sum(self._alive) > 1:
            circulations = np.zeros(self.n_agents)
            for i in range(self.n_agents):
                if self._alive[i]:
                    cl = lift_coefficient(self._alphas[i], bird)
                    wind = self._wind_profile.get_wind(self._states[i, :3])
                    airspeed = np.linalg.norm(self._states[i, 3:] - wind)
                    circulations[i] = compute_circulation(airspeed, cl, bird)
            wake_velocities = compute_all_wake_effects(
                self._states, circulations, bird, self.config.multi_agent.wake_decay_rate
            )

        total_reward = 0.0
        all_terminated = True

        for i in range(self.n_agents):
            if not self._alive[i]:
                continue

            # Map actions
            alpha = sim.min_alpha + (agent_actions[i, 0] + 1) * 0.5 * (sim.max_alpha - sim.min_alpha)
            bank = -sim.max_bank_angle + (agent_actions[i, 1] + 1) * 0.5 * (2 * sim.max_bank_angle)
            self._alphas[i] = alpha
            self._banks[i] = bank

            # Energy before
            wind_before = self._wind_profile.get_wind(self._states[i, :3]) + wake_velocities[i]
            e_before = compute_total_energy(self._states[i], wind_before, bird.mass, sim.g)

            # Physics step
            new_state, step_info = rk4_step(
                self._states[i], alpha, bank,
                self._wind_profile, bird, sim
            )
            self._states[i] = new_state
            self.trajectories[i].append(new_state.copy())

            # Energy after
            wind_after = self._wind_profile.get_wind(self._states[i, :3])
            e_after = compute_total_energy(self._states[i], wind_after, bird.mass, sim.g)

            # Check termination per agent
            z = self._states[i, 2]
            if z <= 0 or step_info["airspeed"] < sim.min_airspeed or z > sim.altitude_limit:
                self._alive[i] = False

            # Reward
            prev_a = self._prev_actions[i] if self._prev_actions is not None else None
            reason = None if self._alive[i] else "crash"
            r = compute_reward(e_before, e_after, agent_actions[i], prev_a,
                               not self._alive[i], reason, self.config.reward)
            total_reward += r

            if self._alive[i]:
                all_terminated = False

        self._step_count += 1
        self._prev_actions = agent_actions.copy()

        terminated = all_terminated
        truncated = self._step_count >= sim.max_episode_steps

        avg_reward = total_reward / self.n_agents
        obs = self._build_all_obs()
        info = {"alive_count": int(np.sum(self._alive))}

        return obs, avg_reward, terminated, truncated, info

    def _build_all_obs(self) -> np.ndarray:
        """Build concatenated observations for all agents."""
        obs_list = []
        for i in range(self.n_agents):
            obs = self._build_agent_obs(i)
            obs_list.append(obs)
        return np.concatenate(obs_list).astype(np.float32)

    def _build_agent_obs(self, agent_idx: int) -> np.ndarray:
        """Build observation for a single agent."""
        from dynamic_soaring.envs.soaring_env import DynamicSoaringEnv

        state = self._states[agent_idx]
        pos, vel = state[:3], state[3:]
        wind = self._wind_profile.get_wind(pos)

        v_air = vel - wind
        airspeed = np.linalg.norm(v_air)
        ground_speed = np.linalg.norm(vel[:2])

        fpa = math.atan2(vel[2], max(ground_speed, 1e-6))
        heading = math.atan2(vel[1], vel[0])
        wind_speed = np.linalg.norm(wind[:2])
        wind_dir = math.atan2(wind[1], wind[0])
        relative_wind = ((wind_dir - heading + math.pi) % (2 * math.pi)) - math.pi
        wind_gradient = self._wind_profile.get_gradient(pos)
        e_ref = self.config.sim.g * 100.0
        specific_energy = (0.5 * airspeed ** 2 + self.config.sim.g * pos[2]) / e_ref

        base_obs = np.array([
            pos[2] / 100.0, airspeed / 20.0, ground_speed / 20.0,
            vel[2] / 10.0, fpa, heading / math.pi,
            self._banks[agent_idx], self._alphas[agent_idx] / 0.3,
            wind_speed / 20.0, wind_gradient / 2.0,
            relative_wind / math.pi, specific_energy, 1.0,
        ])

        if self.comm_variant == "no_comm":
            return base_obs

        # Add neighbor information
        if self.comm_variant == "shared_obs":
            k = min(self.config.multi_agent.n_neighbors, self.n_agents - 1)
        else:
            k = self.n_agents - 1

        # Compute relative positions/velocities to nearest k neighbors
        dists = []
        for j in range(self.n_agents):
            if j == agent_idx:
                continue
            d = np.linalg.norm(self._states[j, :3] - pos)
            dists.append((d, j))
        dists.sort()

        neighbor_obs = []
        for _, j in dists[:k]:
            rel_pos = (self._states[j, :3] - pos) / 100.0
            rel_vel = (self._states[j, 3:] - vel) / 20.0
            neighbor_obs.extend(rel_pos.tolist() + rel_vel.tolist())

        # Pad if fewer neighbors than k
        while len(neighbor_obs) < k * 6:
            neighbor_obs.extend([0.0] * 6)

        return np.concatenate([base_obs, np.array(neighbor_obs[:k * 6])])
