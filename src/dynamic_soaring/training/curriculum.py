"""Curriculum learning for dynamic soaring training."""

from __future__ import annotations

from dataclasses import dataclass

from stable_baselines3.common.callbacks import BaseCallback

from dynamic_soaring.config import CurriculumConfig


# Default 3-stage curriculum
DEFAULT_CURRICULUM = [
    {
        "name": "easy",
        "start_timestep": 0,
        "overrides": {
            "wind": {"reference_speed": 20.0},  # strong wind = easy energy extraction
            "sim": {"max_episode_steps": 1000, "init_altitude_range": [30.0, 50.0]},
            "reward": {"survival_bonus": 0.05},
        },
    },
    {
        "name": "medium",
        "start_timestep": 500_000,
        "overrides": {
            "wind": {"reference_speed": 15.0},
            "sim": {"max_episode_steps": 2000, "init_altitude_range": [20.0, 50.0]},
            "reward": {"survival_bonus": 0.02},
        },
    },
    {
        "name": "full",
        "start_timestep": 1_000_000,
        "overrides": {
            "wind": {"reference_speed": 15.0},
            "sim": {"max_episode_steps": 3000, "init_altitude_range": [20.0, 50.0]},
            "reward": {"survival_bonus": 0.01},
        },
    },
]


class CurriculumCallback(BaseCallback):
    """SB3 callback that adjusts environment parameters at curriculum stage boundaries.

    Uses reset(options=...) to pass stage-specific parameters to the environment.
    """

    def __init__(self, curriculum_config: CurriculumConfig, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.stages = curriculum_config.stages or DEFAULT_CURRICULUM
        self._current_stage_idx = 0
        self._stage_boundaries = sorted(
            [(s["start_timestep"], i) for i, s in enumerate(self.stages)],
            key=lambda x: x[0],
        )

    def _on_step(self) -> bool:
        timestep = self.num_timesteps

        # Check if we should advance to next stage
        new_stage_idx = self._current_stage_idx
        for boundary_ts, stage_idx in self._stage_boundaries:
            if timestep >= boundary_ts:
                new_stage_idx = stage_idx

        if new_stage_idx != self._current_stage_idx:
            stage = self.stages[new_stage_idx]
            self._current_stage_idx = new_stage_idx
            if self.verbose:
                print(f"\n[Curriculum] Advancing to stage '{stage['name']}' "
                      f"at timestep {timestep}")

        return True
