"""Experiment specification for grid-search ablation studies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ConditionSpec:
    """A single experimental condition."""

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentSpec:
    """Full experiment specification: conditions × seeds."""

    experiment_name: str
    base_config_path: str = "configs/default.yaml"
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    conditions: list[ConditionSpec] = field(default_factory=list)
    n_eval_episodes: int = 50
    output_dir: str = "results/"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentSpec:
        with open(path) as f:
            data = yaml.safe_load(f)
        conditions = [
            ConditionSpec(name=c["name"], overrides=c.get("overrides", {}))
            for c in data.get("conditions", [])
        ]
        return cls(
            experiment_name=data["experiment_name"],
            base_config_path=data.get("base_config_path", "configs/default.yaml"),
            seeds=data.get("seeds", [42, 123, 456, 789, 1024]),
            conditions=conditions,
            n_eval_episodes=data.get("n_eval_episodes", 50),
            output_dir=data.get("output_dir", "results/"),
        )

    def run_dir(self, condition: str, seed: int) -> Path:
        return Path(self.output_dir) / self.experiment_name / condition / f"seed_{seed}"
