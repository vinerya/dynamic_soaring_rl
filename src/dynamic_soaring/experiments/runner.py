"""Unified experiment runner for ablation studies and comparisons."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from dynamic_soaring.config import Config
from dynamic_soaring.envs.soaring_env import DynamicSoaringEnv
from dynamic_soaring.evaluation.evaluate import evaluate_policy
from dynamic_soaring.experiments.experiment_config import ExperimentSpec
from dynamic_soaring.training.train import train


def run_single(config: Config, run_dir: Path, n_eval_episodes: int = 50) -> dict:
    """Train + evaluate a single run."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Override log/checkpoint dirs
    config.training.log_dir = str(run_dir / "logs")
    config.training.checkpoint_dir = str(run_dir / "checkpoints")
    config.to_yaml(run_dir / "config.yaml")

    # Train
    model = train(config)
    model_path = str(run_dir / "checkpoints" / "final_model.zip")

    # Evaluate
    results = evaluate_policy(
        model_path=model_path,
        config=config,
        n_episodes=n_eval_episodes,
        seed=config.training.seed + 10000,
    )

    # Save results
    # Separate trajectory data (large) from summary stats
    episodes = results.pop("episodes", [])
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save trajectory summaries (without full arrays)
    episode_summaries = []
    for ep in episodes:
        summary = {k: v for k, v in ep.items() if k != "cycles"}
        summary["n_cycles"] = len(ep.get("cycles", []))
        episode_summaries.append(summary)
    with open(run_dir / "episode_summaries.json", "w") as f:
        json.dump(episode_summaries, f, indent=2)

    results["episodes"] = episodes
    return results


def run_experiment(spec: ExperimentSpec) -> dict[str, dict[str, list]]:
    """Run all conditions × seeds in an experiment."""
    base_config = Config.from_yaml(spec.base_config_path)
    all_results: dict[str, dict[str, list]] = {}

    for condition in spec.conditions:
        condition_results: dict[str, list] = {"rewards": [], "durations": [], "energy_changes": []}

        for seed in spec.seeds:
            config = base_config.merge_overrides(condition.overrides)
            config.training.seed = seed
            run_dir = spec.run_dir(condition.name, seed)

            print(f"\n{'='*60}")
            print(f"Running: {condition.name} | seed={seed}")
            print(f"Output: {run_dir}")
            print(f"{'='*60}\n")

            results = run_single(config, run_dir, spec.n_eval_episodes)
            condition_results["rewards"].append(results["reward_mean"])
            condition_results["durations"].append(results["duration_mean"])
            condition_results["energy_changes"].append(results["energy_change_mean"])

        all_results[condition.name] = condition_results

    # Save aggregate results
    output_dir = Path(spec.output_dir) / spec.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "aggregate_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results
