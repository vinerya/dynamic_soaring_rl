"""Custom SB3 callbacks for dynamic soaring metrics."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SoaringMetricsCallback(BaseCallback):
    """Log domain-specific soaring metrics to TensorBoard."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_energies: list[float] = []
        self._episode_altitudes: list[float] = []
        self._episode_airspeeds: list[float] = []
        self._termination_counts: dict[str, int] = {}

    def _on_step(self) -> bool:
        # Collect per-step info from vectorized envs
        infos = self.locals.get("infos", [])
        for info in infos:
            if "altitude" in info:
                self._episode_altitudes.append(info["altitude"])
            if "airspeed" in info:
                self._episode_airspeeds.append(info["airspeed"])

            # Check for episode end (in VecEnv, done episodes have "episode" key)
            if "episode" in info:
                ep_info = info["episode"]
                self.logger.record("soaring/episode_reward", ep_info["r"])
                self.logger.record("soaring/episode_length", ep_info["l"])

                if self._episode_altitudes:
                    self.logger.record(
                        "soaring/max_altitude", max(self._episode_altitudes)
                    )
                    self.logger.record(
                        "soaring/mean_altitude",
                        np.mean(self._episode_altitudes),
                    )
                if self._episode_airspeeds:
                    self.logger.record(
                        "soaring/mean_airspeed",
                        np.mean(self._episode_airspeeds),
                    )

                # Termination reason
                reason = info.get("termination_reason", "truncated")
                self._termination_counts[reason] = (
                    self._termination_counts.get(reason, 0) + 1
                )
                total = sum(self._termination_counts.values())
                for r, count in self._termination_counts.items():
                    self.logger.record(
                        f"soaring/termination_{r}_pct", count / total * 100
                    )

                # Reset per-episode tracking
                self._episode_altitudes.clear()
                self._episode_airspeeds.clear()

        return True
