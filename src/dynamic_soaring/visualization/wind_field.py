"""Wind profile visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dynamic_soaring.config import Config
from dynamic_soaring.physics.wind import create_wind_profile


def plot_wind_profile(config: Config, max_altitude: float = 100.0, save_path: str | None = None) -> None:
    """Plot wind speed and gradient vs altitude."""
    wind_profile = create_wind_profile(config.wind)

    altitudes = np.linspace(0.1, max_altitude, 200)
    speeds = []
    gradients = []

    for z in altitudes:
        pos = np.array([0.0, 0.0, z])
        wind = wind_profile.get_wind(pos)
        speeds.append(np.linalg.norm(wind[:2]))
        gradients.append(wind_profile.get_gradient(pos))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(speeds, altitudes, "b-", linewidth=2)
    ax1.set_xlabel("Wind Speed (m/s)")
    ax1.set_ylabel("Altitude (m)")
    ax1.set_title("Wind Speed Profile")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=config.wind.reference_height, color="r", linestyle="--",
                alpha=0.5, label=f"Ref height ({config.wind.reference_height}m)")
    ax1.legend()

    ax2.plot(gradients, altitudes, "r-", linewidth=2)
    ax2.set_xlabel("Wind Gradient dU/dz (1/s)")
    ax2.set_ylabel("Altitude (m)")
    ax2.set_title("Wind Gradient Profile")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Wind Profile: {config.wind.profile_type} (U_ref={config.wind.reference_speed} m/s)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
