"""Interactive 3D trajectory visualization using PyVista."""

from __future__ import annotations

import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from dynamic_soaring.config import Config
from dynamic_soaring.physics.wind import create_wind_profile


def plot_trajectory_3d(
    trajectory: np.ndarray,
    config: Config,
    color_by: str = "airspeed",
    show_wind: bool = True,
    title: str = "Dynamic Soaring Trajectory",
) -> None:
    """Render an interactive 3D trajectory with PyVista.

    Args:
        trajectory: (N, 6) array of [x, y, z, vx, vy, vz]
        config: simulation configuration
        color_by: "airspeed", "altitude", or "energy"
        show_wind: whether to show wind vector arrows
        title: plot title
    """
    if not HAS_PYVISTA:
        raise ImportError("PyVista is required for 3D visualization. Install with: pip install pyvista")

    positions = trajectory[:, :3]
    velocities = trajectory[:, 3:]

    wind_profile = create_wind_profile(config.wind)

    # Compute color values
    if color_by == "airspeed":
        winds = np.array([wind_profile.get_wind(p) for p in positions])
        v_air = velocities - winds
        scalars = np.linalg.norm(v_air, axis=1)
        scalar_name = "Airspeed (m/s)"
    elif color_by == "altitude":
        scalars = positions[:, 2]
        scalar_name = "Altitude (m)"
    elif color_by == "energy":
        winds = np.array([wind_profile.get_wind(p) for p in positions])
        v_air = velocities - winds
        airspeeds = np.linalg.norm(v_air, axis=1)
        scalars = 0.5 * airspeeds ** 2 + config.sim.g * positions[:, 2]
        scalar_name = "Total Energy (J/kg)"
    else:
        scalars = positions[:, 2]
        scalar_name = "Altitude (m)"

    # Create plotter
    plotter = pv.Plotter()
    plotter.set_background("lightblue", top="white")

    # Trajectory as a spline tube
    points = positions
    spline = pv.Spline(points, n_points=len(points))
    spline["scalars"] = scalars
    tube = spline.tube(radius=0.5)
    plotter.add_mesh(
        tube, scalars="scalars", cmap="plasma",
        scalar_bar_args={"title": scalar_name},
    )

    # Start and end markers
    plotter.add_mesh(
        pv.Sphere(radius=2.0, center=positions[0]),
        color="green", label="Start",
    )
    plotter.add_mesh(
        pv.Sphere(radius=2.0, center=positions[-1]),
        color="red", label="End",
    )

    # Ocean surface
    x_range = positions[:, 0]
    y_range = positions[:, 1]
    margin = 50
    ocean = pv.Plane(
        center=(np.mean(x_range), np.mean(y_range), 0),
        i_size=np.ptp(x_range) + 2 * margin,
        j_size=np.ptp(y_range) + 2 * margin,
    )
    plotter.add_mesh(ocean, color="steelblue", opacity=0.4, label="Ocean Surface")

    # Wind vectors at multiple altitudes
    if show_wind:
        center_x = np.mean(x_range)
        center_y = np.mean(y_range)
        wind_heights = np.linspace(5, max(positions[:, 2]) * 0.8, 6)

        arrow_points = []
        arrow_vectors = []
        for z in wind_heights:
            pos = np.array([center_x, center_y, z])
            w = wind_profile.get_wind(pos)
            arrow_points.append(pos)
            arrow_vectors.append(w)

        if arrow_points:
            arrow_pts = np.array(arrow_points)
            arrow_vecs = np.array(arrow_vectors)
            # Scale arrows for visibility
            max_wind = np.max(np.linalg.norm(arrow_vecs, axis=1))
            if max_wind > 0:
                scale = 20.0 / max_wind
                arrow_cloud = pv.PolyData(arrow_pts)
                arrow_cloud["vectors"] = arrow_vecs * scale
                arrows = arrow_cloud.glyph(
                    orient="vectors", scale="vectors", factor=1.0,
                    geom=pv.Arrow(),
                )
                plotter.add_mesh(arrows, color="navy", opacity=0.7, label="Wind")

    plotter.add_legend()
    plotter.add_title(title)
    plotter.add_axes()
    plotter.show()


def plot_trajectory_matplotlib(
    trajectory: np.ndarray,
    config: Config,
    save_path: str | None = None,
) -> None:
    """Fallback 3D trajectory plot using matplotlib."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    positions = trajectory[:, :3]
    wind_profile = create_wind_profile(config.wind)
    winds = np.array([wind_profile.get_wind(p) for p in positions])
    v_air = trajectory[:, 3:] - winds
    airspeeds = np.linalg.norm(v_air, axis=1)

    scatter = ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=airspeeds, cmap="plasma", s=1,
    )
    ax.plot(*positions[0], "go", markersize=10, label="Start")
    ax.plot(*positions[-1], "ro", markersize=10, label="End")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title("Dynamic Soaring Trajectory")
    ax.legend()
    fig.colorbar(scatter, label="Airspeed (m/s)", shrink=0.6)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
