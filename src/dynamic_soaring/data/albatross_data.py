"""Load and process albatross GPS trajectory data."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


def load_albatross_gps(path: str | Path) -> list[dict]:
    """Load GPS trajectory data from CSV.

    Expected columns: timestamp, latitude, longitude, altitude
    Timestamps should be in seconds or ISO format.
    Returns list of dicts with lat, lon, alt, time fields.
    """
    records = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {}
            # Flexible column name handling
            for lat_key in ("latitude", "lat", "Latitude"):
                if lat_key in row:
                    record["lat"] = float(row[lat_key])
                    break
            for lon_key in ("longitude", "lon", "Longitude"):
                if lon_key in row:
                    record["lon"] = float(row[lon_key])
                    break
            for alt_key in ("altitude", "alt", "Altitude", "height"):
                if alt_key in row:
                    record["alt"] = float(row[alt_key])
                    break
            for time_key in ("timestamp", "time", "Time", "datetime"):
                if time_key in row:
                    try:
                        record["time"] = float(row[time_key])
                    except ValueError:
                        record["time"] = 0.0  # placeholder for ISO timestamps
                    break

            if "lat" in record and "lon" in record:
                record.setdefault("alt", 0.0)
                record.setdefault("time", 0.0)
                records.append(record)

    return records


def gps_to_enu(records: list[dict]) -> np.ndarray:
    """Convert GPS lat/lon/alt to local ENU (East-North-Up) coordinates.

    Uses first point as origin. Returns (N, 4) array: [east, north, up, time].
    """
    if not records:
        return np.zeros((0, 4))

    ref_lat = math.radians(records[0]["lat"])
    ref_lon = math.radians(records[0]["lon"])
    ref_alt = records[0]["alt"]

    R_earth = 6371000.0  # meters

    points = []
    for r in records:
        lat = math.radians(r["lat"])
        lon = math.radians(r["lon"])

        dlat = lat - ref_lat
        dlon = lon - ref_lon

        east = R_earth * dlon * math.cos(ref_lat)
        north = R_earth * dlat
        up = r["alt"] - ref_alt

        points.append([east, north, up, r["time"]])

    return np.array(points)


def interpolate_trajectory(enu_data: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """Resample irregular GPS data to uniform timestep.

    Args:
        enu_data: (N, 4) array [east, north, up, time]
        dt: desired timestep in seconds

    Returns (M, 6) trajectory array [x, y, z, vx, vy, vz].
    """
    times = enu_data[:, 3]
    if times[-1] <= times[0]:
        # No valid time data, assume uniform 1s spacing
        times = np.arange(len(enu_data)) * 1.0

    # Create uniform time grid
    t_uniform = np.arange(times[0], times[-1], dt)

    # Interpolate positions
    x = np.interp(t_uniform, times, enu_data[:, 0])
    y = np.interp(t_uniform, times, enu_data[:, 1])
    z = np.interp(t_uniform, times, enu_data[:, 2])

    # Compute velocities via finite differences
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    vz = np.gradient(z, dt)

    return np.column_stack([x, y, z, vx, vy, vz])


def filter_soaring_segments(
    trajectory: np.ndarray,
    dt: float,
    min_duration: float = 30.0,
    max_speed_variation: float = 0.3,
) -> list[np.ndarray]:
    """Extract segments likely to be dynamic soaring (sustained gliding).

    Filters for segments with:
    - Duration >= min_duration seconds
    - Relatively steady speed (low variation = no flapping)
    - Altitude oscillation (sign of dynamic soaring)
    """
    min_steps = int(min_duration / dt)
    speeds = np.linalg.norm(trajectory[:, 3:], axis=1)
    altitudes = trajectory[:, 2]

    segments = []
    start = 0

    while start < len(trajectory) - min_steps:
        end = start + min_steps

        # Extend segment while speed variation is low
        while end < len(trajectory):
            segment_speeds = speeds[start:end]
            cv = np.std(segment_speeds) / max(np.mean(segment_speeds), 1e-6)
            if cv > max_speed_variation:
                break
            end += 1

        if end - start >= min_steps:
            segment = trajectory[start:end]
            # Check for altitude oscillation (dynamic soaring signature)
            alt_range = np.max(altitudes[start:end]) - np.min(altitudes[start:end])
            if alt_range > 3.0:
                segments.append(segment)

        start = end

    return segments
