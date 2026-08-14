"""
Synthetic GNSS satellite error data generator — Iteration 1.
"""
import os
import numpy as np
import pandas as pd


def generate_synthetic_gnss_data(
    n_geo: int = 3,
    n_meo: int = 5,
    n_days: int = 8,
    interval_minutes: int = 15,
    seed: int = 42,
    save_path: str = "data/synthetic/gnss_synthetic.csv"
) -> pd.DataFrame:
    """
    Generate synthetic GNSS satellite error dataset.

    Args:
        n_geo: number of GEO satellites
        n_meo: number of MEO satellites
        n_days: total days to generate (7 train + 1 test)
        interval_minutes: sampling interval
        seed: random seed for reproducibility
        save_path: where to save the CSV

    Returns:
        DataFrame with columns:
        [Timestamp, SatelliteID, OrbitType, ClockError_ns, EphemerisError_m]
    """
    np.random.seed(seed)

    n_points = n_days * 24 * 60 // interval_minutes  # 768
    timestamps = pd.date_range(
        start="2024-01-01 00:00:00",
        periods=n_points,
        freq=f"{interval_minutes}min"
    )
    t = np.arange(n_points) * interval_minutes  # minutes from start

    rows = []

    for i in range(1, n_geo + 1):
        sat_id = f"GEO-{i:02d}"
        drift_rate = np.random.uniform(0.0005, 0.002)
        A_24 = np.random.uniform(1.5, 3.0)
        phase_24 = np.random.uniform(0, 2 * np.pi)
        noise_clock = np.random.normal(0, 1, n_points) * 0.3
        noise_eph = np.random.normal(0, 1, n_points) * 0.1

        clock = (
            drift_rate * t
            + A_24 * np.sin(2 * np.pi * t / 1440 + phase_24)
            + noise_clock
        )
        eph = (
            drift_rate * 0.5 * t
            + 0.8 * np.sin(2 * np.pi * t / 1440 + phase_24 + 0.5)
            + noise_eph
        )

        iod_offset = np.zeros(n_points)
        for j in range(8, n_points, 8):
            if np.random.random() < 0.3:
                iod_offset[j:] += np.random.uniform(-0.5, 0.5)

        clock += iod_offset
        clock = np.clip(clock, -20.0, 20.0)
        eph = np.clip(eph, -5.0, 5.0)

        for k in range(n_points):
            rows.append({
                "Timestamp": timestamps[k],
                "SatelliteID": sat_id,
                "OrbitType": "GEO",
                "ClockError_ns": clock[k],
                "EphemerisError_m": eph[k],
            })

    for i in range(1, n_meo + 1):
        sat_id = f"MEO-{i:02d}"
        drift_rate = np.random.uniform(0.0003, 0.001)
        A_12 = np.random.uniform(1.0, 2.5)
        A_24 = np.random.uniform(0.3, 0.8)
        phase_12 = np.random.uniform(0, 2 * np.pi)
        phase_24 = np.random.uniform(0, 2 * np.pi)
        noise_clock = np.random.normal(0, 1, n_points) * 0.25
        noise_eph = np.random.normal(0, 1, n_points) * 0.08

        clock = (
            drift_rate * t
            + A_12 * np.sin(2 * np.pi * t / 720 + phase_12)
            + A_24 * np.sin(2 * np.pi * t / 1440 + phase_24)
            + noise_clock
        )
        eph = (
            drift_rate * 0.4 * t
            + 0.6 * np.sin(2 * np.pi * t / 720 + phase_12 + 0.3)
            + noise_eph
        )

        iod_offset = np.zeros(n_points)
        for j in range(8, n_points, 8):
            if np.random.random() < 0.3:
                iod_offset[j:] += np.random.uniform(-0.5, 0.5)

        clock += iod_offset
        clock = np.clip(clock, -20.0, 20.0)
        eph = np.clip(eph, -5.0, 5.0)

        for k in range(n_points):
            rows.append({
                "Timestamp": timestamps[k],
                "SatelliteID": sat_id,
                "OrbitType": "MEO",
                "ClockError_ns": clock[k],
                "EphemerisError_m": eph[k],
            })

    df = pd.DataFrame(rows)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"Generated {len(df)} rows for {n_geo} GEO and {n_meo} MEO satellites")
    return df
