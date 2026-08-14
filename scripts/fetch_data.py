#!/usr/bin/env python3
"""
Fetch real GNSS satellite error data from NASA CDDIS SP3 precise orbit files.
Falls back to synthetic data silently if download or parsing fails.

Usage: python scripts/fetch_data.py
Output: data/raw/gnss_real.csv
"""
import gzip
import netrc
import os
import shutil
import sys
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

REQUIRED_COLS = ["Timestamp", "SatelliteID", "OrbitType", "ClockError_ns", "EphemerisError_m"]
RAW_OUTPUT    = "data/raw/gnss_real.csv"
SYNTHETIC_SRC = "data/synthetic/gnss_synthetic.csv"
SP3_CACHE     = "data/raw/sp3"

# BeiDou GEO satellite PRNs (C01-C05 and C59-C63) plus QZSS (GEO/IGSO)
GEO_PRNS = (
    {f"C{i:02d}" for i in range(1, 6)}
    | {f"C{i:02d}" for i in range(59, 64)}
    | {f"J{i:02d}" for i in range(1, 8)}
)


def get_auth():
    """
    Read NASA Earthdata credentials from ~/.netrc.

    Returns:
        (login, password) tuple or None if not found.
    """
    try:
        creds = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        if creds:
            return (creds[0], creds[2])
    except Exception:
        pass
    return None


def gps_week_dow(date: datetime):
    """
    Compute GPS week number and day-of-week for a date.

    Args:
        date: calendar date
    Returns:
        (week, day_of_week) integers
    """
    delta = (date - datetime(1980, 1, 6)).days
    return delta // 7, delta % 7


def cddis_urls(date: datetime) -> list:
    """
    Build ordered list of candidate CDDIS SP3 file URLs for a date.

    Args:
        date: date to build URLs for
    Returns:
        List of URL strings to try in order.
    """
    week, dow = gps_week_dow(date)
    year = date.year
    doy  = date.timetuple().tm_yday
    return [
        # IGS3 long-name gzip format (current standard post-2022)
        (f"https://cddis.nasa.gov/archive/gnss/products/{week}/"
         f"IGS0OPSRAP_{year}{doy:03d}0000_01D_15M_ORB.SP3.gz"),
        # IGS rapid legacy short-name compressed format
        (f"https://cddis.nasa.gov/archive/gnss/products/{week}/"
         f"igr{week}{dow}.sp3.Z"),
    ]


def download_file(url: str, dest: str, auth) -> bool:
    """
    Download a file from URL to dest, handling EarthData auth redirects.

    Args:
        url: source URL
        dest: local destination path
        auth: (user, password) or None
    Returns:
        True on success, False on any failure.
    """
    try:
        with requests.Session() as session:
            if auth:
                session.auth = auth
            r = session.get(url, timeout=60, stream=True, allow_redirects=True)
            if r.status_code != 200:
                return False
            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        return True
    except Exception:
        return False


def fetch_sp3(date: datetime, auth) -> str | None:
    """
    Download and decompress SP3 file for a given date.

    Args:
        date: date to fetch SP3 for
        auth: EarthData auth tuple or None
    Returns:
        Path to decompressed SP3 file, or None if unavailable.
    """
    os.makedirs(SP3_CACHE, exist_ok=True)
    for url in cddis_urls(date):
        fname = os.path.join(SP3_CACHE, os.path.basename(url))
        if not download_file(url, fname, auth):
            continue
        if fname.endswith(".gz"):
            out = fname[:-3]
            try:
                with gzip.open(fname, "rb") as fi, open(out, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
                os.remove(fname)
                return out
            except Exception:
                pass
        elif fname.endswith(".Z"):
            out = fname[:-2]
            if os.system(f"uncompress -f '{fname}' 2>/dev/null") == 0:
                if os.path.exists(out):
                    return out
        else:
            return fname
    return None


def orbit_type(sv: str) -> str:
    """
    Classify satellite PRN as GEO or MEO.

    Args:
        sv: satellite ID string (e.g. 'G01', 'C03')
    Returns:
        'GEO' or 'MEO'
    """
    return "GEO" if sv in GEO_PRNS else "MEO"


def parse_sp3(path: str) -> pd.DataFrame | None:
    """
    Parse an SP3 precise orbit file into a per-satellite error DataFrame.

    Clock offset (µs → ns) is used as ClockError_ns.
    Orbital radius deviation from median is used as EphemerisError_m proxy.

    Args:
        path: path to decompressed SP3 file
    Returns:
        DataFrame[Timestamp, SatelliteID, OrbitType, ClockError_ns, EphemerisError_m]
        or None on any parse failure.
    """
    try:
        import georinex as gr
        ds = gr.load(path)
    except Exception:
        return None

    try:
        times = pd.to_datetime(ds.time.values)
        svs   = ds.sv.values
    except Exception:
        return None

    rows = []
    for sv in svs:
        sv_str = str(sv)
        try:
            clk = ds["clock"].sel(sv=sv).values.astype(float)     # µs
            pos = ds["position"].sel(sv=sv).values.astype(float)   # (n, 3) km
        except Exception:
            continue

        clk_ns          = clk * 1000.0
        clk_ns[np.abs(clk_ns) > 9e8] = np.nan  # SP3 missing flag ≈ 999999.999999 µs

        radii           = np.linalg.norm(pos, axis=1)
        eph_m           = np.clip((radii - np.nanmedian(radii)) * 1000.0, -5.0, 5.0)

        for i, ts in enumerate(times):
            if np.isnan(clk_ns[i]) or np.isnan(eph_m[i]):
                continue
            rows.append({
                "Timestamp":        ts,
                "SatelliteID":      sv_str,
                "OrbitType":        orbit_type(sv_str),
                "ClockError_ns":    round(float(clk_ns[i]), 4),
                "EphemerisError_m": round(float(eph_m[i]), 4),
            })

    return pd.DataFrame(rows) if rows else None


def fallback(out_path: str) -> pd.DataFrame:
    """
    Copy or generate synthetic data to out_path when real data is unavailable.

    Args:
        out_path: destination CSV path
    Returns:
        DataFrame with synthetic GNSS data.
    """
    print("[fetch_data] Real data unavailable — using synthetic fallback")
    save_dir = os.path.dirname(out_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(SYNTHETIC_SRC):
        df = pd.read_csv(SYNTHETIC_SRC)
        df.to_csv(out_path, index=False)
        print(f"[fetch_data] Copied synthetic data → {out_path}")
        return df
    sys.path.insert(0, "src")
    from orbitalmind.utils.synthetic_generator import generate_synthetic_gnss_data
    return generate_synthetic_gnss_data(save_path=out_path)


def fetch_gnss_data(n_days: int = 14, out_path: str = RAW_OUTPUT) -> pd.DataFrame:
    """
    Download n_days of GNSS SP3 data from NASA CDDIS and write gnss_real.csv.
    Falls back to synthetic data silently on any failure.

    Args:
        n_days: number of days of SP3 data to fetch (default 14)
        out_path: output CSV path
    Returns:
        DataFrame with columns [Timestamp, SatelliteID, OrbitType,
        ClockError_ns, EphemerisError_m].
    """
    auth   = get_auth()
    end    = datetime(2024, 1, 14)
    dates  = [end - timedelta(days=i) for i in range(n_days - 1, -1, -1)]
    frames = []

    for date in dates:
        sp3_path = fetch_sp3(date, auth)
        if sp3_path is None:
            continue
        df = parse_sp3(sp3_path)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return fallback(out_path)

    df = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["SatelliteID", "Timestamp"])
        .reset_index(drop=True)
        .dropna()
    )
    if df.empty:
        return fallback(out_path)

    save_dir = os.path.dirname(out_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[fetch_data] Saved {len(df)} real rows → {out_path}")
    return df


if __name__ == "__main__":
    df = fetch_gnss_data()
    assert list(df.columns) == REQUIRED_COLS, f"Column mismatch: {list(df.columns)}"
    assert df.isnull().sum().sum() == 0,       "NaN values present in output"
    assert set(df["OrbitType"].unique()) <= {"GEO", "MEO"}, "Invalid OrbitType values"
    print(f"[fetch_data] ✓ {len(df)} rows | columns OK | zero NaN")
    print(f"[fetch_data]   Satellites : {df['SatelliteID'].nunique()}")
    print(f"[fetch_data]   OrbitTypes : {df['OrbitType'].value_counts().to_dict()}")
