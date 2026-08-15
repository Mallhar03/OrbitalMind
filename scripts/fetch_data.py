#!/usr/bin/env python3
"""
Fetch real GNSS satellite error data from NASA CDDIS.

Downloads SP3 precise orbit files and RINEX broadcast navigation files.
Computes EphemerisError_m as signed radial broadcast-minus-precise position.
Falls back to synthetic data silently on any download or parse failure.

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
NAV_CACHE     = "data/raw/nav"

_GM      = 3.986005e14       # m^3/s^2  WGS-84 gravitational constant
_OMEGA_E = 7.2921151467e-5   # rad/s    Earth rotation rate
_GPS_EPOCH = datetime(1980, 1, 6)

# RINEX 2 GPS nav stores these angles in semi-circles; RINEX 3 uses radians.
_SC_FIELDS = frozenset(["M0", "Omega0", "Io", "omega", "DeltaN", "OmegaDot", "IDOT"])

# BeiDou GEO and QZSS (GEO/IGSO) satellite PRNs
GEO_PRNS = (
    {f"C{i:02d}" for i in range(1, 6)}
    | {f"C{i:02d}" for i in range(59, 64)}
    | {f"J{i:02d}" for i in range(1, 8)}
)


def get_auth():
    """Read NASA Earthdata credentials from ~/.netrc."""
    try:
        creds = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        if creds:
            return (creds[0], creds[2])
    except Exception:
        pass
    return None


def gps_week_dow(date: datetime):
    """Return (GPS week, day-of-week) for date."""
    delta = (date - datetime(1980, 1, 6)).days
    return delta // 7, delta % 7


def cddis_urls(date: datetime) -> list:
    """Ordered list of candidate CDDIS SP3 file URLs for date."""
    week, dow = gps_week_dow(date)
    year = date.year
    doy  = date.timetuple().tm_yday
    return [
        (f"https://cddis.nasa.gov/archive/gnss/products/{week}/"
         f"IGS0OPSRAP_{year}{doy:03d}0000_01D_15M_ORB.SP3.gz"),
        (f"https://cddis.nasa.gov/archive/gnss/products/{week}/"
         f"igr{week}{dow}.sp3.Z"),
    ]


def cddis_nav_urls(date: datetime) -> list:
    """Ordered list of candidate CDDIS RINEX navigation file URLs for date."""
    year = date.year
    doy  = date.timetuple().tm_yday
    yy   = date.strftime("%y")
    return [
        # RINEX 3 mixed navigation (angles in radians — preferred)
        (f"https://cddis.nasa.gov/archive/gnss/data/daily/{year}/brdc/"
         f"BRDC00IGS_R_{year}{doy:03d}0000_01D_MN.rnx.gz"),
        # RINEX 2 GPS-only navigation (angles in semi-circles)
        (f"https://cddis.nasa.gov/archive/gnss/data/daily/{year}/{doy:03d}/{yy}n/"
         f"brdc{doy:03d}0.{yy}n.gz"),
    ]


def download_file(url: str, dest: str, auth) -> bool:
    """Download url to dest, following EarthData auth redirects."""
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


def _decompress_gz(fname: str) -> str | None:
    """Decompress .gz file, return decompressed path or None."""
    out = fname[:-3]
    try:
        with gzip.open(fname, "rb") as fi, open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        os.remove(fname)
        return out
    except Exception:
        return None


def fetch_sp3(date: datetime, auth) -> str | None:
    """Download and decompress SP3 file for date. Returns path or None."""
    os.makedirs(SP3_CACHE, exist_ok=True)
    for url in cddis_urls(date):
        fname = os.path.join(SP3_CACHE, os.path.basename(url))
        if not download_file(url, fname, auth):
            continue
        if fname.endswith(".gz"):
            return _decompress_gz(fname)
        elif fname.endswith(".Z"):
            out = fname[:-2]
            if os.system(f"uncompress -f '{fname}' 2>/dev/null") == 0 and os.path.exists(out):
                return out
        else:
            return fname
    return None


def fetch_nav(date: datetime, auth) -> str | None:
    """Download and decompress RINEX navigation file for date. Returns path or None."""
    os.makedirs(NAV_CACHE, exist_ok=True)
    for url in cddis_nav_urls(date):
        fname = os.path.join(NAV_CACHE, os.path.basename(url))
        if not download_file(url, fname, auth):
            continue
        if fname.endswith(".gz"):
            return _decompress_gz(fname)
        else:
            return fname
    return None


def _load_nav_dict(nav_path: str) -> dict:
    """
    Parse RINEX nav file into {sv: [(toe_sow, param_dict), ...]} for GPS SVs.

    Converts RINEX 2 semi-circle angles to radians. RINEX 3 values are already
    in radians. The returned dicts always have angles in radians and rates in rad/s.

    Args:
        nav_path: path to RINEX 2 or 3 navigation file
    Returns:
        Dict mapping GPS SV strings to sorted list of (sow_of_toc, param_dict).
    """
    try:
        import georinex as gr
        nav = gr.load(nav_path, use={"G"})   # GPS only; mixed-nav files may contain unsupported constellations
    except Exception:
        return {}

    needed = ["sqrtA", "Eccentricity", "Io", "Omega0", "omega", "M0",
              "DeltaN", "IDOT", "OmegaDot", "Crs", "Crc", "Cus", "Cuc",
              "Cis", "Cic", "Toe"]

    is_rinex2 = float(nav.attrs.get("version", 3.0)) < 3.0

    try:
        avail = [p for p in needed if p in nav]
        if len(avail) < len(needed):
            return {}
        df = nav[avail].to_dataframe().reset_index()
        df = df.dropna(subset=avail)
        df["sv"] = df["sv"].astype(str)
        df = df[df["sv"].str.startswith("G")]
        if df.empty:
            return {}
        df["sow"] = df["time"].apply(
            lambda t: int(
                (pd.Timestamp(t).to_pydatetime().replace(tzinfo=None) - _GPS_EPOCH
                 ).total_seconds()
            ) % 604800
        )
    except Exception:
        return {}

    result: dict = {}
    for sv, grp in df.groupby("sv"):
        entries = []
        for _, row in grp.iterrows():
            params = {p: float(row[p]) for p in needed}
            if is_rinex2:
                for f in _SC_FIELDS:
                    params[f] *= np.pi   # semi-circles → radians
            entries.append((int(row["sow"]), params))
        entries.sort(key=lambda x: x[0])
        result[str(sv)] = entries
    return result


def _gps_pos_m(params: dict, t_sow: float) -> np.ndarray | None:
    """
    Compute GPS satellite ECEF position (metres) from broadcast nav parameters.

    Implements IS-GPS-200 Table 20-IV. All input angles must be in radians.

    Args:
        params: nav message parameter dict (angles in radians, rates in rad/s)
        t_sow:  GPS seconds-of-week at which to evaluate position
    Returns:
        [X, Y, Z] array in metres, or None on any error.
    """
    try:
        sqrt_a    = float(params["sqrtA"])
        e         = float(params["Eccentricity"])
        i0        = float(params["Io"])
        Omega0    = float(params["Omega0"])
        omega     = float(params["omega"])
        M0        = float(params["M0"])
        dn        = float(params["DeltaN"])
        idot      = float(params["IDOT"])
        omega_dot = float(params["OmegaDot"])
        toe       = float(params["Toe"])
        Crs, Crc  = float(params["Crs"]),  float(params["Crc"])
        Cus, Cuc  = float(params["Cus"]),  float(params["Cuc"])
        Cis, Cic  = float(params["Cis"]),  float(params["Cic"])
    except (KeyError, TypeError, ValueError):
        return None

    a  = sqrt_a ** 2
    n  = np.sqrt(_GM / a**3) + dn

    tk = t_sow - toe
    if tk >  302400: tk -= 604800
    if tk < -302400: tk += 604800

    Mk = M0 + n * tk
    Ek = Mk
    for _ in range(5):
        Ek = Mk + e * np.sin(Ek)

    vk   = np.arctan2(np.sqrt(1 - e**2) * np.sin(Ek), np.cos(Ek) - e)
    phi  = vk + omega
    phi2 = 2 * phi

    u  = phi  + Cus * np.sin(phi2) + Cuc * np.cos(phi2)
    r  = a * (1 - e * np.cos(Ek)) + Crs * np.sin(phi2) + Crc * np.cos(phi2)
    ic = i0 + idot * tk + Cis * np.sin(phi2) + Cic * np.cos(phi2)
    Om = Omega0 + (omega_dot - _OMEGA_E) * tk - _OMEGA_E * toe

    xp, yp = r * np.cos(u), r * np.sin(u)
    cOm, sOm, ci = np.cos(Om), np.sin(Om), np.cos(ic)
    return np.array([
        xp * cOm - yp * ci * sOm,
        xp * sOm + yp * ci * cOm,
        yp * np.sin(ic),
    ])


def _eph_error_m(sv: str, t_sow: float, sp3_pos_km: np.ndarray,
                 nav_dict: dict) -> float:
    """
    Compute signed radial broadcast-minus-precise ephemeris error in metres.

    Positive = broadcast places satellite farther from Earth than SP3.

    Args:
        sv:         GPS SV string (e.g. 'G01')
        t_sow:      GPS seconds-of-week at observation epoch
        sp3_pos_km: precise SP3 position [X, Y, Z] in km (ECEF)
        nav_dict:   output of _load_nav_dict()
    Returns:
        Radial signed error in metres, clipped to [-5.0, 5.0].
    """
    entries = nav_dict.get(sv, [])
    if not entries:
        return 0.0

    best_params, best_dt = None, 1e18
    for sow, params in entries:
        dt = abs(t_sow - sow)
        if dt > 302400:
            dt = 604800 - dt
        if dt < best_dt and dt < 7200:
            best_dt, best_params = dt, params

    if best_params is None:
        return 0.0

    bcast_m = _gps_pos_m(best_params, t_sow)
    if bcast_m is None:
        return 0.0

    sp3_m  = sp3_pos_km * 1000.0
    r_norm = np.linalg.norm(sp3_m)
    if r_norm < 1e6:
        return 0.0
    radial = float(np.dot(bcast_m - sp3_m, sp3_m) / r_norm)
    return float(np.clip(radial, -5.0, 5.0))


def orbit_type(sv: str) -> str:
    """Return 'GEO' or 'MEO' for satellite PRN."""
    return "GEO" if sv in GEO_PRNS else "MEO"


def parse_sp3(path: str, nav_dict: dict | None = None) -> pd.DataFrame | None:
    """
    Parse SP3 precise orbit file into per-satellite error DataFrame.

    When nav_dict is provided, EphemerisError_m is computed as signed radial
    broadcast-minus-precise position difference (metres). Without nav_dict,
    falls back to orbital-radius deviation proxy (may clip heavily).

    Args:
        path:     path to decompressed SP3 file
        nav_dict: GPS broadcast nav data from _load_nav_dict(), or None
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

        clk_ns = clk * 1000.0
        clk_ns[np.abs(clk_ns) > 9e8] = np.nan   # SP3 missing flag

        # Precompute radius stats for fallback proxy (one pass, not per-epoch)
        if nav_dict is None or sv_str not in nav_dict:
            radii   = np.linalg.norm(pos, axis=1)
            med_rad = np.nanmedian(radii)

        for i, ts in enumerate(times):
            if np.isnan(clk_ns[i]) or np.any(np.isnan(pos[i])) or np.all(pos[i] == 0.0):
                continue

            if nav_dict is not None and sv_str.startswith("G"):
                dt = ts.to_pydatetime().replace(tzinfo=None)
                t_sow = int((dt - _GPS_EPOCH).total_seconds()) % 604800
                eph_m = _eph_error_m(sv_str, t_sow, pos[i], nav_dict)
            else:
                eph_m = float(np.clip(
                    (np.linalg.norm(pos[i]) - med_rad) * 1000.0, -5.0, 5.0
                ))

            rows.append({
                "Timestamp":        ts,
                "SatelliteID":      sv_str,
                "OrbitType":        orbit_type(sv_str),
                "ClockError_ns":    round(float(clk_ns[i]), 4),
                "EphemerisError_m": round(float(eph_m), 4),
            })

    return pd.DataFrame(rows) if rows else None


def fallback(out_path: str) -> pd.DataFrame:
    """Copy or generate synthetic data to out_path."""
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
    Download n_days of GNSS SP3 + RINEX nav data from NASA CDDIS.

    EphemerisError_m is the signed radial difference between broadcast and
    precise satellite positions in metres (typical GPS range: ±0.5 to ±3.0 m).
    Falls back to synthetic data on any download or parse failure.

    Args:
        n_days:   number of days of data to fetch (default 14)
        out_path: output CSV path
    Returns:
        DataFrame[Timestamp, SatelliteID, OrbitType, ClockError_ns, EphemerisError_m].
    """
    auth   = get_auth()
    end    = datetime(2024, 1, 14)
    dates  = [end - timedelta(days=i) for i in range(n_days - 1, -1, -1)]
    frames = []

    for date in dates:
        sp3_path = fetch_sp3(date, auth)
        if sp3_path is None:
            continue
        nav_path = fetch_nav(date, auth)
        nav_dict = _load_nav_dict(nav_path) if nav_path else None
        df = parse_sp3(sp3_path, nav_dict=nav_dict)
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
    eph_stats = df.groupby("SatelliteID")["EphemerisError_m"].agg(["mean", "std"])
    print(f"[fetch_data]   EphemerisError_m mean (first 5 SVs):")
    print(eph_stats.head().to_string())
