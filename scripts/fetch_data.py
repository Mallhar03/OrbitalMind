#!/usr/bin/env python3
"""
S1 INGEST — fetch real GNSS clock and ephemeris ERRORS from NASA CDDIS.

Both error columns are broadcast-minus-precise differences, which is what the
problem statement and the solution deck define as the prediction target:

    ClockError_ns    = [broadcast clock polynomial + relativistic correction]
                       - [precise SP3 clock]                      , in ns
    EphemerisError_m = radial component of
                       [broadcast Kepler position] - [precise SP3 position], in m

Two things this script deliberately does NOT do, because earlier versions did
and both corrupted the dataset (see memory/ppt_audit/S1-INGEST-*.md):

  * It never reports the raw SP3 precise clock as if it were a clock error.
    That value is the satellite's own clock bias, hundreds of microseconds
    large, against a deck target of 0.65 ns.
  * It never clips the ephemeris error to a fixed range. Clipping destroyed the
    true magnitude of 7% of rows and would teach a model a false ceiling.
    Outlier handling belongs in S2 preprocessing, not in ingestion.

Orbit type is classified from ORBITAL GEOMETRY, not from a PRN lookup table,
because the contest organisers' labelling convention is not known in advance.
Any satellite with a geosynchronous radius (~42164 km) is GEO, covering both
true equatorial GEO and inclined geosynchronous (IGSO/GSO) satellites, which
share the 24-hour periodicity the deck's GEO branch models.

The source product is the GFZ multi-GNSS rapid combination (GFZ0MGXRAP), which
carries GPS, BeiDou, Galileo, GLONASS and QZSS. The IGS rapid combination used
previously is GPS-only, and GPS is entirely MEO, so it can never satisfy the
deck's GEO/GSO requirement.

Usage:
    python scripts/fetch_data.py --days 8 --strict

Output:
    data/raw/gnss_real.csv        first N-1 days   — the model's input
    data/raw/gnss_holdout.csv     final day only   — held-out truth, DO NOT TRAIN

The holdout day is written to a separate file on purpose. It must never be
concatenated into the training input, fitted on, or inspected while making
modelling choices. It exists to score a forecast that was already produced.
"""
import argparse
import gzip
import json
import netrc
import os
import shutil
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

REQUIRED_COLS = ["Timestamp", "SatelliteID", "OrbitType",
                 "ClockError_ns", "EphemerisError_m"]
RAW_OUTPUT     = "data/raw/gnss_real.csv"
HOLDOUT_OUTPUT = "data/raw/gnss_holdout.csv"
SYNTHETIC_SRC  = "data/synthetic/gnss_synthetic.csv"
SP3_CACHE      = "data/raw/sp3"
NAV_CACHE      = "data/raw/nav"

_GPS_EPOCH = datetime(1980, 1, 6)
_OMEGA_E   = 7.2921151467e-5     # rad/s   Earth rotation rate

# Gravitational constant differs slightly between ICDs.
_GM_BY_SYS = {"G": 3.986005e14, "J": 3.986005e14,
              "E": 3.986004418e14, "C": 3.986004418e14}

# BeiDou System Time runs 14 s behind GPS Time.
_BDS_GPS_OFFSET_S = 14.0

# GLONASS uses a wholly different (non-Keplerian) broadcast model, so it is
# excluded rather than approximated with the wrong algorithm.
CONSTELLATIONS = ("G", "C", "E", "J")

# A geosynchronous orbit sits near 42164 km. Everything else here is MEO.
GEO_RADIUS_KM_MIN = 40000.0

SP3_BAD_CLOCK_US = 999999.0
TARGET_STEP_MIN  = 15


def get_auth():
    """Read NASA Earthdata credentials from ~/.netrc. Returns (user, pw) or None."""
    try:
        creds = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        if creds:
            return (creds[0], creds[2])
    except Exception:
        pass
    return None


def gps_week_dow(date: datetime):
    """Return (GPS week, day-of-week) for date."""
    delta = (date - _GPS_EPOCH).days
    return delta // 7, delta % 7


def sp3_urls(date: datetime) -> list:
    """
    Candidate SP3 URLs for date, multi-GNSS first.

    GFZ0MGXRAP is the multi-GNSS rapid combination and is the only one of these
    that contains GEO/GSO satellites. The IGS entries are GPS-only fallbacks
    kept so a pull still yields something if MGEX is unavailable, but a pull
    that lands on them cannot satisfy the deck's orbit-type claim.
    """
    week, dow = gps_week_dow(date)
    year, doy = date.year, date.timetuple().tm_yday
    return [
        (f"https://cddis.nasa.gov/archive/gnss/products/{week}/"
         f"GFZ0MGXRAP_{year}{doy:03d}0000_01D_05M_ORB.SP3.gz"),
        (f"https://cddis.nasa.gov/archive/gnss/products/{week}/"
         f"IGS0OPSRAP_{year}{doy:03d}0000_01D_15M_ORB.SP3.gz"),
    ]


def nav_urls(date: datetime) -> list:
    """Candidate RINEX broadcast navigation URLs for date (mixed constellation)."""
    year, doy = date.year, date.timetuple().tm_yday
    return [
        (f"https://cddis.nasa.gov/archive/gnss/data/daily/{year}/brdc/"
         f"BRDC00IGS_R_{year}{doy:03d}0000_01D_MN.rnx.gz"),
    ]


def download_file(url: str, dest: str, auth) -> bool:
    """Download url to dest through EarthData auth redirects. True on success."""
    try:
        with requests.Session() as session:
            if auth:
                session.auth = auth
            r = session.get(url, timeout=120, stream=True, allow_redirects=True)
            if r.status_code != 200:
                return False
            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        return True
    except Exception:
        return False


def _decompress(fname: str) -> str | None:
    """Decompress a .gz file in place. Returns decompressed path or None."""
    if fname.endswith(".gz"):
        out = fname[:-3]
        try:
            with gzip.open(fname, "rb") as fi, open(out, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            os.remove(fname)
            return out
        except Exception:
            return None
    return fname


def _fetch(urls: list, cache_dir: str, auth) -> str | None:
    """Try each URL in order, returning the first decompressed local path."""
    os.makedirs(cache_dir, exist_ok=True)
    for url in urls:
        base  = os.path.basename(url)
        plain = base[:-3] if base.endswith(".gz") else base
        cached = os.path.join(cache_dir, plain)
        if os.path.exists(cached) and os.path.getsize(cached) > 0:
            return cached
        fname = os.path.join(cache_dir, base)
        if not download_file(url, fname, auth):
            continue
        out = _decompress(fname)
        if out:
            return out
    return None


def latest_available_date(auth, max_lookback: int = 14) -> datetime | None:
    """
    Find the newest date CDDIS has a multi-GNSS SP3 product for.

    Precise products lag real time by days, so the newest usable date is
    discovered rather than assumed.
    """
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    for back in range(1, max_lookback + 1):
        date = today - timedelta(days=back)
        url  = sp3_urls(date)[0]
        try:
            with requests.Session() as session:
                if auth:
                    session.auth = auth
                r = session.get(url, timeout=30, stream=True, allow_redirects=True)
                ok = r.status_code == 200
                r.close()
            if ok:
                return date
        except Exception:
            continue
    return None


def parse_sp3(path: str) -> dict:
    """
    Parse an SP3-c/d file into per-satellite precise position and clock series.

    Args:
        path: path to a decompressed SP3 file
    Returns:
        {sv: {"times": [datetime], "pos": (n,3) ndarray km, "clk": (n,) ndarray us}}
        Epochs whose clock carries the SP3 bad-value flag are dropped.
    """
    times: list = []
    data: dict = {}
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("*"):
                p = line.split()
                try:
                    times.append(datetime(int(p[1]), int(p[2]), int(p[3]),
                                          int(p[4]), int(p[5]), int(float(p[6]))))
                except (ValueError, IndexError):
                    pass
            elif line.startswith("P") and times:
                sv = line[1:4].strip()
                if not sv or sv[0] not in CONSTELLATIONS:
                    continue
                try:
                    x   = float(line[4:18])
                    y   = float(line[18:32])
                    z   = float(line[32:46])
                    clk = float(line[46:60])
                except ValueError:
                    continue
                if x == 0.0 and y == 0.0 and z == 0.0:
                    continue
                if abs(clk) >= SP3_BAD_CLOCK_US:
                    continue
                d = data.setdefault(sv, {"times": [], "pos": [], "clk": []})
                d["times"].append(times[-1])
                d["pos"].append((x, y, z))
                d["clk"].append(clk)

    return {
        sv: {"times": d["times"],
             "pos": np.asarray(d["pos"], dtype=float),
             "clk": np.asarray(d["clk"], dtype=float)}
        for sv, d in data.items() if len(d["times"]) > 0
    }


def classify_orbit(pos_km: np.ndarray) -> str:
    """
    Classify a satellite as GEO or MEO from its own orbital geometry.

    A PRN lookup table is deliberately not used: the contest organisers'
    labelling convention is unknown, and geometry is self-evident from the data.
    Geosynchronous satellites sit near 42164 km, whether equatorial (GEO) or
    inclined (IGSO/GSO); both share the 24-hour periodicity the deck's GEO
    branch is built around, so both are labelled GEO.

    Args:
        pos_km: (n,3) ECEF positions in kilometres
    Returns:
        "GEO" or "MEO".
    """
    radius = float(np.median(np.linalg.norm(pos_km, axis=1)))
    return "GEO" if radius > GEO_RADIUS_KM_MIN else "MEO"


def load_nav(nav_path: str) -> dict:
    """
    Parse a mixed RINEX 3 navigation file into per-satellite broadcast records.

    Args:
        nav_path: path to a decompressed RINEX navigation file
    Returns:
        {sv: [(toc_datetime, params_dict), ...]} sorted by epoch, where params
        carries the clock polynomial (af0/af1/af2) and the Keplerian elements.
        Angles are radians, as RINEX 3 stores them.
    """
    try:
        import georinex as gr
        nav = gr.load(nav_path, use=set(CONSTELLATIONS))
    except Exception:
        return {}

    clock_fields = ["SVclockBias", "SVclockDrift", "SVclockDriftRate"]
    orbit_fields = ["sqrtA", "Eccentricity", "Io", "Omega0", "omega", "M0",
                    "DeltaN", "IDOT", "OmegaDot", "Crs", "Crc",
                    "Cus", "Cuc", "Cis", "Cic", "Toe"]

    try:
        avail = [f for f in clock_fields + orbit_fields if f in nav]
        if not all(f in avail for f in clock_fields):
            return {}
        df = nav[avail].to_dataframe().reset_index()
        df = df.dropna(subset=clock_fields)
        df["sv"] = df["sv"].astype(str)
        df = df[df["sv"].str[0].isin(CONSTELLATIONS)]
        if df.empty:
            return {}
    except Exception:
        return {}

    out: dict = {}
    for sv, grp in df.groupby("sv"):
        entries = []
        for _, row in grp.iterrows():
            params = {}
            for f in avail:
                try:
                    params[f] = float(row[f])
                except (TypeError, ValueError):
                    params[f] = np.nan
            toc = pd.Timestamp(row["time"]).to_pydatetime().replace(tzinfo=None)
            entries.append((toc, params))
        out[sv] = sorted(entries, key=lambda t: t[0])
    return out


def _select_record(entries: list, epoch: datetime, max_age_s: float = 7200.0):
    """Return the broadcast record whose epoch is nearest to `epoch`, or None."""
    best, best_dt = None, max_age_s
    for toc, params in entries:
        dt = abs((epoch - toc).total_seconds())
        if dt <= best_dt:
            best_dt, best = dt, (toc, params)
    return best


def _kepler_ek(params: dict, tk: float, gm: float):
    """Solve Kepler's equation for eccentric anomaly at tk seconds from toe."""
    try:
        sqrt_a = params["sqrtA"]
        e      = params["Eccentricity"]
        M0     = params["M0"]
        dn     = params["DeltaN"]
    except KeyError:
        return None
    if not np.isfinite([sqrt_a, e, M0, dn]).all() or sqrt_a <= 0:
        return None
    a  = sqrt_a ** 2
    n  = np.sqrt(gm / a ** 3) + dn
    Mk = M0 + n * tk
    Ek = Mk
    for _ in range(10):
        Ek = Mk + e * np.sin(Ek)
    return float(Ek)


def _time_ref(epoch: datetime, sys_char: str) -> datetime:
    """Convert a GPS-time epoch into the constellation's own time scale."""
    if sys_char == "C":
        return epoch - timedelta(seconds=_BDS_GPS_OFFSET_S)
    return epoch


def broadcast_clock_s(params: dict, toc: datetime, epoch: datetime,
                      sys_char: str):
    """
    Evaluate the broadcast satellite clock correction at `epoch`, in seconds.

    Implements af0 + af1*dt + af2*dt^2 only.

    The relativistic eccentricity correction F*e*sqrt(A)*sin(Ek) is deliberately
    NOT applied. Both the broadcast polynomial and the IGS precise clock are
    referenced such that the user adds that term themselves, so it is common to
    both and cancels in a broadcast-minus-precise difference. Adding it injects a
    spurious eccentricity-dependent signal: measured on 2026-08-28 it inflated
    Galileo's error spread from 0.54 ns to 70 ns and E14's from 0.44 ns to a
    +/-400 ns swing, exactly the amplitude of F*e*sqrt(A) for that satellite.

    Args:
        params:   broadcast record parameters
        toc:      time of clock for that record
        epoch:    observation epoch, in GPS time
        sys_char: constellation letter, used for the BDS time offset and GM
    Returns:
        Clock correction in seconds, or None if the record is unusable.
    """
    try:
        af0 = params["SVclockBias"]
        af1 = params["SVclockDrift"]
        af2 = params["SVclockDriftRate"]
    except KeyError:
        return None
    if not np.isfinite([af0, af1, af2]).all():
        return None

    ref = _time_ref(epoch, sys_char)
    dt  = (ref - toc).total_seconds()
    if dt >  302400: dt -= 604800
    if dt < -302400: dt += 604800

    return float(af0 + af1 * dt + af2 * dt * dt)


def broadcast_position_m(params: dict, epoch: datetime, sys_char: str):
    """
    Compute broadcast ECEF position in metres from Keplerian elements.

    Implements the standard ICD algorithm shared by GPS, Galileo, QZSS and
    non-geostationary BeiDou satellites.

    Args:
        params:   broadcast record parameters
        epoch:    observation epoch, in GPS time
        sys_char: constellation letter
    Returns:
        [X, Y, Z] in metres, or None if the record is unusable.
    """
    needed = ["sqrtA", "Eccentricity", "Io", "Omega0", "omega", "M0", "DeltaN",
              "IDOT", "OmegaDot", "Crs", "Crc", "Cus", "Cuc", "Cis", "Cic", "Toe"]
    if any(p not in params or not np.isfinite(params[p]) for p in needed):
        return None

    gm  = _GM_BY_SYS.get(sys_char, 3.986005e14)
    ref = _time_ref(epoch, sys_char)

    sow = ((ref - _GPS_EPOCH).total_seconds()) % 604800.0
    tk  = sow - params["Toe"]
    if tk >  302400: tk -= 604800
    if tk < -302400: tk += 604800

    Ek = _kepler_ek(params, tk, gm)
    if Ek is None:
        return None

    e      = params["Eccentricity"]
    sqrt_a = params["sqrtA"]
    a      = sqrt_a ** 2

    vk   = np.arctan2(np.sqrt(1 - e ** 2) * np.sin(Ek), np.cos(Ek) - e)
    phi  = vk + params["omega"]
    phi2 = 2 * phi

    u  = phi + params["Cus"] * np.sin(phi2) + params["Cuc"] * np.cos(phi2)
    r  = a * (1 - e * np.cos(Ek)) + params["Crs"] * np.sin(phi2) + params["Crc"] * np.cos(phi2)
    ic = params["Io"] + params["IDOT"] * tk + params["Cis"] * np.sin(phi2) + params["Cic"] * np.cos(phi2)
    Om = params["Omega0"] + (params["OmegaDot"] - _OMEGA_E) * tk - _OMEGA_E * params["Toe"]

    xp, yp = r * np.cos(u), r * np.sin(u)
    cOm, sOm, ci = np.cos(Om), np.sin(Om), np.cos(ic)
    return np.array([
        xp * cOm - yp * ci * sOm,
        xp * sOm + yp * ci * cOm,
        yp * np.sin(ic),
    ])


def build_day(sp3_path: str, nav_path):
    """
    Build one day of broadcast-minus-precise errors from an SP3/nav file pair.

    Args:
        sp3_path: decompressed SP3 file
        nav_path: decompressed RINEX navigation file, or None
    Returns:
        DataFrame with REQUIRED_COLS, subsampled to a 15-minute grid, or None.
        Satellites with no usable broadcast record are omitted entirely rather
        than being emitted with a placeholder error of zero.
    """
    sp3 = parse_sp3(sp3_path)
    if not sp3:
        return None
    nav = load_nav(nav_path) if nav_path else {}
    if not nav:
        return None

    rows = []
    for sv, d in sp3.items():
        entries = nav.get(sv)
        if not entries:
            continue
        sys_char = sv[0]
        orbit    = classify_orbit(d["pos"])
        for i, epoch in enumerate(d["times"]):
            if epoch.minute % TARGET_STEP_MIN != 0 or epoch.second != 0:
                continue
            rec = _select_record(entries, _time_ref(epoch, sys_char))
            if rec is None:
                continue
            toc, params = rec

            bclk = broadcast_clock_s(params, toc, epoch, sys_char)
            if bclk is None:
                continue
            clock_err_ns = (bclk - d["clk"][i] * 1e-6) * 1e9

            bpos = broadcast_position_m(params, epoch, sys_char)
            if bpos is None:
                continue
            sp3_m  = d["pos"][i] * 1000.0
            r_norm = np.linalg.norm(sp3_m)
            if r_norm < 1e6:
                continue
            eph_err_m = float(np.dot(bpos - sp3_m, sp3_m) / r_norm)

            rows.append({
                "Timestamp":        epoch,
                "SatelliteID":      sv,
                "OrbitType":        orbit,
                "ClockError_ns":    round(clock_err_ns, 4),
                "EphemerisError_m": round(eph_err_m, 4),
            })

    return pd.DataFrame(rows) if rows else None


def write_provenance(out_path: str, sources: list, input_df, holdout_df,
                     holdout_path: str, synthetic: bool = False) -> str:
    """
    Record where a dataset came from, alongside the CSV.

    Provenance lives in a sidecar JSON rather than an extra column because the
    solution deck fixes the CSV schema at five columns (C-33), and a dataset
    that cannot say whether it is real or synthetic is exactly how synthetic
    results get reported as real.

    Args:
        out_path:     path of the input CSV this describes
        sources:      list of {date, sp3, nav} dicts, one per day fetched
        input_df:     the input DataFrame
        holdout_df:   the held-out DataFrame, or None
        holdout_path: path of the holdout CSV
        synthetic:    True when the data is synthetic rather than real
    Returns:
        Path of the provenance file written.
    """
    path = os.path.splitext(out_path)[0] + ".provenance.json"

    def _span(df):
        if df is None or len(df) == 0:
            return None
        ts = pd.to_datetime(df["Timestamp"], format="mixed")
        return {"first": str(ts.min()), "last": str(ts.max()),
                "days": int(ts.dt.date.nunique()), "rows": int(len(df)),
                "satellites": int(df["SatelliteID"].nunique()),
                "orbit_types": {k: int(v) for k, v in
                                df["OrbitType"].value_counts().items()}}

    record = {
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "origin": "synthetic" if synthetic else "real",
        "archive": None if synthetic else "NASA CDDIS (https://cddis.nasa.gov)",
        "authentication": None if synthetic else "urs.earthdata.nasa.gov via ~/.netrc",
        "products": sources,
        "input_file": out_path,
        "input": _span(input_df),
        "holdout_file": holdout_path,
        "holdout": _span(holdout_df),
        "holdout_warning": (
            "The holdout day is held-out truth. It must never be trained on, "
            "fitted on, or inspected while making modelling choices."
        ),
        "error_definition": {
            "ClockError_ns": "broadcast clock polynomial (af0+af1*dt+af2*dt^2) "
                             "minus precise SP3 clock, in nanoseconds",
            "EphemerisError_m": "radial component of broadcast Kepler position "
                                "minus precise SP3 position, in metres",
            "relativistic_term": "not applied; it cancels between broadcast and "
                                 "IGS precise clocks",
            "clipping": "none; outlier handling belongs to preprocessing",
        },
        "orbit_type_rule": (
            f"median orbital radius > {GEO_RADIUS_KM_MIN:.0f} km is GEO "
            "(covers equatorial GEO and inclined IGSO/GSO), else MEO"
        ),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return path


def fallback(out_path: str, strict: bool = False):
    """
    Copy or generate synthetic data to out_path.

    Args:
        out_path: destination CSV path
        strict:   when True, raise instead of substituting synthetic data
    Returns:
        DataFrame of synthetic data.
    Raises:
        RuntimeError: if strict is True. A silent synthetic substitution is
            indistinguishable from a real download downstream, which is how a
            pipeline ends up reporting synthetic results as real.
    """
    if strict:
        raise RuntimeError(
            "Real GNSS data could not be downloaded from CDDIS and strict mode "
            "is on, so no synthetic fallback was written. Check ~/.netrc "
            "credentials for urs.earthdata.nasa.gov and CDDIS availability."
        )
    print("[fetch_data] Real data unavailable — using synthetic fallback")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(SYNTHETIC_SRC):
        df = pd.read_csv(SYNTHETIC_SRC)
        df.to_csv(out_path, index=False)
    else:
        sys.path.insert(0, "src")
        from orbitalmind.utils.synthetic_generator import generate_synthetic_gnss_data
        df = generate_synthetic_gnss_data(save_path=out_path)
    # Mark it, so a synthetic substitution can never be mistaken for real data.
    write_provenance(out_path, [], df, None, "", synthetic=True)
    return df


def fetch_gnss_data(n_days: int = 8,
                    out_path: str = RAW_OUTPUT,
                    holdout_path: str = HOLDOUT_OUTPUT,
                    end=None,
                    strict: bool = True):
    """
    Download n_days of real GNSS errors, splitting off the final day as truth.

    The first n_days-1 days are written to out_path and are the pipeline's
    input. The final day is written to holdout_path and must never be trained
    on, fitted on, or inspected while making modelling choices.

    Args:
        n_days:       total days to fetch, including the holdout day
        out_path:     CSV for the input days
        holdout_path: CSV for the final held-out day
        end:          last day to fetch. Default: newest available on CDDIS.
        strict:       fail loudly rather than substituting synthetic data.
                      Defaults to True: a silent substitution writes synthetic
                      rows to the same path with the same columns a real pull
                      would use, which is indistinguishable downstream. Opting
                      out has to be deliberate.
    Returns:
        (input_df, holdout_df).
    """
    auth = get_auth()
    if auth is None:
        print("[fetch_data] WARNING: no urs.earthdata.nasa.gov entry in ~/.netrc")

    if end is None:
        end = latest_available_date(auth)
        if end is None:
            return fallback(out_path, strict=strict), None
        print(f"[fetch_data] Newest multi-GNSS product: {end:%Y-%m-%d}")

    dates  = [end - timedelta(days=i) for i in range(n_days - 1, -1, -1)]
    frames, sources = [], []
    for date in dates:
        sp3_path = _fetch(sp3_urls(date), SP3_CACHE, auth)
        if sp3_path is None:
            print(f"[fetch_data] {date:%Y-%m-%d}: no SP3 available")
            continue
        nav_path = _fetch(nav_urls(date), NAV_CACHE, auth)
        df = build_day(sp3_path, nav_path)
        if df is not None and not df.empty:
            frames.append(df)
            sources.append({
                "date": f"{date:%Y-%m-%d}",
                "sp3": os.path.basename(sp3_path),
                "nav": os.path.basename(nav_path) if nav_path else None,
            })
            print(f"[fetch_data] {date:%Y-%m-%d}: {len(df):5d} rows  "
                  f"({os.path.basename(sp3_path)})")
        else:
            print(f"[fetch_data] {date:%Y-%m-%d}: parsed no usable rows")

    if not frames:
        return fallback(out_path, strict=strict), None

    df = (pd.concat(frames, ignore_index=True)
            .sort_values(["SatelliteID", "Timestamp"])
            .reset_index(drop=True)
            .dropna())
    if df.empty:
        return fallback(out_path, strict=strict), None

    # SP3 daily files carry the 00:00 epoch of the following day. That spillover
    # would otherwise become a 94-row "extra day" and be mistaken for the holdout.
    ts = pd.to_datetime(df["Timestamp"])
    df = df[ts.dt.date <= end.date()].reset_index(drop=True)

    # The holdout is the LAST REQUESTED day, so the split is deterministic rather
    # than dependent on whatever date happens to appear last in the data.
    # SP3 daily files also repeat the midnight epoch at BOTH ends, so every
    # boundary between two requested days yields two rows for the same
    # (Timestamp, SatelliteID) with slightly different values — they come from
    # different daily products and different broadcast records. Keep the copy
    # from the day that OWNS the epoch: frames were appended oldest-first, and
    # the sort above is stable, so the later duplicate is day N's own file
    # rather than day N-1's trailing 24:00 endpoint.
    df = (df.drop_duplicates(subset=["Timestamp", "SatelliteID"], keep="last")
            .reset_index(drop=True))

    ts          = pd.to_datetime(df["Timestamp"])
    holdout_day = end.date()
    is_holdout  = ts.dt.date == holdout_day
    input_df    = df[~is_holdout].reset_index(drop=True)
    holdout_df  = df[is_holdout].reset_index(drop=True)

    for path, frame in ((out_path, input_df), (holdout_path, holdout_df)):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        frame.to_csv(path, index=False)

    prov = write_provenance(out_path, sources, input_df, holdout_df, holdout_path)

    print(f"[fetch_data] Input   -> {out_path}      {len(input_df):6d} rows")
    print(f"[fetch_data] Origin  -> {prov}")
    print(f"[fetch_data] Holdout -> {holdout_path}  {len(holdout_df):6d} rows "
          f"({holdout_day}) — DO NOT TRAIN ON THIS")
    return input_df, holdout_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Fetch real GNSS clock/ephemeris errors from NASA CDDIS.")
    ap.add_argument("--days", type=int, default=8,
                    help="total days including the held-out final day")
    ap.add_argument("--end", type=str, default=None,
                    help="last day, YYYY-MM-DD. Default: newest available.")
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="fall back to synthetic data if the real pull fails. "
                         "OFF by default: a silent substitution is "
                         "indistinguishable from real data downstream.")
    ap.add_argument("--strict", action="store_true",
                    help="deprecated; strict failure is now the default")
    ap.add_argument("--out", type=str, default=RAW_OUTPUT)
    ap.add_argument("--holdout", type=str, default=HOLDOUT_OUTPUT)
    args = ap.parse_args()

    end_dt = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None
    df, holdout = fetch_gnss_data(n_days=args.days, out_path=args.out,
                                  holdout_path=args.holdout,
                                  end=end_dt, strict=not args.allow_synthetic)

    assert list(df.columns) == REQUIRED_COLS, f"Column mismatch: {list(df.columns)}"
    assert df.isnull().sum().sum() == 0, "NaN values present in output"
    assert set(df["OrbitType"].unique()) <= {"GEO", "MEO"}, "Invalid OrbitType"

    print(f"\n[fetch_data] OK {len(df)} input rows | columns OK | zero NaN")
    print(f"[fetch_data]   Satellites : {df['SatelliteID'].nunique()}")
    print(f"[fetch_data]   OrbitTypes : {df['OrbitType'].value_counts().to_dict()}")
    print(f"[fetch_data]   GEO sats   : "
          f"{sorted(df[df.OrbitType=='GEO']['SatelliteID'].unique())}")
    for col in ("ClockError_ns", "EphemerisError_m"):
        s = df[col]
        print(f"[fetch_data]   {col:17s} mean {s.mean():10.3f}  std {s.std():10.3f}"
              f"  min {s.min():10.3f}  max {s.max():10.3f}")
