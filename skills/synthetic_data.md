# Skill: Synthetic Data Generator
# Iteration 1

---

## GOAL
Generate a realistic synthetic GNSS satellite error dataset that mimics
real satellite behaviour. This data is used for ALL iterations until
the real dataset arrives on hackathon day.

---

## OUTPUT FILE
data/synthetic/gnss_synthetic.csv

## OUTPUT COLUMNS — EXACT NAMES, NO VARIATION
Timestamp         : datetime, 15-minute intervals, 8 days total
SatelliteID       : string, e.g. "GEO-01", "MEO-03"
OrbitType         : string, exactly "GEO" or "MEO" — no other values
ClockError_ns     : float, satellite clock error in nanoseconds
EphemerisError_m  : float, satellite orbital error in metres

---

## EXACT SPECIFICATIONS

### Time range
- Start: 2024-01-01 00:00:00
- End:   2024-01-08 23:45:00
- Frequency: 15 minutes
- Points per satellite: 768 total (8 days × 96 points/day)
- Training points: 672 (days 1-7)
- Test points: 96 (day 8 — prediction target)

### Satellites to generate
GEO satellites: 3 (GEO-01, GEO-02, GEO-03)
MEO satellites: 5 (MEO-01, MEO-02, MEO-03, MEO-04, MEO-05)
Total: 8 satellites × 768 points = 6144 rows

### GEO satellite signal formula
ClockError_ns =
    drift_rate * t                              # slow linear drift
    + A_24 * sin(2π*t/1440 + phase_24)        # 24-hour cycle (1440 min)
    + 0.3 * N(0,1)                             # gaussian noise

Where:
    t           = time in minutes from start
    drift_rate  = random.uniform(0.0005, 0.002) per satellite
    A_24        = random.uniform(1.5, 3.0) per satellite
    phase_24    = random.uniform(0, 2π) per satellite

EphemerisError_m =
    drift_rate * 0.5 * t
    + 0.8 * sin(2π*t/1440 + phase_24 + 0.5)
    + 0.1 * N(0,1)

### MEO satellite signal formula
ClockError_ns =
    drift_rate * t
    + A_12 * sin(2π*t/720 + phase_12)         # 12-hour cycle (720 min)
    + A_24 * sin(2π*t/1440 + phase_24)        # 24-hour cycle
    + 0.25 * N(0,1)

Where:
    drift_rate  = random.uniform(0.0003, 0.001) per satellite
    A_12        = random.uniform(1.0, 2.5) per satellite
    A_24        = random.uniform(0.3, 0.8) per satellite
    phase_12    = random.uniform(0, 2π) per satellite
    phase_24    = random.uniform(0, 2π) per satellite

EphemerisError_m =
    drift_rate * 0.4 * t
    + 0.6 * sin(2π*t/720 + phase_12 + 0.3)
    + 0.08 * N(0,1)

### IOD jumps (must be present for preprocessing to have something to fix)
- Every 8 steps (= 2 hours), with 30% probability, add a jump
- Jump magnitude: random.uniform(-0.5, 0.5) ns added to all subsequent values
- This simulates real satellite upload discontinuities

### Random seed
ALL random operations use numpy seed 42. No exceptions.

---

## FUNCTION SIGNATURE

File: src/orbitalmind/utils/synthetic_generator.py

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

---

## CONSTRAINTS
- seed=42 everywhere — any numpy random call uses np.random.seed(42) at top
- No external data downloads — fully self-contained generation
- Save CSV automatically when function is called
- Print confirmation: "Generated [N] rows for [G] GEO and [M] MEO satellites"

---

## VERIFY FOR THIS ITERATION
Run: pytest tests/test_synthetic_data.py -v
ALL of these must pass:
- Row count == 6144
- Columns match exactly: Timestamp, SatelliteID, OrbitType, ClockError_ns, EphemerisError_m
- OrbitType values are only "GEO" or "MEO"
- GEO satellite count == 3, MEO satellite count == 5
- No NaN values anywhere
- ClockError_ns range is between -20 and +20 ns (realistic bounds)
- EphemerisError_m range is between -5 and +5 m (realistic bounds)
- CSV file exists at data/synthetic/gnss_synthetic.csv

---

## FAILURE MODES
If row count is wrong: check n_days × 96 × n_satellites calculation
If NaN present: IOD jump logic is corrupting values — check cumsum operations
If range exceeded: drift_rate is too high — cap at 0.002
