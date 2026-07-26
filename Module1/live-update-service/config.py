"""
config.py — FROZEN Module 1 constants.

Every rule and coefficient here was lifted verbatim from the production
scripts in ../development-history/ and VERIFIED to reproduce the official
dataset to within 4e-8 MCM across all 4,800 rows. Nothing in this file is
re-derived at runtime — the daily updater must apply the SAME detection
rules and the SAME volume coefficients that produced the validated
2021-2025 table, never re-fit them. Change nothing here without
re-running scripts/verify_frozen_logic.py.

Source of each rule, for audit:
  S1 detection (orbit, VV<-16, focal_mean 30m):
    ../development-history/GEE/1_SMART_SENSOR_6DAY_TIME_SERIES_S1+S2_raw_export.txt
  S2 detection + tile-mosaic / valid_fraction fix:
    ../development-history/GEE/2_FIX_S2_Tile_Mosaicking_produces_S2_LargeTanks_FIXED.txt
  Sensor-priority consolidation (S2 within +/-3 days else S1 fallback):
    ../development-history/Storage Calculation/3_Rebuild_Consolidated_Dataset_using_FIXED_S2.py
  Volume coefficients (K_NACH, K_TEL, N_TEL, THETA, D_AVG):
    ../development-history/Storage Calculation/2_Volume_Calibration_Nachchaduwa_and_TelivaraiKulam.py
  Official CSV these were verified against:
    ../development-history/Exported CSVs/5_Module1_FINAL_CORRECTED_2021_2025_OFFICIAL_DATASET.csv
"""
import os

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
TABLE_CSV    = os.path.join(DATA_DIR, "tank_storage.csv")   # the shared table (committed to git)
TANK_META    = os.path.join(DATA_DIR, "tank_meta.csv")      # tank_id, pond_name, total_ha (fixed)

# ── GEE ───────────────────────────────────────────────────────────
GEE_PROJECT = "gee-water-monitoring"
TANK_ASSET  = "projects/gee-water-monitoring/assets/nachchaduwa-32-final"
NAME_COL    = "pond_name"

# ── Detection rules (from GEE master + S2 tile-mosaic fix) ─────────
S1_COLLECTION   = "COPERNICUS/S1_GRD"
S1_ORBIT        = "DESCENDING"          # descending only — matches the official CSV
S1_VV_THRESHOLD = -16                   # dB; water = VV < -16
SPECKLE_RADIUS  = 30                    # focal_mean(30, 'circle', 'meters')

S2_COLLECTION   = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_MAX    = 20                    # CLOUDY_PIXEL_PERCENTAGE < 20
S2_MNDWI_BANDS  = ["B3", "B11"]         # MNDWI = (B3 - B11)/(B3 + B11), water = MNDWI > 0
S2_VALID_FRAC   = 0.5                   # keep S2 obs only if >50% of expected pixels valid
S2_WINDOW_DAYS  = 3                     # large tanks: prefer S2 within +/- 3 days of the S1 date

SCALE_M  = 10
MAXPIXELS = int(1e8)

# The 7 large tanks (>50 ha) — S2 preferred, S1 fallback. All others: S1 only.
LARGE_TANKS = [
    "Alittana Wewa", "Mahakanumulla Wewa", "Nachchaduwa Wewa",
    "Periyakulama Wewa", "Tirappane Wewa", "Todamaduwa Wewa",
    "Wannanmaduwa Wewa",
]

# ── Frozen volume coefficients (verified against official CSV) ─────
N_LIEBE = 1.43
K_NACH  = 0.001757252759          # Tier 1: Nachchaduwa, gauge-calibrated  V = K_NACH * A^1.43
K_TEL   = 0.007313524564          # Tier 2: Telivarai Kulam power law
N_TEL   = 1.246083422777
THETA   = 5406.2253441350         # Tier 3: Imbulana linear-storage coefficient
D_AVG   = 1.93

CONF_NACH   = "Validated (R\u00b2=0.618 vs gauge)"
CONF_OTHERS = "Sri Lanka small-tank power law (Telivarai Kulam), Imbulana ceiling-capped"

# ── Output schema (order matters — matches the official CSV) ───────
BASE_COLUMNS = [
    "tank_id", "pond_name", "date", "year", "month",
    "water_ha", "total_ha", "fill_pct", "sensor_used",
    "s2_date", "day_gap", "volume_mcm", "volume_confidence",
]
# additive provenance columns, appended at the end (consumers may ignore them)
EXTRA_COLUMNS = ["record_status", "last_updated"]
ALL_COLUMNS   = BASE_COLUMNS + EXTRA_COLUMNS


# ── Frozen volume function (copied verbatim from calibration script) ──
def _telivarai_volume(area_ha):
    return K_TEL * (max(area_ha, 0.0) ** N_TEL)

def _imbulana_ceiling(area_ha):
    return (THETA * area_ha * D_AVG) / 1e6

def final_volume(pond_name, water_ha, total_ha):
    """Stratified volume in MCM. Identical math to 2_Volume_Calibration...py."""
    if pond_name == "Nachchaduwa Wewa":
        return K_NACH * (water_ha ** N_LIEBE)
    tv = _telivarai_volume(water_ha)
    ceiling = _imbulana_ceiling(total_ha)
    fill_frac = water_ha / total_ha if total_ha > 0 else 0.0
    return min(tv, ceiling * fill_frac * 1.5)

def volume_confidence(pond_name):
    return CONF_NACH if pond_name == "Nachchaduwa Wewa" else CONF_OTHERS
