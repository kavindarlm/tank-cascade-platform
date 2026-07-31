# ============================================================
# storage_generate.py — FINAL VERSION (alpha_water_transfer_coefficient)
# Integrates real Module 1 output (satellite-derived fill_pct)
# Uses PCHIP interpolation (shape-preserving, no overshoot)
# instead of plain linear interpolation.
#
# UPDATED: now also exports tank_peak_volume_mcm to
# tank_peak_volumes.json — this is the raw MCM peak per tank,
# previously computed internally and discarded. inference.py
# needs it to convert live volume_mcm readings from the DB into
# the same 0-100% scale the model was trained on.
# ============================================================

import pandas as pd
import numpy as np
import json
from scipy.interpolate import PchipInterpolator

# -------------------------------
# CELL 1: Load datasets
# -------------------------------
weather_df      = pd.read_csv("all_tanks_combined.csv")
metadata_df     = pd.read_csv("tank_metadata.csv")
connectivity_df = pd.read_csv("connectivity.csv")   # now uses 'alpha' column
module1_df      = pd.read_csv("Module1_FINAL_WITH_VOLUME_2021_2025.csv")

# --- Parse weather dates ---
weather_df["date"] = pd.to_datetime(weather_df["date"], format='mixed')
weather_df["date"] = weather_df["date"].dt.normalize()
weather_df = weather_df.sort_values(by=["tank_id", "date"])

# --- Parse Module 1 dates ---
module1_df["date"] = pd.to_datetime(module1_df["date"], format='mixed')
module1_df["date"] = module1_df["date"].dt.normalize()

# --- Resolve true satellite observation date ---
# date = S1 grid/record date
# s2_date = actual optical acquisition date when S2_MNDWI substitution happened
# observation_date = the real date when the volume measurement occurred

if "s2_date" in module1_df.columns:
    module1_df["s2_date"] = pd.to_datetime(
        module1_df["s2_date"],
        format='mixed',
        errors="coerce"
    )
    module1_df["s2_date"] = module1_df["s2_date"].dt.normalize()

    module1_df["observation_date"] = (
        module1_df["s2_date"]
        .fillna(module1_df["date"])
    )

    substituted_rows = module1_df["s2_date"].notna().sum()

    print(
        f"Module 1 observation date resolved: "
        f"{substituted_rows} rows use s2_date, "
        f"{len(module1_df)-substituted_rows} rows use date."
    )

else:
    module1_df["observation_date"] = module1_df["date"]

    print(
        "Module 1 has no s2_date column. "
        "Using date as observation_date."
    )

if "tank_id" not in module1_df.columns:
    print("NOTE: Module 1 file has no tank_id column — falling back to "
          "pond_name -> tank_id mapping via tank_metadata.csv")
    name_to_id = metadata_df.set_index("tank_name")["tank_id"].to_dict()
    module1_df["tank_id"] = module1_df["pond_name"].map(name_to_id)
else:
    name_to_id_metadata = metadata_df.set_index("tank_name")["tank_id"].to_dict()
    check_df = module1_df[["pond_name", "tank_id"]].drop_duplicates()
    check_df["tank_id_from_metadata"] = check_df["pond_name"].map(name_to_id_metadata)

    mismatches = check_df[
        check_df["tank_id_from_metadata"].notna() &
        (check_df["tank_id"] != check_df["tank_id_from_metadata"])
    ]
    if len(mismatches) > 0:
        print(f"WARNING: {len(mismatches)} tank(s) have a DIFFERENT tank_id in "
              f"tank_metadata.csv than in Module 1's output. Using Module 1's "
              f"tank_id (canonical) — but this mismatch should be fixed at the "
              f"source (tank_metadata.csv) rather than silently overridden:")
        print(mismatches)

missing = module1_df[module1_df["tank_id"].isna()]["pond_name"].unique()
if len(missing) > 0:
    print(f"WARNING: {len(missing)} pond_name(s) in Module 1 output could not "
          f"be assigned a tank_id: {missing}")

module1_df = (
    module1_df
    .dropna(subset=["tank_id"])
    .sort_values(["tank_id", "observation_date"])
)

df = weather_df.merge(metadata_df, on=["tank_id", "tank_name"])

print(f"Loaded {df['tank_id'].nunique()} tanks, {len(df)} weather rows")
print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Module 1: {module1_df['tank_id'].nunique()} tanks, {len(module1_df)} satellite observations")

_m1_start, _m1_end = (
    module1_df["observation_date"].min(),
    module1_df["observation_date"].max()
)
_rows_before = len(df)
df = df[(df["date"] >= _m1_start) & (df["date"] <= _m1_end)].reset_index(drop=True)
print(f"Restricted to Module 1 coverage ({_m1_start.date()} to {_m1_end.date()}): "
      f"{_rows_before} -> {len(df)} rows")

# -------------------------------
# Connectivity map — now keyed on alpha (physical transfer coefficient),
# not gnn_influence_value.
# -------------------------------
connectivity_map = {}
for _, row in connectivity_df.iterrows():
    connectivity_map.setdefault(row["to_tank"], []).append(
        (row["from_tank"], row["alpha"])
    )

# -------------------------------
# Parameters
# -------------------------------
# UPSTREAM_FACTOR was originally tuned against gnn_influence_value weights
# (~0.05-0.34 range). alpha_water_transfer_coefficient sits much higher
# (~0.84-0.97 range) since it represents actual conserved-water fraction,
# not a GNN sensitivity score. The two are NOT on the same scale.
#
# Starting point below is a rough rescale (old weights were roughly 6x
# smaller on average than the new alpha values), so UPSTREAM_FACTOR is
# cut proportionally as a first pass. TREAT THIS AS PROVISIONAL:
# after running this script, check final_df['storage'] distribution
# (should stay mostly within [0,100] and produce a plausible mix of
# drought/normal/overflow days) and adjust UPSTREAM_FACTOR if the
# resulting storage curves look saturated or flattened.
UPSTREAM_FACTOR = 2.5          # was 15 — provisional, re-validate against storage distribution
LONG_GAP_THRESHOLD_DAYS = 20
FALLBACK_STORAGE = 50.0

# ============================================================
# PCHIP interpolation of fill_pct to daily resolution
# UPDATED: now returns tank_peak_volume_mcm alongside the daily
# storage records, instead of discarding it after normalizing.
# ============================================================
def interpolate_storage_pchip(module1_df, all_dates):
    daily_records = []
    tank_peak_volume_mcm = {}   # NEW: capture what was previously discarded

    for tank_id in module1_df["tank_id"].unique():
        tank_obs = (
            module1_df[module1_df["tank_id"] == tank_id]
            .sort_values("observation_date")
        )

        tank_obs = tank_obs.groupby("observation_date", as_index=False).agg({
            "volume_mcm": "mean",
            "volume_confidence": "first"
        })

        if len(tank_obs) < 2:
            print(f"  Skipping {tank_id}: only {len(tank_obs)} usable observation(s)")
            continue

        obs_dates = tank_obs["observation_date"].values
        obs_volume = tank_obs["volume_mcm"].values

        tank_peak_volume = obs_volume.max()
        if tank_peak_volume <= 0:
            print(f"  Skipping {tank_id}: zero/negative peak volume, cannot normalize")
            continue
        obs_fill = (obs_volume / tank_peak_volume) * 100

        tank_peak_volume_mcm[int(tank_id)] = float(tank_peak_volume)   # NEW: save it

        base_confidence_tier = tank_obs["volume_confidence"].iloc[0]

        ref_date = obs_dates[0]
        obs_days = ((obs_dates - ref_date) / np.timedelta64(1, "D")).astype(int)
        all_days = ((all_dates.values - ref_date) / np.timedelta64(1, "D")).astype(int)

        pchip = PchipInterpolator(obs_days, obs_fill, extrapolate=True)
        interpolated_fill = pchip(all_days)
        interpolated_fill = np.clip(interpolated_fill, 0, 100)

        dist_to_nearest_obs = np.array([
            np.min(np.abs(obs_days - d)) for d in all_days
        ])
        distance_confidence = np.clip(
            1.0 - (dist_to_nearest_obs / LONG_GAP_THRESHOLD_DAYS), 0.3, 1.0
        )

        tier_multiplier = 1.0 if "Validated" in str(base_confidence_tier) else 0.85

        is_extrapolated = (all_days < obs_days.min()) | (all_days > obs_days.max())

        final_confidence = distance_confidence * tier_multiplier
        final_confidence[is_extrapolated] *= 0.7

        for date, fill_val, conf in zip(all_dates, interpolated_fill, final_confidence):
            daily_records.append({
                "tank_id": tank_id,
                "date": date,
                "storage": fill_val,
                "storage_confidence": conf,
                "is_extrapolated": bool(date < obs_dates.min() or date > obs_dates.max())
            })

    return pd.DataFrame(daily_records), tank_peak_volume_mcm


print("\nInterpolating Module 1 fill_pct to daily resolution using PCHIP...")

module1_coverage_start = module1_df["observation_date"].min()
module1_coverage_end   = module1_df["observation_date"].max()

print(f"Module 1 observed coverage: {module1_coverage_start.date()} to {module1_coverage_end.date()}")
print(f"Weather data range:         {df['date'].min().date()} to {df['date'].max().date()}")

if df["date"].min() < module1_coverage_start or df["date"].max() > module1_coverage_end:
    print(f"WARNING: weather data extends beyond Module 1's satellite coverage. "
          f"Restricting the training dataset to the covered window "
          f"({module1_coverage_start.date()} to {module1_coverage_end.date()}) "
          f"to avoid long-range PCHIP extrapolation artifacts.")

all_dates = pd.date_range(
    start=max(df["date"].min(), module1_coverage_start),
    end=min(df["date"].max(), module1_coverage_end),
    freq="D"
)
daily_storage_df, tank_peak_volume_mcm = interpolate_storage_pchip(module1_df, all_dates)
print(f"  Created {len(daily_storage_df)} daily storage records")
print(f"  Mean confidence: {daily_storage_df['storage_confidence'].mean():.2f}")

# NEW: save the raw MCM peaks — needed by inference.py to convert
# live volume_mcm readings to the same 0-100% scale used in training.
with open("tank_peak_volumes.json", "w") as f:
    json.dump(tank_peak_volume_mcm, f, indent=2)
print(f"Saved tank_peak_volumes.json for {len(tank_peak_volume_mcm)} tanks")

storage_by_date = {}
for _, row in daily_storage_df.iterrows():
    d, t = row["date"], row["tank_id"]
    storage_by_date.setdefault(d, {})[t] = {
        "storage": row["storage"],
        "confidence": row["storage_confidence"]
    }

print("Cell 1 complete.")


# ============================================================
# CELL 2: Compute upstream_inflow using real interpolated storage
# and the physical alpha transfer coefficient
# ============================================================
results = []
unique_dates = sorted(df["date"].unique())

for date in unique_dates:
    daily_df = df[df["date"] == date]

    upstream_inflow_dict = {}
    for tank_id in df["tank_id"].unique():
        inflow = 0.0
        if tank_id in connectivity_map:
            for upstream_tank, alpha in connectivity_map[tank_id]:
                if date in storage_by_date and upstream_tank in storage_by_date[date]:
                    upstream_storage = storage_by_date[date][upstream_tank]["storage"]
                else:
                    upstream_storage = FALLBACK_STORAGE
                inflow += upstream_storage * alpha * UPSTREAM_FACTOR
        upstream_inflow_dict[tank_id] = inflow

    for _, row in daily_df.iterrows():
        tank_id = row["tank_id"]

        if date in storage_by_date and tank_id in storage_by_date[date]:
            real_storage   = storage_by_date[date][tank_id]["storage"]
            storage_conf   = storage_by_date[date][tank_id]["confidence"]
        else:
            real_storage   = FALLBACK_STORAGE
            storage_conf   = 0.0

        results.append({
            "tank_id"             : tank_id,
            "tank_name"           : row["tank_name"],
            "date"                : date,
            "precipitation_sum"   : row["precipitation_sum"],
            "temperature_2m_mean" : row["temperature_2m_mean"],
            "et0"                 : row["et0"],
            "catchment_area_km2"  : row["catchment_area_km2"],
            "runoff_coeff"        : row["runoff_coeff"],
            "upstream_inflow"     : upstream_inflow_dict[tank_id],
            "storage"             : real_storage,
            "storage_confidence"  : storage_conf,
        })

final_df = pd.DataFrame(results)
final_df.to_csv("synthetic_storage_base.csv", index=False)

print(f"Saved: {len(final_df)} rows, {final_df['tank_id'].nunique()} tanks")
print(f"Storage range: {final_df['storage'].min():.2f}-{final_df['storage'].max():.2f}%")
print(f"Mean storage_confidence: {final_df['storage_confidence'].mean():.2f}")
print(final_df.head())