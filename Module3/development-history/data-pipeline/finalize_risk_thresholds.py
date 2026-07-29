# ============================================================
# FINAL RISK THRESHOLD LOGIC — symmetric per-tank relative
# thresholds for BOTH drought and overflow, computed from
# training-period data only, saved as a fixed artifact.
#
# UPDATED: now merges in tank_peak_volume_mcm from
# tank_peak_volumes.json (produced by storage_generate.py), so
# risk_thresholds.json is the ONE file inference.py needs to load.
# ============================================================
import pandas as pd
import json

DROUGHT_REL_FRAC  = 0.20   # drought  = below 20% of this tank's own peak
OVERFLOW_REL_FRAC = 0.75   # overflow = above 75% of this tank's own peak

df = pd.read_csv("final_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

sorted_dates = df["date"].sort_values().unique()
cutoff_date = sorted_dates[int(len(sorted_dates) * 0.8)]
print(f"Computing tank_peak using data before {pd.Timestamp(cutoff_date).date()} only")

peak_source = df[df["date"] < cutoff_date]
tank_peak = peak_source.groupby("tank_id")["storage"].max()

missing_tanks = set(df["tank_id"].unique()) - set(tank_peak.index)
if missing_tanks:
    fallback_peak = df.groupby("tank_id")["storage"].max()
    for t in missing_tanks:
        tank_peak[t] = fallback_peak[t]

drought_threshold_per_tank  = (tank_peak * DROUGHT_REL_FRAC).to_dict()
overflow_threshold_per_tank = (tank_peak * OVERFLOW_REL_FRAC).to_dict()

# NEW: load the raw MCM peaks produced by storage_generate.py
with open("tank_peak_volumes.json") as f:
    tank_peak_volume_mcm = json.load(f)

with open("risk_thresholds.json", "w") as f:
    json.dump({
        "drought_relative_frac": DROUGHT_REL_FRAC,
        "overflow_relative_frac": OVERFLOW_REL_FRAC,
        "tank_peak": tank_peak.to_dict(),
        "tank_peak_volume_mcm": tank_peak_volume_mcm,   # NEW
        "drought_threshold_per_tank": drought_threshold_per_tank,
        "overflow_threshold_per_tank": overflow_threshold_per_tank,
        "upstream_factor": 2.5,   # NEW — must match storage_generate.py's UPSTREAM_FACTOR exactly
        "computed_from_data_before": str(pd.Timestamp(cutoff_date).date())
    }, f, indent=2)

print("Saved risk_thresholds.json — reuse at inference time, never recompute live.")
print(f"  Includes tank_peak_volume_mcm for {len(tank_peak_volume_mcm)} tanks")

df["drought_threshold_this_tank"]  = df["tank_id"].map(drought_threshold_per_tank)
df["overflow_threshold_this_tank"] = df["tank_id"].map(overflow_threshold_per_tank)

def assign_risk_row(row):
    """
    - drought:  storage < 20% of THIS TANK'S OWN observed peak
    - overflow: storage > 75% of THIS TANK'S OWN observed peak
    - normal:   everything in between
    Both thresholds are per-tank relative, symmetric definitions —
    consistent methodology for both extreme classes, avoiding the
    earlier problem where a fixed absolute drought floor caused
    some tanks to be "always drought" (>70% of days) and others
    "never drought" (0% of days).
    """
    storage = row["target_t+1"]
    if storage < row["drought_threshold_this_tank"]:
        return 0
    elif storage > row["overflow_threshold_this_tank"]:
        return 2
    else:
        return 1

df["risk_class"] = df.apply(assign_risk_row, axis=1)
df.to_csv("final_dataset.csv", index=False)

print("\nFinal class distribution:")
print(df["risk_class"].value_counts(normalize=True).round(3))

print("\nPer-tank drought representation (should be far more even than before):")
below_drought_by_tank = df.groupby("tank_id").apply(
    lambda g: (g["target_t+1"] < g["drought_threshold_this_tank"]).mean() * 100
)
print(below_drought_by_tank.sort_values(ascending=False))

print("\nPer-tank overflow representation:")
above_overflow_by_tank = df.groupby("tank_id").apply(
    lambda g: (g["target_t+1"] > g["overflow_threshold_this_tank"]).mean() * 100
)
print(above_overflow_by_tank.sort_values(ascending=False))