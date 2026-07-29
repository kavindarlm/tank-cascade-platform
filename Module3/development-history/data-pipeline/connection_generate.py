import pandas as pd

# -------------------------------
# Load datasets
# -------------------------------
influence_df = pd.read_csv("alpha_edges.csv")  # output from module 2 (physical channel model)
tanks_df     = pd.read_csv("tanks.csv")

# -------------------------------
# DIRECT-ONLY CHECK
# alpha_edges.csv is described as covering "each validated direct
# hydraulic connection" — i.e. it should already be direct-only, with
# no hops column to filter on. This assertion catches it early if that
# assumption ever turns out to be wrong (e.g. she later adds multi-hop
# paths), instead of silently mixing hop levels into the pipeline.
# -------------------------------
if "hops" in influence_df.columns:
    n_indirect = (influence_df["hops"] != 1).sum()
    if n_indirect > 0:
        raise ValueError(
            f"alpha_edges.csv unexpectedly contains a 'hops' column with "
            f"{n_indirect} non-direct row(s). Confirm with Module 2 whether "
            f"this file now includes indirect paths before proceeding — "
            f"the old hops==1 filter needs to be reinstated if so."
        )
    influence_df = influence_df[influence_df["hops"] == 1].reset_index(drop=True)
else:
    print("✓ No 'hops' column present — treating alpha_edges.csv as direct-only, as documented.")

print(f"✓ Direct connections: {len(influence_df)} rows")
print(influence_df[["from_tank", "to_tank", "alpha_water_transfer_coefficient"]])

# -------------------------------
# Build name → id mapping
# -------------------------------
tanks_df["tank_name_clean"] = tanks_df["tank_name"].str.strip()
name_to_id = dict(zip(tanks_df["tank_name_clean"], tanks_df["tank_id"]))

# -------------------------------
# Clean influence data names
# -------------------------------
influence_df["from_tank_clean"] = influence_df["from_tank"].str.strip()
influence_df["to_tank_clean"]   = influence_df["to_tank"].str.strip()

# -------------------------------
# Map names to IDs
# -------------------------------
influence_df["from_tank_id"] = influence_df["from_tank_clean"].map(name_to_id)
influence_df["to_tank_id"]   = influence_df["to_tank_clean"].map(name_to_id)

# -------------------------------
# Check for unmatched names
# -------------------------------
unmatched_from = influence_df[influence_df["from_tank_id"].isna()]["from_tank_clean"].unique()
unmatched_to   = influence_df[influence_df["to_tank_id"].isna()]["to_tank_clean"].unique()

if len(unmatched_from) > 0:
    print(f"\n⚠️  Unmatched from_tank names: {unmatched_from}")
if len(unmatched_to) > 0:
    print(f"\n⚠️  Unmatched to_tank names: {unmatched_to}")

before = len(influence_df)
influence_df = influence_df.dropna(subset=["from_tank_id", "to_tank_id"])
after  = len(influence_df)
if before != after:
    print(f"\n⚠️  Dropped {before - after} rows due to unmatched names")

influence_df["from_tank_id"] = influence_df["from_tank_id"].astype(int)
influence_df["to_tank_id"]   = influence_df["to_tank_id"].astype(int)

# -------------------------------
# Sanity check: alpha must be in [0, 1]
# -------------------------------
bad_alpha = influence_df[
    (influence_df["alpha_water_transfer_coefficient"] < 0) |
    (influence_df["alpha_water_transfer_coefficient"] > 1)
]
if len(bad_alpha) > 0:
    print(f"\n⚠️  {len(bad_alpha)} row(s) have alpha outside [0,1] — check with Module 2:")
    print(bad_alpha)

# -------------------------------
# Sanity check: per-source Σα should be ≤ 0.98 (per Module 2's spec)
# -------------------------------
alpha_sums = influence_df.groupby("from_tank_id")["alpha_water_transfer_coefficient"].sum()
over_cap = alpha_sums[alpha_sums > 0.98]
if len(over_cap) > 0:
    print(f"\n⚠️  {len(over_cap)} source tank(s) exceed the Σα ≤ 0.98 cap — check with Module 2:")
    print(over_cap)

# -------------------------------
# Build connectivity.csv (direct connections only)
# -------------------------------
connectivity_df = influence_df[[
    "from_tank_id",
    "to_tank_id",
    "alpha_water_transfer_coefficient"
]].rename(columns={
    "from_tank_id"                   : "from_tank",
    "to_tank_id"                     : "to_tank",
    "alpha_water_transfer_coefficient": "alpha"
})

connectivity_df = connectivity_df.sort_values(
    ["from_tank", "to_tank"]
).reset_index(drop=True)

print(f"\n✓ connectivity.csv: {len(connectivity_df)} direct connections")
print(connectivity_df)

# -------------------------------
# Build downstream_demand per tank
# Only counts direct downstream tanks (1 hop)
# -------------------------------
downstream_count = (
    connectivity_df
    .groupby("from_tank")["to_tank"]
    .nunique()
    .reset_index()
    .rename(columns={"from_tank": "tank_id", "to_tank": "downstream_demand"})
)

# total_outflow_alpha: sum of α to all direct downstream tanks.
# Physically meaningful now — it's the fraction of this tank's outflow
# that is conserved and reaches *mapped* downstream neighbours;
# (1 - total_outflow_alpha) is loss to seepage/evaporation/unmapped paths.
downstream_alpha = (
    connectivity_df
    .groupby("from_tank")["alpha"]
    .sum()
    .reset_index()
    .rename(columns={"from_tank": "tank_id", "alpha": "total_outflow_alpha"})
)

tank_network_features = tanks_df[["tank_id", "tank_name"]].merge(
    downstream_count, on="tank_id", how="left"
).merge(
    downstream_alpha, on="tank_id", how="left"
)

tank_network_features["downstream_demand"]    = \
    tank_network_features["downstream_demand"].fillna(0).astype(int)
tank_network_features["total_outflow_alpha"]  = \
    tank_network_features["total_outflow_alpha"].fillna(0.0)

print(f"\n✓ Tank network features:")
print(tank_network_features.to_string(index=False))

# -------------------------------
# Save both files
# -------------------------------
connectivity_df.to_csv("connectivity.csv", index=False)
tank_network_features.to_csv("tank_network_features.csv", index=False)

print("\nSaved: connectivity.csv")
print("Saved: tank_network_features.csv")