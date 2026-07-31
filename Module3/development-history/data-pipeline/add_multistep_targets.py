import pandas as pd

df = pd.read_csv("synthetic_storage_base.csv")
network_df = pd.read_csv("tank_network_features.csv")

df = df.sort_values(["tank_id", "date"]).reset_index(drop=True)

df = df.merge(
    network_df[["tank_id", "downstream_demand", "total_outflow_alpha"]],  # was total_outflow_influence
    on="tank_id", how="left"
)

for i in range(1, 8):
    df[f"target_t+{i}"] = df.groupby("tank_id")["storage"].shift(-i)

df = df.dropna(subset=[f"target_t+{i}" for i in range(1, 8)])
df.to_csv("final_dataset.csv", index=False)