"""
build_tank_params.py
===================
Regenerate tank_params.csv from the merged tank dataset using the CURRENT
config values for SHAPE_FACTOR, DEAD_STORAGE_FRACTION, MAX_RELEASE_FRACTION
and EVAP_MM_PER_DAY.

Because capacity (S_max) is derived rather than measured for most tanks, it is
the single most influential assumption in Module 4: f2 (overflow) and C1 both
scale directly with it. Keeping the derivation in one script driven by config
makes the sensitivity sweep trivial - change SHAPE_FACTOR, rerun, recompare.

Derivation
----------
    S_max = measured capacity                     if available (e.g. Nachchaduwa)
          = WSA_m2 * depth_m * SHAPE_FACTOR       otherwise
    S_min = S_max * DEAD_STORAGE_FRACTION         (ASSUMPTION)
    R_max = S_max * MAX_RELEASE_FRACTION          (ASSUMPTION)
    E     = EVAP_MM_PER_DAY/1000 * WSA_m2         (m3/day, constant across horizon)

Usage
-----
    python build_tank_params.py                        # uses config defaults
    python build_tank_params.py --shape-factor 0.69    # override for a sweep
    python build_tank_params.py --out data/tank_params_sf070.csv
"""

import argparse
import os
import numpy as np
import pandas as pd

from module4.config import Config


MERGED_DEFAULT = "/mnt/user-data/outputs/Merged_Tank_Data.xlsx"


def to_id(name):
    return str(name).strip().replace(" ", "_").replace("__", "_")


def build(merged_path, cfg: Config, out_path):
    df = pd.read_excel(merged_path, sheet_name="Merged_Data")
    N = len(df)
    df["tank_id"] = df["tank_name"].apply(to_id)

    S_max = np.full(N, np.nan)
    source = []

    for i in range(N):
        row = df.iloc[i]
        if pd.notna(row.get("capacity_acre_feet")):
            # Measured capacity - bypass the shape factor entirely
            S_max[i] = row["capacity_acre_feet"] * cfg.ACRE_FT_TO_M3
            source.append("measured")
        elif pd.notna(row.get("tank_area_sqkm")) and pd.notna(row.get("depth_m")):
            wsa_m2 = row["tank_area_sqkm"] * cfg.SQKM_TO_SQM
            S_max[i] = wsa_m2 * row["depth_m"] * cfg.SHAPE_FACTOR
            source.append(f"derived (SF={cfg.SHAPE_FACTOR})")
        else:
            source.append("MISSING")

    S_min = np.where(~np.isnan(S_max), S_max * cfg.DEAD_STORAGE_FRACTION, np.nan)
    R_max = np.where(~np.isnan(S_max), S_max * cfg.MAX_RELEASE_FRACTION, np.nan)

    # Evaporation volume from water spread area
    E = np.full(N, np.nan)
    wsa = df["tank_area_sqkm"]
    have_wsa = wsa.notna().to_numpy()
    E[have_wsa] = (cfg.EVAP_MM_PER_DAY / 1000.0) * \
                  wsa[have_wsa].to_numpy() * cfg.SQKM_TO_SQM

    params = pd.DataFrame({"tank_id": df["tank_id"]})
    params["S_min"] = np.round(S_min, 1)
    params["S_max"] = np.round(S_max, 1)
    params["R_max"] = np.round(R_max, 1)
    for d in range(cfg.T):
        params[f"E_day{d}"] = np.round(E, 2)
    params["catchment_area_km2"] = np.round(
        df["catchment_area_km2"].to_numpy(dtype=float), 4)
    params["S_max_source"] = source

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    params.to_csv(out_path, index=False)

    n_ok = int((~np.isnan(S_max)).sum())
    n_measured = source.count("measured")
    print(f"tank_params written -> {out_path}")
    print(f"  SHAPE_FACTOR = {cfg.SHAPE_FACTOR}")
    print(f"  tanks with capacity : {n_ok}/{N}  "
          f"({n_measured} measured, {n_ok - n_measured} derived)")
    print(f"  tanks MISSING       : {N - n_ok}")
    if n_ok:
        print(f"  S_max range         : "
              f"{np.nanmin(S_max):,.0f} - {np.nanmax(S_max):,.0f} m3")
        print(f"  total system capacity: {np.nansum(S_max):,.0f} m3")
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=MERGED_DEFAULT,
                    help="path to Merged_Tank_Data.xlsx")
    ap.add_argument("--shape-factor", type=float, default=None,
                    help="override cfg.SHAPE_FACTOR (for sensitivity sweeps)")
    ap.add_argument("--out", default=None, help="output CSV path")
    args = ap.parse_args()

    cfg = Config()
    if args.shape_factor is not None:
        cfg.SHAPE_FACTOR = args.shape_factor
    cfg.validate()

    out = args.out or os.path.join(cfg.DATA_DIR, cfg.FILE_TANK_PARAMS)
    build(args.merged, cfg, out)


if __name__ == "__main__":
    main()
