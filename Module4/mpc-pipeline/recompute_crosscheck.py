"""
recompute_crosscheck.py
========================
Corrects every day's module3_crosscheck.csv for the MODULE3_STORAGE_BASIS
fix (area -> volume), WITHOUT re-running the simulation.

WHY THIS IS SAFE AND SUFFICIENT
--------------------------------
The cross-check compares two independently-computed ratios:

    m4_volume_ratio  - Module 4's own simulated trajectory, relative to its
                       own baseline day. Computed from state/release/inflow -
                       has NOTHING to do with MODULE3_STORAGE_BASIS and is
                       therefore ALREADY CORRECT in every saved file.

    m3_volume_ratio   - Module 3's forecast percentage, relative to its own
                       baseline day, optionally raised to AREA_TO_VOLUME_
                       EXPONENT if MODULE3_STORAGE_BASIS == "area". This is
                       the ONLY thing the flag changes, and it is a pure
                       function of the raw t+1..t+7 percentages already
                       sitting in the forecast dataset - nothing about the
                       optimisation, release decision, or storage evolution
                       is involved.

So this script re-derives m3_volume_ratio directly from the raw forecast
CSV under the corrected basis, keeps m4_volume_ratio exactly as originally
computed, and rewrites ratio_difference / abs_ratio_difference from the two.
Every other output file (mpc_decisions.csv, day_summary.json's storage and
selected blocks, topsis_weights_log.csv) is untouched, because none of them
depend on this flag.

USAGE
-----
    python recompute_crosscheck.py output_25-03-08 \\
        --forecast data/module4_forecasts_mar_aug_2025.csv

    python recompute_crosscheck.py output \\
        --forecast "data/module4_forecasts_mar_jul_2026_tft.csv"

By default this OVERWRITES each day's module3_crosscheck.csv in place. Use
--dry-run first to see the before/after summary without writing anything.

AFTER RUNNING THIS
-------------------
Re-run evaluate_season.py and generate_plots.py as normal - both read the
now-corrected CSVs and need no changes themselves:

    python evaluate_season.py <folder> --season yala --duration 135 \\
        --demand-csv data/demand.csv --tank-params-csv data/tank_params.csv
    python generate_plots.py <folder>/_evaluation

evaluate_season.py's own "anchor_correction" figure is a SEPARATE, further
correction (excluding small-anchor tank-days) applied on top of whatever is
in the CSVs - after this script runs, that figure becomes "volume basis AND
anchor-excluded", while mean_direction_agreement_pct_RAW becomes "volume
basis only". Both are now meaningful; neither was, before this fix.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", help="the run's output folder, e.g. output_25-03-08")
    ap.add_argument("--forecast", required=True,
                    help="path to the Module 1+3 forecast CSV used for this run")
    ap.add_argument("--baseline-day", type=int, default=1,
                    help="forecast day used as the ratio anchor (matches "
                         "CROSSCHECK_BASELINE_DAY in config.py, default 1)")
    ap.add_argument("--divergence-threshold", type=float, default=0.15,
                    help="mean |ratio diff| above which a tank is flagged "
                         "(matches CROSSCHECK_DIVERGENCE_THRESHOLD, default 0.15)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the before/after summary without writing any file")
    return ap.parse_args()


def load_tank_index_map(coords_path=os.path.join("data_", "tank_coordinates.csv")):
    coords = pd.read_csv(coords_path)
    return list(coords["tank_id"])


def load_forecast_dataset(path, tank_names):
    df = pd.read_csv(path)
    df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
    index_to_name = {i + 1: name for i, name in enumerate(tank_names)}
    df["tank_name"] = df["tank_id"].map(index_to_name)
    return df


def corrected_m3_ratio(pct_row, baseline_day):
    """
    pct_row : Series/array of length T, t+1..t+T percentages for one tank on
              one forecast date, in day order.
    Returns the volume ratio series under MODULE3_STORAGE_BASIS='volume':
    simply the raw percentage ratio to the baseline day, NO exponent.
    """
    vals = np.asarray(pct_row, dtype=float)
    base = vals[baseline_day - 1]
    if not np.isfinite(base) or base <= 0:
        return np.full_like(vals, np.nan)
    return vals / base


def main():
    args = parse_args()

    day_dirs = sorted(d for d in glob.glob(os.path.join(args.output_dir, "*"))
                      if os.path.isdir(d) and os.path.exists(
                          os.path.join(d, "module3_crosscheck.csv")))
    if not day_dirs:
        raise SystemExit(f"No module3_crosscheck.csv files found under "
                         f"{args.output_dir}/*/")

    names = load_tank_index_map()
    fc = load_forecast_dataset(args.forecast, names)
    t_cols = [f"t+{d}" for d in range(1, 8)]
    missing_t = [c for c in t_cols if c not in fc.columns]
    if missing_t:
        raise SystemExit(f"forecast dataset is missing columns {missing_t} - "
                         f"cannot recompute.")

    n_days_fixed = 0
    n_days_no_forecast_row = 0
    before_diffs, after_diffs = [], []
    before_agree, after_agree = [], []

    print(f"Recomputing cross-check under MODULE3_STORAGE_BASIS='volume' "
         f"for {len(day_dirs)} day(s)...")
    if args.dry_run:
        print("(--dry-run: no files will be written)\n")

    for d in day_dirs:
        date_str = os.path.basename(d)
        try:
            date = pd.Timestamp(date_str).date()
        except (ValueError, TypeError):
            continue

        cc_path = os.path.join(d, "module3_crosscheck.csv")
        cc = pd.read_csv(cc_path)
        if "m4_volume_ratio" not in cc.columns or "tank_id" not in cc.columns:
            continue

        day_rows = fc[fc["forecast_date"] == date]
        if day_rows.empty:
            n_days_no_forecast_row += 1
            continue
        day_rows = day_rows.set_index("tank_name")

        before_diffs.extend(cc["abs_ratio_difference"].dropna().tolist())
        last_day = cc["day"].max()
        before_sub = cc[cc["day"] == last_day].copy()
        before_agree.append(before_sub)

        new_m3 = np.full(len(cc), np.nan)
        for tank_id in cc["tank_id"].unique():
            if tank_id not in day_rows.index:
                continue
            pct = day_rows.loc[tank_id, t_cols]
            ratio = corrected_m3_ratio(pct, args.baseline_day)
            mask = cc["tank_id"] == tank_id
            # cc rows for this tank are day 1..T in order
            idx = cc.loc[mask].sort_values("day").index
            new_m3[idx] = ratio[:len(idx)]

        cc["m3_volume_ratio"] = new_m3
        cc["ratio_difference"] = cc["m4_volume_ratio"] - cc["m3_volume_ratio"]
        cc["abs_ratio_difference"] = cc["ratio_difference"].abs()

        after_diffs.extend(cc["abs_ratio_difference"].dropna().tolist())
        after_sub = cc[cc["day"] == last_day].copy()
        after_agree.append(after_sub)

        if not args.dry_run:
            cc.to_csv(cc_path, index=False)
        n_days_fixed += 1

    def direction_agreement(subframes):
        if not subframes:
            return None
        all_rows = pd.concat(subframes, ignore_index=True)
        agree = np.sign(all_rows["m4_volume_ratio"] - 1) == \
                np.sign(all_rows["m3_volume_ratio"] - 1)
        return float(agree.mean()) * 100

    print(f"\ndays with a matching forecast row : {n_days_fixed}")
    if n_days_no_forecast_row:
        print(f"days with NO matching forecast row : {n_days_no_forecast_row} "
             f"(left unchanged - likely error/gap days)")

    print(f"\n{'':20}{'BEFORE (area)':>18}{'AFTER (volume)':>18}")
    print(f"{'mean |ratio diff|':<20}{np.mean(before_diffs):>18.4f}"
         f"{np.mean(after_diffs):>18.4f}")
    print(f"{'direction agreement':<20}{direction_agreement(before_agree):>17.2f}%"
         f"{direction_agreement(after_agree):>17.2f}%")

    if args.dry_run:
        print("\n--dry-run: no files written. Re-run without --dry-run to apply.")
    else:
        print(f"\n{n_days_fixed} file(s) updated in place under {args.output_dir}/")
        print("Next: re-run evaluate_season.py and generate_plots.py - both "
             "read these CSVs and need no changes.")


if __name__ == "__main__":
    main()