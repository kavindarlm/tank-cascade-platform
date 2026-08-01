"""
sensitivity_capacity.py
======================
Sweep SHAPE_FACTOR across its credible band and report how derived capacity -
and therefore the headroom available to the overflow objective f2 - responds.

WHY THIS MATTERS (Issue 1)
Capacity S_max is derived, not measured, for 22 of 23 usable tanks. Both f2
(overflow) and constraint C1 scale directly with it:
  - S_max too HIGH -> f2 can never fire; controller believes it has headroom
                      that does not exist and systematically under-releases.
  - S_max too LOW  -> phantom overflow; f2 fires on plans that would not spill.

The only calibration point is Nachchaduwa (measured 45,150 acre-ft AND known
WSA + depth), which implies SHAPE_FACTOR = 0.692. That tank is a major reservoir
in a defined valley; the Mahakanumulla village tanks are broader and shallower,
so the credible band is taken as 0.50 - 0.70 with 0.60 as the default.

This script quantifies the resulting uncertainty envelope so it can be reported
in the evaluation chapter rather than left as a hidden assumption.

Usage
-----
    python sensitivity_capacity.py
    python sensitivity_capacity.py --factors 0.45 0.5 0.6 0.7 0.75
"""

import argparse
import numpy as np
import pandas as pd

from module4.config import Config
from build_tank_params import build, MERGED_DEFAULT


def calibration_point(merged_path, cfg):
    """Recover the shape factor implied by the one tank with measured capacity."""
    df = pd.read_excel(merged_path, sheet_name="Merged_Data")
    m = df[df["capacity_acre_feet"].notna()]
    out = []
    for _, r in m.iterrows():
        if pd.isna(r.get("tank_area_sqkm")) or pd.isna(r.get("depth_m")):
            continue
        measured = r["capacity_acre_feet"] * cfg.ACRE_FT_TO_M3
        box = r["tank_area_sqkm"] * cfg.SQKM_TO_SQM * r["depth_m"]
        out.append((r["tank_name"], measured, box, measured / box))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=MERGED_DEFAULT)
    ap.add_argument("--factors", type=float, nargs="+",
                    default=[0.50, 0.55, 0.60, 0.65, 0.70])
    ap.add_argument("--out", default="outputs/sensitivity_capacity.csv")
    args = ap.parse_args()

    cfg = Config().validate()

    print("=" * 68)
    print("  CAPACITY SENSITIVITY  -  SHAPE_FACTOR sweep")
    print("=" * 68)

    cal = calibration_point(args.merged, cfg)
    if cal:
        print("\n  Calibration point(s) with MEASURED capacity:")
        for name, meas, box, sf in cal:
            print(f"    {name}: measured {meas:,.0f} m3 | area x depth {box:,.0f} m3"
                  f"  -> implied SHAPE_FACTOR = {sf:.3f}")

    print(f"\n  Sweeping SHAPE_FACTOR over {args.factors}\n")

    rows = []
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    for sf in args.factors:
        c = Config()
        c.SHAPE_FACTOR = sf
        c.validate()
        path = os.path.join(tmpdir, f"tp_{sf}.csv")
        # build() prints its own summary; capture the frame it returns
        params = build(args.merged, c, path)
        derived = params[params["S_max_source"].astype(str).str.startswith("derived")]
        rows.append({
            "shape_factor": sf,
            "total_capacity_m3": float(params["S_max"].sum(skipna=True)),
            "mean_derived_capacity_m3": float(derived["S_max"].mean()),
            "median_derived_capacity_m3": float(derived["S_max"].median()),
            "min_derived_capacity_m3": float(derived["S_max"].min()),
            "max_derived_capacity_m3": float(derived["S_max"].max()),
        })
        print()

    df = pd.DataFrame(rows)
    base = df[df["shape_factor"] == cfg.SHAPE_FACTOR]
    if len(base):
        b = float(base["total_capacity_m3"].iloc[0])
        df["pct_change_vs_default"] = ((df["total_capacity_m3"] / b - 1) * 100).round(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    print("=" * 68)
    print("  SUMMARY")
    print("=" * 68)
    show = df.copy()
    for c_ in show.columns:
        if c_.endswith("_m3"):
            show[c_] = show[c_].map(lambda v: f"{v:,.0f}")
    print(show.to_string(index=False))
    print(f"\n  Written -> {args.out}")
    print("\n  INTERPRETATION: total system capacity varies by "
          f"{df['total_capacity_m3'].max()/df['total_capacity_m3'].min():.2f}x "
          "across the credible band.")
    print("  Report f1/f2/f3/f4 seasonal totals at the band endpoints to show the")
    print("  strategy RANKING is stable even though absolute volumes shift.")


if __name__ == "__main__":
    main()
