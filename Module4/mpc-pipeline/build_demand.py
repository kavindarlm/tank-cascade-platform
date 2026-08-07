"""
build_demand.py
==============
Convert tank command areas into a daily paddy demand matrix using the OFFICIAL
Sri Lankan government Crop Water Requirement (CWR) tables for Anuradhapura.

Source: CSIAP 2024, "District Wise Reference Crop Evapotranspiration (ETo) and
Crop Water Requirements (ETc)", Anuradhapura District, Paddy - Tables 5.4/5.5/5.6.

Handles BOTH seasons and ALL THREE paddy age classes:
    Maha : 135 / 105 / 90-day paddy
    Yala : 135 / 105 / 90-day paddy
(The seasonal length and the demand curve differ between them - Yala demand is
substantially higher because of hotter, drier weather.)

USAGE
-----
    # default: Maha, 135-day
    python build_demand.py

    # Yala season, 105-day paddy
    python build_demand.py --season yala --duration 105

    # custom command-area file and output
    python build_demand.py --command-area data/command_area.csv --out data/demand.csv

    # different conveyance efficiency (sensitivity)
    python build_demand.py --conveyance 0.65

The command-area file must have columns: tank_id, command_area_acres.
"""

import argparse
import numpy as np
import pandas as pd

from module4.config import Config
from module4.demand_generator import (
    generate_demand, CWR_SEASONAL_TOTAL, ANURADHAPURA_PADDY_CWR
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--command-area", default="data/command_area.csv",
                    help="CSV with tank_id, command_area_acres")
    ap.add_argument("--season", choices=["maha", "yala"], default="maha")
    ap.add_argument("--duration", type=int, choices=[135, 105, 90], default=135,
                    help="paddy crop duration (age class)")
    ap.add_argument("--conveyance", type=float, default=None,
                    help="override conveyance efficiency (default 0.70)")
    ap.add_argument("--out", default="data/demand.csv")
    args = ap.parse_args()

    cfg = Config()
    cfg.SEASON = args.season
    cfg.PADDY_DURATION_DAYS = args.duration
    if args.conveyance is not None:
        cfg.CONVEYANCE_EFFICIENCY = args.conveyance
    cfg.validate()

    # ---- Load command areas ----
    cmd = pd.read_csv(args.command_area)
    for col in ("tank_id", "command_area_acres"):
        if col not in cmd.columns:
            raise ValueError(f"command-area file must have a '{col}' column")

    # ---- Generate demand ----
    demand_m3, diag = generate_demand(cmd["command_area_acres"].to_numpy(), cfg)

    # ---- Write ----
    out = pd.DataFrame(demand_m3.round(1),
                       columns=[f"day{d}" for d in range(diag["season_days"])])
    out.insert(0, "tank_id", cmd["tank_id"].values)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    # ---- Report ----
    print("=" * 64)
    print(f"  DEMAND GENERATED  -  Anuradhapura paddy, {args.season.upper()} "
          f"{args.duration}-day")
    print("=" * 64)
    print(f"  Source            : CSIAP 2024 government CWR tables")
    print(f"  Tanks             : {len(cmd)}")
    print(f"  Season length     : {diag['season_days']} days")
    print(f"  Field CWR total   : {diag['field_cwr_total_mm']} mm  "
          f"(official {diag['official_total_mm']} mm)  "
          f"{'OK' if abs(diag['field_cwr_total_mm']-diag['official_total_mm'])<1 else 'CHECK'}")
    print(f"  Conveyance eff.   : {diag['conveyance_efficiency']}")
    print(f"  Gross at tank     : {diag['gross_at_tank_total_mm']} mm")
    print(f"  Total command area: {cmd['command_area_acres'].sum():,.0f} acres")
    print(f"  Peak daily demand : {demand_m3.max():,.0f} m3/day (single tank)")
    print(f"  Written -> {args.out}")

    # Show the demand curve shape (network total per stage)
    total_daily = demand_m3.sum(axis=0)
    print(f"\n  Network demand curve (total across all tanks):")
    marks = [(0, "land prep"), (25, "initial"), (55, "development"),
             (80, "mid-season"), (diag['season_days']-5, "harvest")]
    for day, label in marks:
        if day < diag['season_days']:
            print(f"    day {day:>3} ({label:<11}): {total_daily[day]:>12,.0f} m3/day")


if __name__ == "__main__":
    main()
