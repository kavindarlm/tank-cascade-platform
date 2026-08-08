"""
run_daily_cli.py
=================
CLI wrapper around mpc_loop.run_mpc()/save_all() for a single live-Module-3
day. Exists so mpc_api.py can launch a daily run as an isolated subprocess
(the same approach run_simulation.py already uses for the season job),
without adding argparse to main.py itself.

Does NOT change any pipeline logic in mpc_loop/nsga2/topsis/objectives/
constraints - it only makes sure demand.csv reflects the RIGHT day of the
RIGHT season before those unmodified modules ever see it. See "SEASON /
DEMAND CORRECTNESS" below for why that step exists.

SEASON / DEMAND CORRECTNESS
----------------------------
data/demand.csv on disk is a SINGLE fixed season+duration snapshot (Maha,
135-day - see build_demand.py) indexed purely by "day N since planting", with
no calendar dates in it at all. mpc_loop.run_mpc()'s loop always starts at
t_current=0 for a live single day (SEASON_LENGTH is forced to 1), so reading
that file directly would silently use the SAME day-0..6 demand - the
land-preparation/soaking phase, the highest-demand days of the entire season
- for every RUN_DATE, forever, regardless of the season the date is actually
in.

To fix this without touching mpc_loop.py's well-tested loop, this script
follows the exact same recipe simulation_runner.run_one_day() already uses
for the season/backtest path: generate the full-season demand fresh for
THIS run's season/duration (module4.simulation_data.generate_season_demand),
slice out the T-day window starting at the correct day-of-season offset
(RUN_DATE - SEASON_START_DATE), and stage it - alongside byte-identical
copies of the other static inputs - into a scratch data directory that this
run's Config points DATA_DIR at. Module 3's live risk/storage inputs are
untouched by any of this; they already correctly track RUN_DATE via
data_loader.load_module3_risk_live / storage_from_module3_t1.

season/season-start are BEST-EFFORT DEFAULTS (derived from RUN_DATE's month,
Apr-Aug -> yala else maha, and the most recent Mar-1/Oct-1 on or before
RUN_DATE) when not passed explicitly - correct for the common case, but pass
--season/--season-start explicitly once the real planting date is known, since
demand accuracy depends entirely on that anchor.

Usage:
    python run_daily_cli.py --date 2026-07-29
    python run_daily_cli.py --date 2026-07-29 --fast-test
    python run_daily_cli.py --date 2026-07-29 --season yala --season-start 2026-04-15
"""
import argparse
import json
import os
import shutil
import sys
from datetime import date

import pandas as pd

from module4.config import Config
from module4.mpc_loop import run_mpc, save_all
from module4.demand_generator import demand_window, generate_demand
from module4 import simulation_data as sd


def _derive_season(run_date: date) -> str:
    """
    Best-effort season from the calendar month, matching config.py's own
    documented convention (maha Oct-Mar, yala Apr-Aug). Override with
    --season for a planting that doesn't follow the typical boundary.
    """
    return "yala" if 4 <= run_date.month <= 8 else "maha"


def _default_season_start(season: str, run_date: date) -> date:
    """
    Best-effort season start when --season-start is not given: the most
    recent Mar 1 (yala) / Oct 1 (maha) on or before run_date - the same
    start-date convention run_simulation.py's own usage examples already use
    (e.g. the original 2025 Yala run started 2025-03-01). Override with
    --season-start once the real planting date is known; demand accuracy
    depends entirely on this anchor.
    """
    month, day = (3, 1) if season == "yala" else (10, 1)
    candidate = date(run_date.year, month, day)
    if candidate > run_date:
        candidate = date(run_date.year - 1, month, day)
    return candidate


def _stage_daily_demand(cfg: Config, tank_names, day_offset: int) -> str:
    """
    Build a scratch data/ directory carrying the CORRECT season/duration
    demand window for this run (see module docstring), plus byte-identical
    copies of the other static inputs the live path can still fall back to
    (tank_storage.csv, for a tank Module 3 rejected today - see
    data_loader.storage_from_module3_t1). Returns the scratch directory path;
    caller points cfg.DATA_DIR at it.
    """
    # NOTE: deliberately NOT sd.generate_season_demand() here. That helper
    # passes cfg.SEASON/cfg.PADDY_DURATION_DAYS straight into
    # recover_command_area_acres()'s season/duration args too - but that
    # function inverts the ONE physical file that exists on disk
    # (data/demand.csv, built maha/135 - see build_demand.py), so recovery
    # must ALWAYS use maha/135 regardless of which season this run wants
    # demand FOR. Passing e.g. season="yala" through corrupts the recovered
    # command areas by ~30% (verified empirically) instead of raising - a
    # silent unit-basis mismatch, not a crash. Recovering with the correct,
    # fixed defaults first and applying THIS run's season/duration only to
    # generate_demand() (which only touches the CWR curve, not area
    # recovery) avoids it.
    area_acres = sd.recover_command_area_acres().reindex(tank_names)
    if area_acres.isna().any():
        raise ValueError(f"could not recover command area for tank(s): "
                         f"{list(area_acres[area_acres.isna()].index)}")
    demand_full, demand_diag = generate_demand(area_acres.to_numpy(), cfg)
    if day_offset >= demand_diag["season_days"]:
        print(f"  [!] day {day_offset + 1} is past the {demand_diag['season_days']}-day "
              f"{cfg.SEASON} season - holding the final day's demand "
              f"(harvest; no further irrigation expected).")
    dwin = demand_window(demand_full, day_offset, cfg.T)

    scratch_dir = "_daily_scratch"
    sd.stage_static_files(scratch_dir, source_data_dir=cfg.DATA_DIR)
    shutil.copyfile(os.path.join(cfg.DATA_DIR, cfg.FILE_TANK_STORAGE),
                    os.path.join(scratch_dir, cfg.FILE_TANK_STORAGE))

    demand_df = pd.DataFrame(dwin.round(2),
                             columns=[f"day{d}" for d in range(dwin.shape[1])])
    demand_df.insert(0, "tank_id", tank_names)
    demand_df.to_csv(os.path.join(scratch_dir, cfg.FILE_DEMAND), index=False)
    return scratch_dir


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="RUN_DATE, YYYY-MM-DD")
    ap.add_argument("--season", choices=["yala", "maha"], default=None,
                    help="crop season for demand lookup. Default: derived "
                         "from --date's month (Apr-Aug -> yala, else maha).")
    ap.add_argument("--duration", type=int, choices=[135, 105, 90], default=135,
                    help="paddy duration class in days. Default: 135.")
    ap.add_argument("--season-start", default=None,
                    help="YYYY-MM-DD the current season's cultivation began "
                         "(demand day 0). Default: most recent Mar 1 (yala) "
                         "/ Oct 1 (maha) on or before --date.")
    ap.add_argument("--fast-test", action="store_true",
                    help="shrink NSGA-II (pop=20, gen=10, K=5) for a quick run - "
                         "not for real results")
    args = ap.parse_args()

    run_date = date.fromisoformat(args.date)
    season = args.season or _derive_season(run_date)
    season_start = (date.fromisoformat(args.season_start) if args.season_start
                    else _default_season_start(season, run_date))
    day_offset = (run_date - season_start).days
    if day_offset < 0:
        print(f"ERROR: --date {run_date} is before the season start "
              f"{season_start} - day-of-season would be negative. Pass "
              f"--season-start explicitly if that boundary is wrong for "
              f"this run.", file=sys.stderr)
        return 1

    print(f"Season -> {season}   duration {args.duration} days   "
          f"season_start {season_start}   day_of_season {day_offset + 1}")

    overrides = {"POP_SIZE": 20, "N_GEN": 10, "K": 5} if args.fast_test else {}
    cfg = Config(MODULE3_LIVE_FORECAST=True, RUN_DATE=args.date,
                SEASON=season, PADDY_DURATION_DAYS=args.duration,
                SEASON_START_DATE=season_start.isoformat(),
                **overrides).validate()

    static_ref = sd.load_static_reference(Config())
    tank_names = static_ref["tank_ids"]
    cfg.DATA_DIR = _stage_daily_demand(cfg, tank_names, day_offset)

    results = run_mpc(cfg, season_length=cfg.SEASON_LENGTH, verbose=True)
    paths = save_all(results, cfg)

    wl = results["weights_log"][0] if results["weights_log"] else {}
    dec = results["decisions"][0] if results["decisions"] else None
    feasible = bool(wl.get("feasible", False))

    total_release_m3 = 0.0
    top_releases = []
    n_tanks = 0
    if dec is not None:
        n_tanks = len(dec["tank_ids"])
        total_release_m3 = round(float(sum(dec["R_today"])), 2)
        pairs = sorted(zip(dec["tank_ids"], dec["R_today"]), key=lambda x: -x[1])[:5]
        top_releases = [{"tank_id": t, "release_m3": round(float(r), 2)} for t, r in pairs]

    summary = {
        "date": args.date,
        "season": season,
        "duration_days": args.duration,
        "season_start": season_start.isoformat(),
        "day_of_season": day_offset + 1,
        "fast_test": args.fast_test,
        "feasible": feasible,
        "output_dir": cfg.OUTPUT_DIR,
        "paths": paths,
        "p_drought": wl.get("p_drought"),
        "p_overflow": wl.get("p_overflow"),
        "weights": {
            "shortage": wl.get("w_shortage"),
            "overflow": wl.get("w_overflow"),
            "equity": wl.get("w_equity"),
            "loss": wl.get("w_loss"),
        },
        "c3_tier": wl.get("c3_tier"),
        "reserve_deficit_m3": wl.get("reserve_deficit_m3"),
        "n_tanks": n_tanks,
        "total_release_m3": total_release_m3,
        "top_releases": top_releases,
    }
    status_path = os.path.join(cfg.OUTPUT_DIR, "last_run_status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
