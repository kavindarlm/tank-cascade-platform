"""
run_simulation.py
==================
Entry point for the day-by-day MO-MPC simulation.

Drives Module 1 + Module 3 inputs through the existing, unmodified MPC
pipeline (NSGA-II / TOPSIS / constraints / objectives) one day at a time.

This does NOT change Module 4's optimisation, MPC, NSGA-II, TOPSIS,
objective, or constraint logic in any way - it only adapts the daily inputs
(today's storage from t+1, today's risk, today's demand slice, the frozen
rainfall history) and calls the existing mpc_loop.run_mpc / output.save_all
once per simulated day.

Re-running with the same --output directory resumes: days that already
completed are skipped.

Usage - original 2025 Yala run (unchanged)
-------------------------------------------
    python run_simulation.py
    python run_simulation.py --start 2025-03-01 --end 2025-08-31

Usage - different year, same season
-------------------------------------
    python run_simulation.py \\
        --start 2024-03-01 --end 2024-08-31 \\
        --forecast data/module_forecasts_mar_aug_2024.csv \\
        --output output_2024

    Pre-requisite: fetch rainfall for the three years BEFORE that season:
        python fetch_rainfall.py --years 2021 2022 2023 --season 03-01 08-31

Usage - Maha season
---------------------
    python run_simulation.py \\
        --start 2025-10-01 --end 2026-02-28 \\
        --season maha --duration 135 \\
        --forecast data/module_forecasts_oct_feb_2025.csv \\
        --output output_maha

    Pre-requisite: fetch Maha rainfall (handles the Dec/Jan wrap automatically):
        python fetch_rainfall.py --years 2022 2023 2024 --season 10-01 02-28

    Note: for Maha the cascade should FILL, not deplete. The audit_day.py
    "FILLING" warning is correct behaviour in Maha - ignore it.
    Calibrate RUNOFF_COEFFICIENT separately for Maha (typically 0.20-0.30).

Usage - sub-range spot-check
------------------------------
    python run_simulation.py --start 2025-06-01 --end 2025-06-30

    Uses the forecast file and rainfall already in data/. No other changes.

Smoke test (fast, not real results)
-------------------------------------
    python run_simulation.py --fast-test --start 2025-03-01 --end 2025-03-01
"""

import argparse
from datetime import date

from module4.simulation_runner import run_simulation, SIM_START, SIM_END


# Season duration defaults - used when --duration is not supplied.
_DURATION_DEFAULTS = {
    "yala": 135,
    "maha": 135,
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # ---- date range --------------------------------------------------------
    ap.add_argument("--start", default=SIM_START.isoformat(),
                    help="first simulated day (YYYY-MM-DD). "
                         f"Default: {SIM_START.isoformat()} (original Yala 2025 run)")
    ap.add_argument("--end", default=SIM_END.isoformat(),
                    help="last simulated day, inclusive (YYYY-MM-DD). "
                         f"Default: {SIM_END.isoformat()}")

    # ---- season / demand ---------------------------------------------------
    ap.add_argument("--season", default="yala", choices=["yala", "maha"],
                    help="crop season: yala (Mar-Aug) or maha (Oct-Feb). "
                         "Sets the demand curve and the SEASON config flag. "
                         "Default: yala")
    ap.add_argument("--duration", type=int, default=None,
                    help="paddy duration class in days: 105, 135, or 90. "
                         "Default: 105 for yala, 135 for maha")

    # ---- data sources ------------------------------------------------------
    ap.add_argument("--forecast", default=None,
                    help="path to the Module 1+3 forecast CSV "
                         "(one row per tank per day). "
                         "Default: data/module_forecasts_and mar_aug_2025.csv "
                         "(the original 2025 Yala file). "
                         "Supply a different path for a different year or season.")

    # ---- output / scratch --------------------------------------------------
    ap.add_argument("--output", default="output",
                    help="root output directory (day folders + summary_NN folders). "
                         "Default: output")
    ap.add_argument("--static-dir", default="_sim_scratch",
                    help="scratch directory for the staged per-day input CSVs. "
                         "Default: _sim_scratch")
    ap.add_argument("--consolidated-output", action="store_true",
                    help="ADDITIVE, opt-in: also append every simulated day as "
                         "rows onto run-wide CSVs at the output root "
                         "(consolidated_mpc_decisions.csv, "
                         "consolidated_topsis_weights_log.csv, "
                         "consolidated_module3_crosscheck.csv, "
                         "consolidated_daily_summary.csv), so the whole run "
                         "can be read as a few day-indexed CSVs instead of "
                         "walking per-day folders. Does NOT remove or change "
                         "the existing output/<date>/ folders - both are "
                         "written when this flag is set. Default: off "
                         "(unchanged, folder-only behaviour).")

    # ---- testing -----------------------------------------------------------
    ap.add_argument("--fast-test", action="store_true",
                    help="TESTING ONLY: shrink NSGA-II (pop=20, gen=10, K=5) to "
                         "smoke-test the wiring in seconds instead of minutes. "
                         "Never use this for real results.")

    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    # Resolve duration: explicit > season default
    duration = args.duration if args.duration is not None \
        else _DURATION_DEFAULTS[args.season]

    cfg_overrides = {"POP_SIZE": 20, "N_GEN": 10, "K": 5} if args.fast_test else None

    # ---- print a clear run header ------------------------------------------
    print(f"Running simulation {start} .. {end}")
    print(f"Season      -> {args.season}   duration {duration} days")
    print(f"Forecast    -> {args.forecast or 'data/module_forecasts_and mar_aug_2025.csv (default)'}")
    print(f"Output      -> {args.output}")
    print(f"Scratch     -> {args.static_dir}")
    if cfg_overrides:
        print(f"FAST-TEST MODE (not real results): {cfg_overrides}")
    if args.consolidated_output:
        print("Consolidated output -> ON (also writing run-wide CSVs "
              "alongside the per-day folders)")

    run_simulation(
        start_date=start,
        end_date=end,
        output_root=args.output,
        static_dir=args.static_dir,
        forecast_path=args.forecast,
        season=args.season,
        duration=duration,
        cfg_overrides=cfg_overrides,
        consolidated_output=args.consolidated_output,
    )


if __name__ == "__main__":
    main()