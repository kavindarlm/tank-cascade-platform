"""
baseline_comparator.py
=======================
Runs the SAME day-by-day pipeline as run_simulation.py, but replaces
NSGA-II + TOPSIS with a simple, non-optimised proportional-allocation rule.
Produces output in the identical folder/file shape run_simulation.py
produces, so evaluate_season.py and generate_plots.py's --baseline-dir /
--baseline-eval-dir comparisons work on it unmodified.

WHY THIS EXISTS
-----------------
Every result the MO-MPC pipeline has produced (satisfaction %, f1-f4,
conveyance efficiency) describes what the optimised controller DID. None of
it has a comparison point - there is no way to say whether 71.9% mean
satisfaction is good, mediocre, or barely better than doing nothing clever
at all. This script answers that by running a deliberately simple,
explainable rule - one a real, uncoordinated irrigation scheme could follow
without any of this project's machinery - through the IDENTICAL simulator,
so the two are directly, fairly comparable.

THE BASELINE RULE
--------------------
For each optimised tank i, each day t of the planning horizon:

    R[i,t] = min( D[i,t],
                  R_max[i],
                  S_current[i] + cumulative_expected_inflow[i,0..t]
                     - cumulative_already_committed_release[i,0..t-1]
                     - S_min[i] )

In words: release up to today's demand, capped by the sluice, capped by what
the tank will physically have on hand given its starting storage plus
expected inflow so far, minus whatever this same rule has already committed
to release earlier in the horizon. This is a sequential, per-tank, no-
lookahead-beyond-arithmetic rule. It does not know about the network (no use
of alpha for planning), does not weigh competing objectives, does not adapt
to Module 3's risk forecast, and does not search - it is the "give each tank
what it asks for, up to what it can physically supply" rule.

WHAT IS SHARED WITH THE REAL PIPELINE  (this is what makes the comparison fair)
-----------------------------------------------------------------------------
  - state construction            : data_loader.build_state, UNCHANGED - same
                                     storage read, same demand slice, same
                                     scenario generation, same rainfall model
  - the cascade mass balance      : state_transition.simulate_cascade /
                                     simulate_all_scenarios, UNCHANGED
  - the objective functions       : objectives.evaluate_objectives_expected,
                                     UNCHANGED
  - the constraint checker        : constraints.check_all_constraints,
                                     UNCHANGED - used here to REPORT whether
                                     the baseline's plan is feasible, not to
                                     repair it if it isn't (the rule has no
                                     mechanism to fix an infeasible day, so
                                     infeasibility is recorded honestly)
  - the terminal-sink rule        : nsga2_optimizer.sink_release_rule,
                                     UNCHANGED - the sink is excluded from
                                     optimisation in BOTH arms, so using the
                                     identical rule for it isolates the
                                     comparison to the OPTIMISED tanks' policy
  - the Module 3 cross-check      : cross_validation.compare_with_module3,
                                     UNCHANGED

WHAT DIFFERS  (this is the entire point - only this differs)
-----------------------------------------------------------------
  - no NSGA-II search
  - no TOPSIS selection
  - no risk-adaptive Option A weighting
  - no baseline-relative C3/C1 reformulation AS A SEARCH MECHANISM (there is
    no search to reformulate a feasible region for - constraints are only
    checked and reported, never used to guide the rule)

USAGE
-----
    python baseline_comparator.py --start 2025-03-01 --end 2025-08-31 \\
        --season yala --duration 135 \\
        --forecast "data/module4_forecasts_mar_aug_2025.csv" \\
        --output output_baseline_2025

    python baseline_comparator.py --start 2026-03-01 --end 2026-07-26 \\
        --season yala --duration 135 \\
        --forecast "data/module4_forecasts_mar_jul_2026_tft.csv" \\
        --output output_baseline_2026

Resumable exactly like run_simulation.py: days already marked complete are
skipped, so an interrupted run can simply be re-invoked.

THEN evaluate and compare exactly like any other run
(--alpha-csv defaults to Module 2's live alpha_matrix.csv output):
    python evaluate_season.py output_baseline_2025 --season yala --duration 135 \\
        --demand-csv data/demand.csv --tank-params-csv data/tank_params.csv

    python evaluate_season.py output_25-03-08 --season yala --duration 135 \\
        --demand-csv data/demand.csv --tank-params-csv data/tank_params.csv \\
        --baseline-dir output_baseline_2025

    python generate_plots.py output_25-03-08/_evaluation \\
        --baseline-eval-dir output_baseline_2025/_evaluation
"""

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from datetime import date, timedelta

import numpy as np
import pandas as pd

from module4.config import Config
from module4 import data_loader as dl
from module4 import simulation_data as sd
from module4 import simulation_runner as sr
from module4 import state_transition as st
from module4 import objectives as obj
from module4 import constraints as ct
from module4 import cross_validation as cv
from module4.nsga2_optimizer import sink_release_rule
from module4.demand_generator import demand_window


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="first day, YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="last day, YYYY-MM-DD (inclusive)")
    ap.add_argument("--season", default="yala", choices=["yala", "maha"])
    ap.add_argument("--duration", type=int, required=True,
                    help="paddy duration class (105/135/90) - MUST match the "
                         "value used for the MO-MPC run being compared against")
    ap.add_argument("--forecast", required=True,
                    help="path to the Module 1+3 forecast CSV - MUST match "
                         "the file used for the MO-MPC run being compared "
                         "against, or the comparison is not apples-to-apples")
    ap.add_argument("--output", default="output_baseline",
                    help="output root (default: output_baseline)")
    ap.add_argument("--static-dir", default="_baseline_scratch",
                    help="scratch directory for staged per-day inputs")
    return ap.parse_args()


def _iter_dates(start_date, end_date):
    d = start_date
    while d <= end_date:
        yield d
        d += timedelta(days=1)


def _is_day_done(day_dir):
    p = os.path.join(day_dir, "day_summary.json")
    if not os.path.exists(p):
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f).get("status") == "complete"
    except (json.JSONDecodeError, OSError):
        return False


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# ======================================================================
# The baseline rule itself
# ======================================================================

def baseline_release_plan(state, cfg: Config):
    """
    Sequential proportional-allocation rule for the OPTIMISED tanks (see
    module docstring). Sink tanks are left at zero here and combined with
    sink_release_rule's output by the caller, exactly mirroring how the real
    optimiser separates the two.

    Returns
    -------
    (N, T) array, non-zero only at state["opt_idx"] rows.
    """
    N, T = int(state["N"]), int(state["T"])
    opt = np.asarray(state["opt_idx"], dtype=int)

    D = np.asarray(state["D"], dtype=float)
    R_max = np.asarray(state["R_max"], dtype=float)
    S_min = np.asarray(state["S_min"], dtype=float)
    S0 = np.asarray(state["S_current"], dtype=float)
    Q = np.asarray(state["Q_expected"], dtype=float)

    R = np.zeros((N, T))
    cum_inflow = np.zeros(N)
    cum_release = np.zeros(N)

    for t in range(T):
        cum_inflow[opt] += Q[opt, t]
        available = np.maximum(
            0.0, S0[opt] + cum_inflow[opt] - cum_release[opt] - S_min[opt])
        r = np.minimum(np.minimum(D[opt, t], R_max[opt]), available)
        R[opt, t] = r
        cum_release[opt] += r

    return R


# ======================================================================
# Per-day run, mirroring simulation_runner.run_one_day's structure
# ======================================================================

def run_one_baseline_day(ctx, sim_date, progress_log, season_start):
    day_offset = sr._day_offset(sim_date, season_start)
    day_dir = os.path.join(ctx.output_root, sim_date.isoformat())
    os.makedirs(day_dir, exist_ok=True)

    if _is_day_done(day_dir):
        progress_log.write(f"[{sim_date}] already complete - skipping\n")
        with open(os.path.join(day_dir, "day_summary.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    t0 = time.perf_counter()

    day_rows = sd.day_slice(ctx.forecast_df, sim_date, n_tanks=len(ctx.tank_names))
    day_rows = day_rows.loc[ctx.tank_names]
    S_current = sd.storage_pct_to_volume(day_rows["t+1"], ctx.S_max_by_name)
    module3_frame = sd.build_module3_frame(day_rows)
    dwin = demand_window(ctx.demand_m3, day_offset, ctx.cfg_season.T)

    sd.write_day_inputs(ctx.static_dir, ctx.tank_names, S_current, dwin,
                        module3_frame, ctx.cfg_season)

    day_cfg = Config(
        DATA_DIR=ctx.static_dir,
        OUTPUT_DIR=day_dir,
        RELOAD_STORAGE_EACH_DAY=True,
        SEASON=ctx.cfg_season.SEASON,
        SEASON_LENGTH=1,
        PADDY_DURATION_DAYS=ctx.cfg_season.PADDY_DURATION_DAYS,
        VERBOSE_LOGGING=False,
    )
    day_cfg.validate()

    static = dl.load_static_inputs(day_cfg)
    module3_df = dl.load_module3_risk(day_cfg)
    S_current_loaded = dl.load_current_storage(day_cfg)
    state = dl.build_state(static, module3_df, 0, day_cfg, S_current=S_current_loaded)

    # ---- THE ONLY THING THAT DIFFERS FROM THE REAL PIPELINE --------------
    R_sink = sink_release_rule(state, day_cfg)
    R_opt = baseline_release_plan(state, day_cfg)
    R_full = R_sink.copy()
    R_full[state["opt_idx"], :] = R_opt[state["opt_idx"], :]
    # ------------------------------------------------------------------------

    D_sim = state["D"] if getattr(day_cfg, "ENABLE_CROP_CONSUMPTION", True) else None
    sk = dict(
        storage_limited_evap=getattr(day_cfg, "STORAGE_LIMITED_EVAPORATION", True),
        upstream_iters=getattr(day_cfg, "UPSTREAM_FIXED_POINT_ITERS", 3),
    )

    sim_expected = st.simulate_cascade(
        state["S_current"], R_full, state["Q_expected"], state["alpha"],
        state["E"], state["S_min"], state["S_max"], D_sim, **sk)

    g = ct.check_all_constraints(R_full, sim_expected, state, day_cfg,
                                 rows=state["opt_idx"])
    feasible = bool(float(np.sum(np.maximum(g, 0.0))) <= 1e-6)

    sim_results = st.simulate_all_scenarios(
        state["S_current"], R_full, state["scenarios"], state["alpha"],
        state["E"], state["S_min"], state["S_max"], D_sim, **sk)
    f1, f2, f3, f4 = obj.evaluate_objectives_expected(
        sim_results, state["D"], rows=state["opt_idx"])

    total_release = float(sim_expected["R_effective"].sum())
    total_consumed = float(sim_expected["consumed"].sum())
    total_return_flow = float(sim_expected["return_flow"].sum())
    total_channel_loss = total_return_flow - float(
        (state["alpha"] @ sim_expected["return_flow"]).sum())

    demand_totals = state["D"].sum(axis=1)
    consumed_totals = sim_expected["consumed"].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sat_pct = np.where(demand_totals > 0,
                           consumed_totals / demand_totals * 100.0, np.nan)
    sat_valid = sat_pct[~np.isnan(sat_pct)]

    selected = {
        "rank": 1, "closeness": None,   # no TOPSIS ranking - not applicable
        "f1_shortage": float(f1), "f2_overflow": float(f2),
        "f3_equity": float(f3), "f4_water_loss": float(f4),
        "total_release": total_release, "total_consumed": total_consumed,
        "total_return_flow": total_return_flow,
        "total_channel_loss": total_channel_loss,
        "satisfaction_pct_mean": float(np.mean(sat_valid)) if len(sat_valid) else None,
        "satisfaction_pct_min": float(np.min(sat_valid)) if len(sat_valid) else None,
        "satisfaction_pct_max": float(np.max(sat_valid)) if len(sat_valid) else None,
    }

    comp_df = cv.compare_with_module3(state, R_full, module3_df, day_cfg)
    if comp_df is not None and len(comp_df) > 0:
        comp_df.to_csv(os.path.join(day_dir, "module3_crosscheck.csv"), index=False)
        from module4.cross_validation import divergence_summary
        crosscheck = divergence_summary(comp_df)
    else:
        crosscheck = None

    pd.DataFrame({
        "day": 1, "tank_id": state["tank_ids"],
        "release_m3": np.round(R_full[:, 0], 2),
    }).to_csv(os.path.join(day_dir, "mpc_decisions.csv"), index=False)

    pd.DataFrame([{
        "day": 1, "p_drought": state["p_drought"], "p_overflow": state["p_overflow"],
        "w_shortage": None, "w_overflow": None, "w_equity": None, "w_loss": None,
        "feasible": feasible, "c3_tier": "baseline_no_search",
        "reserve_deficit_m3": None,
    }]).to_csv(os.path.join(day_dir, "topsis_weights_log.csv"), index=False)

    elapsed = time.perf_counter() - t0
    summary = {
        "date": sim_date.isoformat(), "day_offset": day_offset,
        "status": "complete", "runtime_seconds": round(elapsed, 2),
        "feasible": feasible,
        "p_drought": state["p_drought"], "p_overflow": state["p_overflow"],
        "weights": None,
        "storage": {
            "total_today_m3": round(float(np.sum(state["S_current"])), 2),
            "per_tank_m3": {k: round(float(v), 2)
                           for k, v in zip(state["tank_ids"], state["S_current"])},
        },
        "selected": selected,
        "crosscheck": crosscheck,
        "output_files": {
            "decisions": os.path.join(day_dir, "mpc_decisions.csv"),
            "weights": os.path.join(day_dir, "topsis_weights_log.csv"),
            "crosscheck": (os.path.join(day_dir, "module3_crosscheck.csv")
                          if crosscheck else None),
        },
    }
    with open(os.path.join(day_dir, "day_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    progress_log.write(
        f"[{sim_date}] day {day_offset + 1}  feasible={feasible}  "
        f"{elapsed:5.1f}s  total_release={total_release:,.0f} m3  "
        f"sat_mean={selected['satisfaction_pct_mean']}\n")
    return summary


def _handle_day_error(ctx, sim_date, exc, progress_log, season_start):
    day_offset = sr._day_offset(sim_date, season_start)
    day_dir = os.path.join(ctx.output_root, sim_date.isoformat())
    os.makedirs(day_dir, exist_ok=True)
    with open(os.path.join(day_dir, "error.txt"), "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())
    summary = {
        "date": sim_date.isoformat(), "day_offset": day_offset,
        "status": "error", "error": str(exc),
        "feasible": False, "p_drought": None, "p_overflow": None,
        "weights": None, "storage": {"total_today_m3": None, "per_tank_m3": {}},
        "selected": None, "crosscheck": None, "output_files": {},
    }
    with open(os.path.join(day_dir, "day_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    progress_log.write(f"[{sim_date}] ERROR: {exc!r}\n")


# ======================================================================
# Orchestration - reuses SimulationContext for staging, unmodified
# ======================================================================

def run_baseline(start_date, end_date, output_root, static_dir,
                 forecast_path, season, duration):
    cfg_season = sr._default_season_config(season, duration)
    season_start = start_date

    os.makedirs(output_root, exist_ok=True)
    progress_path = os.path.join(output_root, "run_progress.log")
    progress_file = open(progress_path, "a", encoding="utf-8")
    tee = _Tee(sys.__stdout__, progress_file)

    tee.write(f"\n=== baseline_comparator starting  {start_date} .. {end_date}  "
             f"(completed days are skipped) ===\n")
    tee.write("Policy: proportional allocation (see module docstring) - "
             "NOT NSGA-II/TOPSIS.\n")

    ctx = sr.SimulationContext(cfg_season, static_dir, output_root,
                               forecast_path=forecast_path)
    tee.write(f"tanks: {len(ctx.tank_names)}   {season} demand: "
             f"season_days={ctx.demand_diag['season_days']}  "
             f"duration_class={ctx.demand_diag['duration_days']}\n")

    for sim_date in _iter_dates(start_date, end_date):
        try:
            run_one_baseline_day(ctx, sim_date, tee, season_start)
        except Exception as e:                       # noqa: BLE001
            _handle_day_error(ctx, sim_date, e, tee, season_start)

    tee.write(f"=== baseline_comparator finished at {end_date} ===\n")
    progress_file.close()


def main():
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print(f"Running BASELINE (proportional allocation) {start} .. {end}")
    print(f"Season      -> {args.season}   duration {args.duration} days")
    print(f"Forecast    -> {args.forecast}")
    print(f"Output      -> {args.output}")
    print("This is NOT the optimised controller - see module docstring for "
         "the rule being scored.\n")

    run_baseline(start, end, args.output, args.static_dir,
                args.forecast, args.season, args.duration)


if __name__ == "__main__":
    main()