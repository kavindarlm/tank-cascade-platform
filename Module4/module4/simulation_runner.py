"""
simulation_runner.py
====================
Day-by-day driver for the MO-MPC simulation.

For every simulated day this module:
  1. reads that day's row from the forecast CSV (Module 1 storage + Module 3
     risk), resolved at run-start from the --forecast argument,
  2. converts t+1 (% of capacity) to m3 for every tank -> today's storage,
  3. slices the ONCE-generated season demand at the matching day offset,
  4. reuses the frozen climatological rainfall history (see simulation_data.py
     docstring for why this is frozen rather than re-fetched daily),
  5. stages those inputs as CSVs and calls the EXISTING, UNMODIFIED
     mpc_loop.run_mpc(season_length=1) + output.save_all - i.e. runs the real
     OBSERVE / PREDICT / OPTIMISE / DECIDE / CROSS-CHECK / ACT pipeline
     exactly as main.py does for one day, with RELOAD_STORAGE_EACH_DAY=True
     so NO internal state propagation happens (tomorrow's storage always
     comes from tomorrow's dataset row, per the task's storage convention -
     t+1 is each day's actual measured storage, not a value to simulate
     forward from),
  6. saves every artifact - the full console log plus the three standard
     CSVs (mpc_decisions, topsis_weights_log, module3_crosscheck) plus a
     compact day_summary.json - into output/<date>/,
  7. every 15 days (and on the run's final day), rolls up a summary_NN/
     folder aggregating the days in that block.

The loop is resumable: a day whose day_summary.json already has
status == "complete" is skipped, so this script can be safely re-invoked
after an interruption (it will pick up on the first missing/errored day).

MULTI-RUN SUPPORT
  The constants SIM_START and SIM_END are kept for backward compatibility
  (external scripts that import them keep working). The simulation engine
  itself derives the day offset from the season_start passed at runtime,
  so a different year or season never reads the wrong demand slice.
"""

import contextlib
import json
import os
import sys
import time
import traceback
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import Config
from .demand_generator import demand_window
from .mpc_loop import run_mpc, save_all
from . import simulation_data as sd
from . import consolidated_output as co

# Kept for backward compatibility - do not remove.
SIM_START = date(2025, 3, 1)
SIM_END = date(2025, 8, 31)
SUMMARY_BLOCK_DAYS = 15


# ======================================================================
# helpers
# ======================================================================

class _Tee:
    """Duplicate writes to several streams (real stdout + a persistent log)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _default_yala_config() -> Config:
    """Original default - kept unchanged for backward compatibility."""
    cfg = Config()
    cfg.SEASON = "yala"
    cfg.PADDY_DURATION_DAYS = 105
    cfg.validate()
    return cfg


def _default_season_config(season: str, duration: int) -> Config:
    """
    Build a season Config for any season / duration combination.

    This is what the runner uses when --season and --duration are passed via
    the CLI. The underlying Config dataclass and all algorithm defaults are
    untouched - only SEASON and PADDY_DURATION_DAYS are set here.

    Parameters
    ----------
    season   : "yala" or "maha"
    duration : paddy duration class in days (105, 135, or 90)
    """
    cfg = Config()
    cfg.SEASON = season
    cfg.PADDY_DURATION_DAYS = int(duration)
    cfg.validate()
    return cfg


def _day_offset(sim_date: date, season_start: date) -> int:
    """
    Day index relative to the season start of THIS run.

    Using the actual season start (instead of the old hardcoded SIM_START)
    means demand slicing is correct for any year or season: day 0 is always
    the first day of the run regardless of the calendar year.

    The 15-day block numbering is also derived from this offset, so summary
    folders are numbered consistently from block 1 within each run.
    """
    return (sim_date - season_start).days


def _iter_dates(start_date: date, end_date: date):
    d = start_date
    while d <= end_date:
        yield d
        d += timedelta(days=1)


def _is_day_done(day_dir: str) -> bool:
    marker = os.path.join(day_dir, "day_summary.json")
    if not os.path.exists(marker):
        return False
    try:
        with open(marker, "r", encoding="utf-8") as f:
            js = json.load(f)
        return js.get("status") == "complete"
    except (json.JSONDecodeError, OSError):
        return False


# ======================================================================
# per-run context (loaded / generated ONCE)
# ======================================================================

class SimulationContext:
    """Everything loaded or generated once at the start of the run."""

    def __init__(self, cfg_season: Config, static_dir: str, output_root: str,
                 forecast_path: str = None, cfg_overrides: dict = None):
        self.cfg_season = cfg_season
        self.static_dir = static_dir
        self.output_root = output_root
        # Testing/validation hook ONLY: e.g. {"POP_SIZE": 20, "N_GEN": 10} to
        # smoke-test the day-by-day wiring in seconds instead of minutes.
        # None (the default) changes nothing - every NSGA-II/TOPSIS/objective/
        # constraint setting stays at the documented default for a real run.
        self.cfg_overrides = cfg_overrides or {}

        static = sd.load_static_reference(Config())
        self.tank_names = list(static["tank_ids"])
        self.S_max_by_name = pd.Series(static["S_max"], index=self.tank_names)

        sd.stage_static_files(static_dir)

        # Load the forecast dataset from the path supplied at runtime.
        # When None the original 2025 Yala file is used (backward compat).
        fpath = forecast_path or sd.FORECAST_DATASET_PATH
        self.forecast_df = sd.load_forecast_dataset(
            path=fpath, tank_names=self.tank_names)

        # Generate season demand using the generalised function so Yala and
        # Maha (and any other season) are handled through the same path.
        self.demand_m3, self.demand_diag = sd.generate_season_demand(
            self.tank_names, cfg_season)

        os.makedirs(output_root, exist_ok=True)


# ======================================================================
# one simulated day
# ======================================================================

def run_one_day(ctx: SimulationContext, sim_date: date, progress_log,
                season_start: date) -> dict:
    """Run one simulated day end-to-end and return its day_summary dict."""
    day_offset = _day_offset(sim_date, season_start)
    day_dir = os.path.join(ctx.output_root, sim_date.isoformat())
    os.makedirs(day_dir, exist_ok=True)

    if _is_day_done(day_dir):
        progress_log.write(f"[{sim_date}] already complete - skipping\n")
        with open(os.path.join(day_dir, "day_summary.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    t0 = time.perf_counter()

    day_rows = sd.day_slice(ctx.forecast_df, sim_date)
    day_rows = day_rows.loc[ctx.tank_names]                       # canonical order

    S_current = sd.storage_pct_to_volume(day_rows["t+1"], ctx.S_max_by_name)
    module3_frame = sd.build_module3_frame(day_rows)

    dwin = demand_window(ctx.demand_m3, day_offset, ctx.cfg_season.T)   # (N, T)

    sd.write_day_inputs(ctx.static_dir, ctx.tank_names, S_current,
                        dwin, module3_frame, ctx.cfg_season)

    day_cfg = Config(
        DATA_DIR=ctx.static_dir,
        OUTPUT_DIR=day_dir,
        RELOAD_STORAGE_EACH_DAY=True,
        SEASON=ctx.cfg_season.SEASON,
        SEASON_LENGTH=1,
        PADDY_DURATION_DAYS=ctx.cfg_season.PADDY_DURATION_DAYS,
        **ctx.cfg_overrides,
    )
    day_cfg.validate()

    # Tee stdout to BOTH the real terminal (so the usual banners / stage
    # rules / NSGA-II progress / tables still show live, exactly as running
    # main.py always did) AND this day's console_log.txt (so the daily
    # folder holds everything that was printed, per the task spec).
    log_path = os.path.join(day_dir, "console_log.txt")
    with open(log_path, "w", encoding="utf-8") as logf:
        with contextlib.redirect_stdout(_Tee(sys.__stdout__, logf)):
            print(f"Simulation day {day_offset + 1}  ({sim_date.isoformat()})")
            results = run_mpc(day_cfg, season_length=1, verbose=True)
            paths = save_all(results, day_cfg)

    elapsed = time.perf_counter() - t0

    summary = _build_day_summary(sim_date, day_offset, results, S_current,
                                 elapsed, paths)
    with open(os.path.join(day_dir, "day_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Additive, opt-in: on top of the per-day folder above, also append this
    # day as rows onto run-wide CSVs. Does not touch or replace anything
    # written above; only runs when the season Config asks for it.
    if ctx.cfg_season.CONSOLIDATED_OUTPUT:
        co.append_day(ctx.output_root, sim_date, results, ctx.cfg_season)
        co.append_day_summary(ctx.output_root, summary, ctx.cfg_season)

    # Total days in THIS run (from the context's forecast dataset date range).
    all_dates = sorted(ctx.forecast_df["forecast_date"].unique())
    total_days = len(all_dates)
    progress_log.write(
        f"[{sim_date}] day {day_offset + 1}/{total_days}  "
        f"feasible={summary['feasible']}  {elapsed:7.1f}s  "
        f"total_storage={summary['storage']['total_today_m3']:,.0f} m3\n"
    )
    return summary


def _build_day_summary(sim_date, day_offset, results, S_current, elapsed, paths) -> dict:
    wl = results["weights_log"][0] if results["weights_log"] else None
    decisions = results["decisions"][0] if results["decisions"] else None
    comp_df = results["crosschecks"][0][1] if results["crosschecks"] else None

    summary = {
        "date": sim_date.isoformat(),
        "day_offset": day_offset,
        "status": "complete",
        "runtime_seconds": round(elapsed, 2),
        "feasible": bool(wl["feasible"]) if wl else False,
        "p_drought": wl["p_drought"] if wl else None,
        "p_overflow": wl["p_overflow"] if wl else None,
        "weights": ({
            "shortage": wl["w_shortage"], "overflow": wl["w_overflow"],
            "equity": wl["w_equity"], "loss": wl["w_loss"],
        } if wl else None),
        "storage": {
            "total_today_m3": round(float(np.sum(S_current)), 2),
            "per_tank_m3": {k: round(float(v), 2) for k, v in S_current.items()},
        },
        "selected": None,
        "crosscheck": None,
        "output_files": dict(paths),
    }

    if decisions is not None:
        strat = decisions["strategy"]
        sat = strat["satisfaction_pct"]
        summary["selected"] = {
            "rank": strat["rank"],
            "closeness": strat["closeness"],
            "f1_shortage": strat["f1_shortage"],
            "f2_overflow": strat["f2_overflow"],
            "f3_equity": strat["f3_equity"],
            "f4_water_loss": strat["f4_water_loss"],
            "total_release": strat["total_release"],
            "total_consumed": strat["total_consumed"],
            "total_return_flow": strat["total_return_flow"],
            "total_channel_loss": strat["total_channel_loss"],
            "satisfaction_pct_mean": float(np.mean(sat)),
            "satisfaction_pct_min": float(np.min(sat)),
            "satisfaction_pct_max": float(np.max(sat)),
        }

    if comp_df is not None and len(comp_df) > 0:
        from .cross_validation import divergence_summary
        summary["crosscheck"] = divergence_summary(comp_df)

    return summary


def _handle_day_error(ctx: SimulationContext, sim_date: date, exc: Exception,
                      progress_log, season_start: date) -> dict:
    day_offset = _day_offset(sim_date, season_start)
    day_dir = os.path.join(ctx.output_root, sim_date.isoformat())
    os.makedirs(day_dir, exist_ok=True)
    with open(os.path.join(day_dir, "error.txt"), "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())

    summary = {
        "date": sim_date.isoformat(), "day_offset": day_offset,
        "status": "error", "error": str(exc),
        "feasible": False, "p_drought": None, "p_overflow": None,
        "weights": None,
        "storage": {"total_today_m3": None, "per_tank_m3": {}},
        "selected": None, "crosscheck": None, "output_files": {},
    }
    with open(os.path.join(day_dir, "day_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    progress_log.write(f"[{sim_date}] ERROR: {exc!r}  (see error.txt; day left "
                       f"incomplete so a re-run will retry it)\n")
    return summary


# ======================================================================
# 15-day summaries
# ======================================================================

def _flatten_day_rows(rows) -> pd.DataFrame:
    recs = []
    for r in rows:
        sel = r.get("selected") or {}
        cc = r.get("crosscheck") or {}
        recs.append({
            "date": r["date"],
            "day_offset": r["day_offset"],
            "status": r.get("status"),
            "feasible": r["feasible"],
            "p_drought": r["p_drought"],
            "p_overflow": r["p_overflow"],
            "total_storage_m3": r["storage"]["total_today_m3"],
            "f1_shortage": sel.get("f1_shortage"),
            "f2_overflow": sel.get("f2_overflow"),
            "f3_equity": sel.get("f3_equity"),
            "f4_water_loss": sel.get("f4_water_loss"),
            "total_release": sel.get("total_release"),
            "total_consumed": sel.get("total_consumed"),
            "total_return_flow": sel.get("total_return_flow"),
            "closeness": sel.get("closeness"),
            "satisfaction_pct_mean": sel.get("satisfaction_pct_mean"),
            "direction_agreement_rate": cc.get("direction_agreement_rate"),
            "mean_abs_ratio_diff": cc.get("mean_abs_ratio_diff"),
            "n_flagged_tanks": len(cc.get("flagged_tanks", []) or []),
        })
    return pd.DataFrame(recs)


def _stat(df: pd.DataFrame, col: str):
    s = df[col].dropna()
    if len(s) == 0:
        return None
    return {"mean": float(s.mean()), "sum": float(s.sum()),
           "min": float(s.min()), "max": float(s.max())}


def _aggregate_stats(rows, block_start: date, block_end: date) -> dict:
    df = _flatten_day_rows(rows)
    n = len(df)
    n_feasible = int(df["feasible"].sum())

    flagged_all = set()
    for r in rows:
        cc = r.get("crosscheck") or {}
        flagged_all.update(cc.get("flagged_tanks", []) or [])

    storage = df["total_storage_m3"].dropna()

    return {
        "period": {
            "start": block_start.isoformat(), "end": block_end.isoformat(),
            "n_days": n, "n_feasible": n_feasible,
            "feasibility_rate": (n_feasible / n) if n else None,
        },
        "storage": ({
            "start_m3": float(storage.iloc[0]),
            "end_m3": float(storage.iloc[-1]),
            "mean_m3": float(storage.mean()),
            "min_m3": float(storage.min()),
            "max_m3": float(storage.max()),
        } if len(storage) else None),
        "release": _stat(df, "total_release"),
        "consumed": _stat(df, "total_consumed"),
        "return_flow": _stat(df, "total_return_flow"),
        "shortage_f1": _stat(df, "f1_shortage"),
        "overflow_f2": _stat(df, "f2_overflow"),
        "equity_f3": _stat(df, "f3_equity"),
        "water_loss_f4": _stat(df, "f4_water_loss"),
        "closeness": _stat(df, "closeness"),
        "satisfaction_pct_mean": _stat(df, "satisfaction_pct_mean"),
        "risk": {
            "p_drought_mean": (float(df["p_drought"].mean())
                              if df["p_drought"].notna().any() else None),
            "p_overflow_mean": (float(df["p_overflow"].mean())
                               if df["p_overflow"].notna().any() else None),
        },
        "crosscheck": {
            "direction_agreement_rate": _stat(df, "direction_agreement_rate"),
            "mean_abs_ratio_diff": _stat(df, "mean_abs_ratio_diff"),
            "tanks_ever_flagged": sorted(flagged_all),
        },
    }


def _write_summary_chart(daily_df: pd.DataFrame, summary_dir: str, label: str,
                         block_start: date, block_end: date):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    x = pd.to_datetime(daily_df["date"])

    axes[0].plot(x, daily_df["total_storage_m3"] / 1e6, marker="o", color="#2563eb")
    axes[0].set_ylabel("total storage\n(million m3)")
    axes[0].set_title(f"{label}   {block_start.isoformat()} to {block_end.isoformat()}")

    axes[1].plot(x, daily_df["total_release"] / 1e3, marker="o", color="#059669", label="release")
    axes[1].plot(x, daily_df["f1_shortage"] / 1e3, marker="o", color="#dc2626", label="shortage (f1)")
    axes[1].plot(x, daily_df["f2_overflow"] / 1e3, marker="o", color="#d97706", label="overflow (f2)")
    axes[1].set_ylabel("thousand m3")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(x, daily_df["p_drought"], marker="o", color="#7c3aed", label="P(drought)")
    axes[2].plot(x, daily_df["p_overflow"], marker="o", color="#0891b2", label="P(overflow)")
    axes[2].set_ylabel("risk probability")
    axes[2].set_ylim(0, 1)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_xlabel("date")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(summary_dir, "trend_chart.png"), dpi=120)
    plt.close(fig)


def build_summary(output_root: str, block_index: int, block_start: date,
                  block_end: date, progress_log=None):
    """Aggregate every completed/errored day in [block_start, block_end] into
    output/summary_NN/. Safe to call repeatedly (always rebuilds in place)."""
    label = f"summary_{block_index:02d}"
    summary_dir = os.path.join(output_root, label)
    os.makedirs(summary_dir, exist_ok=True)

    rows = []
    for d in _iter_dates(block_start, block_end):
        js_path = os.path.join(output_root, d.isoformat(), "day_summary.json")
        if os.path.exists(js_path):
            with open(js_path, "r", encoding="utf-8") as f:
                rows.append(json.load(f))

    if not rows:
        return None

    daily_df = _flatten_day_rows(rows)
    daily_df.to_csv(os.path.join(summary_dir, "daily_metrics.csv"), index=False)

    stats = _aggregate_stats(rows, block_start, block_end)
    with open(os.path.join(summary_dir, "summary_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if daily_df["total_storage_m3"].notna().any():
        _write_summary_chart(daily_df, summary_dir, label, block_start, block_end)

    if progress_log:
        progress_log.write(f"[{label}] {block_start}..{block_end}  "
                           f"{len(rows)} day(s) aggregated -> {summary_dir}\n")
    return stats


# ======================================================================
# top-level orchestration
# ======================================================================

def run_simulation(start_date: date = SIM_START, end_date: date = SIM_END,
                   output_root: str = "output", static_dir: str = "_sim_scratch",
                   forecast_path: str = None, season: str = "yala",
                   duration: int = 105, cfg_season: Config = None,
                   cfg_overrides: dict = None, consolidated_output: bool = False):
    """
    Run the day-by-day simulation for [start_date, end_date] (inclusive).
    Safe to re-invoke: days already completed are skipped, so an interrupted
    run can simply be started again with the same arguments.

    Parameters
    ----------
    start_date    : first simulated day
    end_date      : last simulated day (inclusive)
    output_root   : root output directory (day folders + summary_NN folders)
    static_dir    : scratch directory for staged per-day input CSVs
    forecast_path : path to the Module 1+3 forecast CSV. Defaults to the
                    original 2025 Yala file. Supply a different path for a
                    different year or season without changing any code.
    season        : "yala" or "maha" - sets cfg.SEASON
    duration      : paddy duration class in days (105 / 135 / 90)
    cfg_season    : fully-built Config to use instead of building one from
                    season + duration. Useful for programmatic callers.
    cfg_overrides : testing hook ONLY (e.g. {"POP_SIZE": 20, "N_GEN": 10}
                    to smoke-test in seconds). Leave None for real runs.
    consolidated_output : if True, sets cfg_season.CONSOLIDATED_OUTPUT so the
                    day-by-day loop ALSO appends every day as rows onto the
                    run-wide CSVs in module4/consolidated_output.py, on top
                    of (not instead of) the usual output/<date>/ folders.
                    Ignored (does nothing) if a pre-built cfg_season already
                    sets the flag itself. Default False - identical to the
                    original behaviour.
    """
    if cfg_season is None:
        cfg_season = _default_season_config(season, duration)
    if consolidated_output:
        cfg_season.CONSOLIDATED_OUTPUT = True

    # season_start is the anchor for day offsets. Using start_date ensures
    # that day 0 is always the first day of THIS run, regardless of the
    # calendar year or season. This replaces the old hardcoded SIM_START.
    season_start = start_date

    os.makedirs(output_root, exist_ok=True)
    progress_path = os.path.join(output_root, "run_progress.log")
    progress_file = open(progress_path, "a", encoding="utf-8")
    tee = _Tee(sys.__stdout__, progress_file)

    tee.write(f"\n=== run_simulation starting  {start_date} .. {end_date}  "
             f"(completed days are skipped) ===\n")
    if cfg_overrides:
        tee.write(f"*** cfg_overrides active (TESTING ONLY): {cfg_overrides} ***\n")

    ctx = SimulationContext(cfg_season, static_dir, output_root,
                            forecast_path=forecast_path,
                            cfg_overrides=cfg_overrides)
    tee.write(f"tanks: {len(ctx.tank_names)}   "
             f"{season} demand: season_days={ctx.demand_diag['season_days']}  "
             f"duration_class={ctx.demand_diag['duration_days']}\n")

    last_seen = None
    try:
        for sim_date in _iter_dates(start_date, end_date):
            last_seen = sim_date
            try:
                run_one_day(ctx, sim_date, tee, season_start)
            except Exception as e:                      # noqa: BLE001 - isolate one bad day
                _handle_day_error(ctx, sim_date, e, tee, season_start)

            offset = _day_offset(sim_date, season_start)
            is_block_end = (offset % SUMMARY_BLOCK_DAYS == SUMMARY_BLOCK_DAYS - 1)
            is_last_day = (sim_date == end_date)
            if is_block_end or is_last_day:
                block_index = offset // SUMMARY_BLOCK_DAYS + 1
                block_start = sim_date - timedelta(days=offset % SUMMARY_BLOCK_DAYS)
                build_summary(output_root, block_index, block_start, sim_date, tee)
    finally:
        tee.write(f"=== run_simulation stopped/finished at {last_seen} ===\n")
        progress_file.close()