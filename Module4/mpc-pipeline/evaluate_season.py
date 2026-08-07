"""
evaluate_season.py
===================
Comprehensive, reusable evaluation of a completed (or partial) MO-MPC season
run. Works on ANY output folder produced by run_simulation.py - the 2025 run,
the 2026 run, a future Maha run, or a baseline comparator run - without
touching any algorithm file.

USAGE
-----
    python evaluate_season.py output_25-03-08
    python evaluate_season.py output_2026 --season yala --duration 135 ^
        --demand-csv data/demand_2026.csv
        (--alpha-csv defaults to Module 2's live alpha_matrix.csv output;
        pass it explicitly to pin an older/different alpha matrix)

WHY EACH METRIC EXISTS
-----------------------
Every metric below traces back to a specific question raised (and in several
cases, a specific bug found) during development of this pipeline. Comments at
each section explain the "why", not just the "what", so this script remains
readable as documentation of the evaluation methodology, not just code.

WHAT IS AND ISN'T RECONSTRUCTABLE FROM output_dir ALONE
----------------------------------------------------------
day_summary.json persists SEASON-LEVEL totals (release, consumed, return_flow,
channel_loss, f1..f4, satisfaction, per-tank STORAGE) and the cross-check.
mpc_decisions.csv persists PER-TANK RELEASE only - no demand, no consumed.

Per-tank demand does not exist in output_dir at all; it is regenerated here
via the same generate_season_demand() call the runner used. This requires the
ORIGINAL demand.csv (or an explicit --demand-csv), because data/ is a scratch
area that gets overwritten between runs - this is exactly the failure mode
that corrupted the 2026 fast-test earlier, and this script refuses to guess:
if the supplied demand file's shape doesn't match the run being evaluated, it
warns loudly and skips per-tank demand/satisfaction rather than silently
reporting numbers computed against the wrong season.

Per-tank channel-loss capture is a STRUCTURAL property of network_alpha.csv
(it does not vary by day), so it is computed once from an explicit
--alpha-csv, with the same "don't silently trust the live data/ folder"
discipline.

OUTPUTS (written to --out, default: <output_dir>/_evaluation/)
-----------------------------------------------------------------
    season_metrics.csv       one row per day - every day_summary + weights field
    monthly_breakdown.csv    month-level aggregation, with a demand-
                              extrapolation exposure flag per month
    c3_decomposition.csv     per-tank held/free classification + season
                              storage change, and the aggregate split
    per_tank_summary.csv     release/demand/satisfaction/zero-days per tank,
                              with automatic anomaly flagging
    alpha_channel_loss.csv   per-tank network capture fraction, corrected
                              channel-loss statistic excluding structural sinks
    crosscheck_summary.csv   per-day and per-tank Module 3 agreement
    summary_report.json      every headline number in one nested dict, for
                              reuse in dissertation tables
    summary_report.txt       human-readable console-style summary
"""

import argparse
import glob
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

# Module 2's live GNN output, resolved from this file's own location (not
# cwd) so the default works regardless of where this script is invoked from.
# NOT alpha_matrix_v2.csv. Mirrors module4/config.py's FILE_NETWORK_ALPHA.
_MODULE2_ALPHA_MATRIX_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "Module2", "development-history", "outputs", "alpha_matrix.csv"))


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir",
                    help="the run's output folder, e.g. output_25-03-08")
    ap.add_argument("--out", default=None,
                    help="where to write evaluation artefacts "
                         "(default: <output_dir>/_evaluation)")
    ap.add_argument("--season", default="yala", choices=["yala", "maha"],
                    help="season, for demand regeneration (default: yala)")
    ap.add_argument("--duration", type=int, default=None,
                    help="paddy duration class for demand regeneration "
                         "(105/135/90). If omitted, per-tank demand/"
                         "satisfaction metrics are skipped.")
    ap.add_argument("--demand-csv", default=None,
                    help="path to the demand.csv ACTUALLY USED for this run. "
                         "If omitted, per-tank demand metrics are skipped "
                         "rather than risking a mismatch with whatever is "
                         "currently in data/.")
    ap.add_argument("--alpha-csv", default=_MODULE2_ALPHA_MATRIX_PATH,
                    help="path to the alpha matrix for the channel-loss "
                         "diagnostic (default: Module 2's live "
                         "alpha_matrix.csv output)")
    ap.add_argument("--tank-params-csv", default="data/tank_params.csv",
                    help="path to tank_params.csv, for S_min/S_max/C3 floor "
                         "(default: data/tank_params.csv)")
    ap.add_argument("--baseline-dir", default=None,
                    help="OPTIONAL second output folder (e.g. a proportional-"
                         "allocation baseline run) to compare against. If "
                         "omitted, the comparison section is skipped.")
    ap.add_argument("--real-release-csv", default=None,
                    help="OPTIONAL CSV of observed real-world release "
                         "(e.g. Nachchaduwa) with columns 'date,release_m3', "
                         "for external validation. Skipped if not supplied.")
    ap.add_argument("--c3-buffer-fraction", type=float, default=0.05,
                    help="the C3 seasonal buffer fraction used by the run "
                         "(default 0.05, matching config.py's default)")
    ap.add_argument("--sink-tanks",
                    default="Nachchaduwa_Wewa,Kudaittikattiya_Wewa,"
                            "Settikulama_Wewa,Galwaduwawa_Wewa",
                    help="comma-separated tank_ids that are terminal-sink "
                         "reservoirs for THIS run (default matches config.py's "
                         "TERMINAL_SINK_TANKS). These tanks are excluded from "
                         "C1-C5 and f1-f4 and are released by the fixed "
                         "min(D, R_max, avail) rule, not by the optimiser - "
                         "see sink_release_rule() in nsga2_optimizer.py. "
                         "Pass '' for a run with no sink tanks configured.")
    return ap.parse_args()


def warn(msg):
    print(f"  [!] {msg}", file=sys.stderr)


# ======================================================================
# Loading
# ======================================================================

def load_day_summaries(output_dir):
    """Every completed or errored day, in date order."""
    paths = sorted(glob.glob(os.path.join(output_dir, "*", "day_summary.json")))
    days = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            days.append(json.load(f))
    if not days:
        raise SystemExit(f"No day_summary.json files found under {output_dir}/*/")
    days.sort(key=lambda d: d["date"])
    return days


def load_weights_log(output_dir):
    """
    Concatenate topsis_weights_log.csv across all day folders. Each file's
    internal 'day' column always reads 1 (it is the FOLDER's own day-zero
    index, reset per folder for SEASON_LENGTH=1 runs) - so the folder name,
    not the 'day' column, is the reliable date key. This is exactly the trap
    that produced the "day 1" repeated-12-times table earlier in development.
    """
    rows = []
    for p in sorted(glob.glob(os.path.join(output_dir, "*", "topsis_weights_log.csv"))):
        date = os.path.basename(os.path.dirname(p))
        df = pd.read_csv(p)
        if len(df) == 0:
            continue
        row = df.iloc[0].to_dict()
        row["date"] = date
        rows.append(row)
    return pd.DataFrame(rows)


def load_decisions(output_dir):
    """Concatenate mpc_decisions.csv (release_m3 per tank) across all days."""
    frames = []
    for p in sorted(glob.glob(os.path.join(output_dir, "*", "mpc_decisions.csv"))):
        date = os.path.basename(os.path.dirname(p))
        df = pd.read_csv(p)
        df["date"] = date
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["date", "tank_id", "release_m3"])
    return pd.concat(frames, ignore_index=True)


def load_crosscheck(output_dir):
    """Concatenate module3_crosscheck.csv across all days."""
    frames = []
    for p in sorted(glob.glob(os.path.join(output_dir, "*", "module3_crosscheck.csv"))):
        date = os.path.basename(os.path.dirname(p))
        df = pd.read_csv(p)
        df["date"] = date
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_tank_names(days):
    """
    Tank list, in a stable order, read from the first day that has per-tank
    storage - avoids depending on tank_coordinates.csv being present or
    unchanged since the run.
    """
    for d in days:
        pt = (d.get("storage") or {}).get("per_tank_m3") or {}
        if pt:
            return sorted(pt.keys())
    raise SystemExit("No day has per-tank storage data - cannot proceed.")


# ======================================================================
# Demand reconstruction (guarded - see module docstring)
# ======================================================================

def try_reconstruct_demand(tank_names, season_days_needed, args):
    """
    Regenerate the full-season per-tank demand array using the SAME
    generate_season_demand() the runner calls, from an explicitly supplied
    demand.csv (never the live data/ folder by default - see module
    docstring for why).

    Returns (demand_m3 (N, season_days), season_days) or (None, None) if
    reconstruction was not requested or failed a sanity check.
    """
    if args.demand_csv is None or args.duration is None:
        warn("--demand-csv / --duration not supplied - skipping per-tank "
             "demand and satisfaction metrics (season-level totals from "
             "day_summary.json are still fully computed).")
        return None, None

    try:
        sys.path.insert(0, os.getcwd())
        from module4.config import Config
        import module4.simulation_data as sd
    except ImportError as e:
        warn(f"could not import module4 to regenerate demand ({e}) - "
             f"skipping per-tank demand metrics. Run this script from the "
             f"folder containing module4/.")
        return None, None

    cfg = Config()
    cfg.SEASON = args.season
    cfg.PADDY_DURATION_DAYS = args.duration
    try:
        cfg.validate()
    except Exception as e:
        warn(f"config validation failed ({e}) - skipping demand metrics.")
        return None, None

    old_path = sd.SOURCE_DEMAND_CSV
    sd.SOURCE_DEMAND_CSV = args.demand_csv
    try:
        demand_m3, diag = sd.generate_season_demand(tank_names, cfg)
    except Exception as e:
        warn(f"demand regeneration failed ({e}) - skipping demand metrics.")
        return None, None
    finally:
        sd.SOURCE_DEMAND_CSV = old_path

    season_days = demand_m3.shape[1]
    # Sanity check: warn (but do not silently fabricate) if the reconstructed
    # season is shorter than what the run actually covers - this is exactly
    # the 160-vs-184-day gap that caused the demand-extrapolation tail.
    if season_days < season_days_needed:
        warn(f"reconstructed demand covers {season_days} days but the run "
             f"spans {season_days_needed} days. Days beyond {season_days} "
             f"were run against the CLAMPED final-day demand rate (see the "
             f"'demand_extrapolated' flag in monthly_breakdown.csv).")
    return demand_m3, season_days


def demand_on_day(demand_m3, day_offset):
    """Same clamp rule as demand_generator.demand_window: hold the last
    column once the offset runs past the reconstructed array."""
    if demand_m3 is None:
        return None
    idx = min(day_offset, demand_m3.shape[1] - 1)
    return demand_m3[:, idx]


# ======================================================================
# A. Feasibility metrics
# ======================================================================

def compute_feasibility_metrics(days, weights_df):
    n = len(days)
    n_complete = sum(1 for d in days if d.get("status") == "complete")
    n_error = sum(1 for d in days if d.get("status") == "error")
    n_feasible = sum(1 for d in days if d.get("feasible"))

    tier_counts = {"hard": 0, "relaxed": 0, "none": 0, "unknown": 0}
    if "c3_tier" in weights_df.columns:
        for v in weights_df["c3_tier"]:
            tier_counts[v if v in tier_counts else "unknown"] += 1
    else:
        tier_counts["unknown"] = n

    return {
        "total_days": n,
        "days_complete": n_complete,
        "days_error": n_error,
        "days_feasible": n_feasible,
        "feasibility_rate_pct": round(n_feasible / n * 100, 2) if n else None,
        "c3_tier_hard": tier_counts["hard"],
        "c3_tier_relaxed": tier_counts["relaxed"],
        "c3_tier_none_infeasible": tier_counts["none"],
        "c3_tier_unknown": tier_counts["unknown"],
    }


# ======================================================================
# B. Water delivery metrics (season-level, from day_summary.json)
# ======================================================================

def compute_delivery_metrics(days):
    sel = [d.get("selected") for d in days if d.get("selected")]
    if not sel:
        warn("no day has a 'selected' plan - delivery metrics unavailable.")
        return {}

    total_release = sum(s["total_release"] for s in sel)
    total_consumed = sum(s["total_consumed"] for s in sel)
    total_return = sum(s["total_return_flow"] for s in sel)
    total_loss = sum(s["total_channel_loss"] for s in sel)
    total_f1 = sum(s["f1_shortage"] for s in sel)
    total_f2 = sum(s["f2_overflow"] for s in sel)
    mean_f3 = float(np.mean([s["f3_equity"] for s in sel]))
    total_f4 = sum(s["f4_water_loss"] for s in sel)
    sat_mean = [s["satisfaction_pct_mean"] for s in sel]

    return {
        "total_release_m3": round(total_release, 1),
        "total_consumed_m3": round(total_consumed, 1),
        "total_return_flow_m3": round(total_return, 1),
        "total_channel_loss_m3_asreported": round(total_loss, 1),
        "total_f1_shortage_m3": round(total_f1, 1),
        "total_f2_overflow_m3": round(total_f2, 1),
        "mean_f3_equity": round(mean_f3, 4),
        "total_f4_water_loss_m3": round(total_f4, 1),
        "mean_daily_release_m3": round(total_release / len(sel), 1),
        "season_mean_satisfaction_pct": round(float(np.mean(sat_mean)), 2),
        "season_min_satisfaction_pct": round(float(np.min(sat_mean)), 2),
        "season_min_satisfaction_date":
            days[int(np.argmin([(d.get("selected") or {}).get("satisfaction_pct_mean", 100)
                               for d in days]))]["date"],
        "season_max_satisfaction_pct": round(float(np.max(sat_mean)), 2),
    }


def compute_monthly_breakdown(days, demand_m3, season_days_reconstructed):
    """
    Month-level release/consumed/satisfaction, WITH an explicit
    demand_extrapolated flag. See module docstring: any day whose offset
    exceeds the reconstructed demand array is running on the CLAMPED final
    rate, not real CWR-table data. Silently aggregating that into a monthly
    total without flagging it is exactly the mistake that made the original
    "August cliff" look like a bug instead of a data-coverage limit.
    """
    rows = []
    for d in days:
        sel = d.get("selected") or {}
        month = d["date"][:7]
        offset = d.get("day_offset")
        extrapolated = (demand_m3 is not None and season_days_reconstructed
                        and offset is not None
                        and offset > season_days_reconstructed - 1)
        rows.append({
            "date": d["date"], "month": month,
            "release": sel.get("total_release"),
            "consumed": sel.get("total_consumed"),
            "f1_shortage": sel.get("f1_shortage"),
            "satisfaction_pct": sel.get("satisfaction_pct_mean"),
            "demand_extrapolated": bool(extrapolated),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    agg = df.groupby("month").agg(
        n_days=("date", "count"),
        n_days_extrapolated=("demand_extrapolated", "sum"),
        release=("release", "sum"),
        consumed=("consumed", "sum"),
        shortage=("f1_shortage", "sum"),
        mean_satisfaction_pct=("satisfaction_pct", "mean"),
    ).reset_index()
    agg["pct_days_extrapolated"] = (agg["n_days_extrapolated"] / agg["n_days"] * 100).round(1)
    return agg


# ======================================================================
# C. Water balance metrics
# ======================================================================

def compute_water_balance(days):
    """
    Mass balance closure: release == consumed + return_flow, EXACTLY (per
    equation 4.7 - consumed and return_flow are the only two things effective
    release splits into). channel_loss is NOT a third term - it is defined as
        channel_loss = return_flow - sum(alpha @ return_flow)
    i.e. the portion of return_flow the network fails to deliver anywhere.
    Adding it a second time double-counts, which is the exact bug that made
    an earlier version of this check report a false mass-balance failure.
    """
    sel = [d.get("selected") for d in days if d.get("selected")]
    if not sel:
        return {}

    errors = []
    for s in sel:
        reconstructed = s["total_consumed"] + s["total_return_flow"]
        errors.append(s["total_release"] - reconstructed)
    errors = np.array(errors)

    total_release = sum(s["total_release"] for s in sel)
    total_consumed = sum(s["total_consumed"] for s in sel)
    total_return = sum(s["total_return_flow"] for s in sel)

    return {
        "mass_balance_max_abs_error_m3": round(float(np.max(np.abs(errors))), 6),
        "mass_balance_mean_abs_error_m3": round(float(np.mean(np.abs(errors))), 6),
        "mass_balance_closes_exactly": bool(np.max(np.abs(errors)) < 1.0),
        "conveyance_efficiency_pct":
            round(total_consumed / total_release * 100, 2) if total_release else None,
        "return_flow_share_of_release_pct":
            round(total_return / total_release * 100, 2) if total_release else None,
    }


def compute_channel_loss(alpha_csv_path, decisions_df, tank_names):
    """
    Per-tank network capture fraction (column sum of alpha, per the
    documented convention row=destination, column=source - see
    EXPLANATION.md section 7.4). A tank with column sum == 0 is a STRUCTURAL
    network terminus (nothing captures its outflow) and must be EXCLUDED from
    any "mean channel loss" statistic, or a true cascade terminus (which is
    not a loss at all) inflates the average. This is exactly the correction
    applied after Nachchaduwa and Kuda_Kanumulla both showed 0.000000.
    """
    if not os.path.exists(alpha_csv_path):
        warn(f"alpha matrix not found at {alpha_csv_path} - skipping "
             f"channel-loss diagnostic.")
        return pd.DataFrame(), {}

    alpha = pd.read_csv(alpha_csv_path, index_col=0)
    # Module 2's raw output uses space-separated tank names (and inconsistent
    # spacing, e.g. "Settikulama  Wewa"); Module 4 uses underscores
    # everywhere else (tank_ids, mpc_decisions.csv, etc.). Normalize so the
    # label-based lookups below actually match instead of silently treating
    # every tank as "missing".
    rename = {c: re.sub(r"\s+", "_", c.strip()) for c in alpha.columns}
    alpha = alpha.rename(columns=rename, index=rename)
    missing = set(tank_names) - set(alpha.columns)
    if missing:
        warn(f"{len(missing)} tank(s) in the run are not columns of "
             f"{alpha_csv_path} - alpha matrix may not match this run's "
             f"tank set. Channel-loss numbers may be unreliable.")

    col_sums = alpha.sum(axis=0)
    df = pd.DataFrame({
        "tank_id": col_sums.index,
        "network_capture_fraction": col_sums.values,
        "structural_terminus": col_sums.values < 1e-9,
    }).sort_values("network_capture_fraction")

    # Weight the "corrected" mean by each tank's actual release volume where
    # available, so a rarely-releasing tank does not distort the average as
    # much as one that moves a lot of water.
    rel_by_tank = decisions_df.groupby("tank_id")["release_m3"].sum() \
        if len(decisions_df) else pd.Series(dtype=float)
    df["season_total_release_m3"] = df["tank_id"].map(rel_by_tank).fillna(0.0)

    non_terminus = df[~df["structural_terminus"]]
    weights = non_terminus["season_total_release_m3"]
    if weights.sum() > 0:
        weighted_mean_capture = float(
            (non_terminus["network_capture_fraction"] * weights).sum() / weights.sum())
    else:
        weighted_mean_capture = float(non_terminus["network_capture_fraction"].mean())

    summary = {
        "n_tanks": len(df),
        "n_structural_termini": int(df["structural_terminus"].sum()),
        "structural_termini": list(df[df["structural_terminus"]]["tank_id"]),
        "raw_mean_channel_loss_pct_INCLUDING_termini":
            round((1 - df["network_capture_fraction"].mean()) * 100, 2),
        "corrected_mean_channel_loss_pct_EXCLUDING_termini":
            round((1 - weighted_mean_capture) * 100, 2),
        "lowest_capture_non_terminus_tanks":
            non_terminus.nsmallest(3, "network_capture_fraction")[
                ["tank_id", "network_capture_fraction"]].to_dict("records"),
    }
    return df, summary


# ======================================================================
# D. Storage / C3 decomposition
# ======================================================================

def compute_c3_decomposition(days, tank_params_csv, c3_buffer_fraction, tank_names,
                             sink_tank_names=None):
    """
    Splits the season's storage change into tanks the seasonal reserve floor
    holds back versus tanks free to draw down. This is the single most
    important diagnostic used throughout development: an aggregate storage
    RISE can be pure reserve recovery in a handful of depleted tanks while
    every other tank correctly depletes - reporting only the aggregate hides
    that entirely.

    IMPORTANT: "held" is classified over the WHOLE SEASON (fraction of days
    each tank's storage sits below its floor), NOT just day 1. A day-1-only
    snapshot misses any tank that starts above its floor and crosses below it
    later - which is exactly what happened with Galkulama_Wewa: it read
    'free' on day 1, then spent the rest of the season oscillating at or
    below its floor (mean storage 27,332 m3 against a floor of 27,866 m3),
    and every earlier version of this function missed it entirely. A tank is
    now classified 'held' if it spends more than half its days below floor,
    which is what the controller actually experienced across the season.

    SINK TANKS (added): C3 is a decision-variable constraint inside NSGA-II
    (constraints.py / nsga2_optimizer.py). Tanks in TERMINAL_SINK_TANKS are
    never a decision variable - their release comes from sink_release_rule()
    - so C3 literally cannot "hold" them, whatever their storage does. Before
    this parameter existed, a sink tank sitting below its own S_min+buffer
    floor was indistinguishable from a real C3-held tank in this function's
    output, which mislabelled Settikulama_Wewa as C3-held in the 2026 run.
    held_by_c3_season is now forced False for sink tanks; the raw
    pct_days_below_floor is kept (it's still a true physical observation),
    and is_sink_tank marks which rows to interpret that way.
    """
    sink_tank_names = set(sink_tank_names or [])
    if not os.path.exists(tank_params_csv):
        warn(f"tank_params not found at {tank_params_csv} - skipping the "
             f"C3 held/free decomposition.")
        return pd.DataFrame(), {}

    params = pd.read_csv(tank_params_csv).set_index("tank_id")
    complete_days = [d for d in days if d.get("status") == "complete"]
    if not complete_days:
        return pd.DataFrame(), {}
    first_storage = (complete_days[0].get("storage") or {}).get("per_tank_m3") or {}
    last_storage = (complete_days[-1].get("storage") or {}).get("per_tank_m3") or {}

    # Season-wide below-floor day counts, per tank.
    below_floor_days = {tid: 0 for tid in tank_names}
    below_smin_days = {tid: 0 for tid in tank_names}
    n_obs = {tid: 0 for tid in tank_names}
    for d in complete_days:
        pt = (d.get("storage") or {}).get("per_tank_m3") or {}
        for tid, s in pt.items():
            if tid not in params.index:
                continue
            s_min = params.loc[tid, "S_min"]
            s_max = params.loc[tid, "S_max"]
            floor = s_min + c3_buffer_fraction * s_max
            n_obs[tid] = n_obs.get(tid, 0) + 1
            if s < floor:
                below_floor_days[tid] = below_floor_days.get(tid, 0) + 1
            if s < s_min:
                below_smin_days[tid] = below_smin_days.get(tid, 0) + 1

    rows = []
    for tid in tank_names:
        if tid not in params.index or tid not in first_storage or tid not in last_storage:
            continue
        s_min = params.loc[tid, "S_min"]
        s_max = params.loc[tid, "S_max"]
        floor = s_min + c3_buffer_fraction * s_max
        s0 = first_storage[tid]
        s1 = last_storage[tid]
        n = max(n_obs.get(tid, 0), 1)
        pct_below_floor = below_floor_days.get(tid, 0) / n * 100
        is_sink = tid in sink_tank_names
        rows.append({
            "tank_id": tid, "S_min": s_min, "S_max": s_max, "C3_floor": floor,
            "storage_day1": s0, "storage_last_day": s1,
            "change_m3": s1 - s0,
            "days_below_floor": below_floor_days.get(tid, 0),
            "days_below_S_min": below_smin_days.get(tid, 0),
            "n_days_observed": n,
            "pct_days_below_floor": round(pct_below_floor, 1),
            "is_sink_tank": is_sink,
            # Season-wide classification: "held" if below floor on the
            # MAJORITY of days, not just day 1. Forced False for sink tanks -
            # C3 is not evaluated on them at all inside the optimiser, so
            # they cannot be "C3-held" regardless of where their storage
            # sits; use is_sink_tank + pct_days_below_floor to describe them
            # instead (their level is governed by sink_release_rule's
            # min(D, R_max, avail), not by the reserve constraint).
            "held_by_c3_season": bool(pct_below_floor > 50.0) and not is_sink,
            # Kept for backward compatibility with anything reading the old
            # column name; now clearly documented as day-1-only.
            "held_by_c3_on_day1": bool(s0 < floor) and not is_sink,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}

    held = df[df["held_by_c3_season"]]
    free = df[~df["held_by_c3_season"] & ~df["is_sink_tank"]]
    sinks = df[df["is_sink_tank"]]
    held_d1 = df[df["held_by_c3_on_day1"]]
    summary = {
        "n_tanks_held_season_majority": len(held),
        "n_tanks_free_season_majority": len(free),
        "n_sink_tanks_excluded_from_c3": len(sinks),
        "n_tanks_held_day1_only": len(held_d1),
        "held_tanks_net_change_m3": round(float(held["change_m3"].sum()), 1),
        "free_tanks_net_change_m3": round(float(free["change_m3"].sum()), 1),
        "sink_tanks_net_change_m3": round(float(sinks["change_m3"].sum()), 1)
            if len(sinks) else 0.0,
        "total_net_change_m3": round(float(df["change_m3"].sum()), 1),
        "held_tank_names_season": list(held["tank_id"]),
        "sink_tanks_below_floor_majority_of_season": list(
            sinks[sinks["pct_days_below_floor"] > 50.0]["tank_id"]),
        "note": "held/free now classified by MAJORITY of season days below "
               "floor, not day 1 alone, AND excludes sink tanks (C3 does not "
               "apply to TERMINAL_SINK_TANKS - see docstring). Sink tanks "
               "sitting below their floor are listed separately in "
               "sink_tanks_below_floor_majority_of_season; that reflects "
               "sink_release_rule()'s availability, not the C3 constraint.",
    }
    return df, summary


def compute_storage_trajectory(days, sink_tank_names=None):
    """Village (non-sink) vs sink storage over time, for plotting."""
    sink_tank_names = set(sink_tank_names or [])
    rows = []
    for d in days:
        pt = (d.get("storage") or {}).get("per_tank_m3") or {}
        if not pt:
            continue
        village = sum(v for k, v in pt.items() if k not in sink_tank_names)
        sink = sum(v for k, v in pt.items() if k in sink_tank_names)
        rows.append({"date": d["date"], "village_storage_m3": village,
                     "sink_storage_m3": sink, "total_storage_m3": village + sink})
    return pd.DataFrame(rows)


# ======================================================================
# E. Cross-check summary
# ======================================================================

def compute_crosscheck_summary(days, output_dir=None, tank_params_csv=None,
                               anchor_multiple=2.0):
    """
    As before, PLUS a corrected direction-agreement figure that excludes
    tank-days whose ANCHOR storage was small relative to S_min.

    WHY: the cross-check ratio is S(t)/S(anchor). When anchor storage is a
    small absolute number (a tank near dead storage on its anchor day),
    ordinary absolute inflow variation produces enormous, unstable ratios -
    confirmed on this project's own data: for Settikulama_Wewa, anchor
    storage vs. day-7 ratio magnitude correlated at r = -0.88, and the
    single smallest anchor (254 m3, ~1.4% of S_min) produced the LARGEST
    ratio in the entire season (11.49x). This is a property of the ratio-
    normalisation METHOD, not a disagreement between Module 3's forecast and
    Module 4's simulated physics - it says nothing about whether the two
    modules actually agree on that tank-day.

    The raw (uncorrected) figures are still returned, unchanged, so nothing
    already reported is silently altered - the corrected figure is
    additional, not a replacement, and the exclusion count is reported so
    the correction itself is auditable.
    """
    cc = [d.get("crosscheck") for d in days if d.get("crosscheck")]
    if not cc:
        warn("no day has cross-check data - skipping.")
        return {}, pd.DataFrame()

    da = [c["direction_agreement_rate"] for c in cc if c.get("direction_agreement_rate") is not None]
    diff = [c["mean_abs_ratio_diff"] for c in cc if c.get("mean_abs_ratio_diff") is not None]

    from collections import Counter
    flag_counts = Counter()
    for c in cc:
        for t in (c.get("flagged_tanks") or []):
            flag_counts[t] += 1

    per_day = pd.DataFrame([{
        "date": d["date"],
        "direction_agreement_pct": (d["crosscheck"]["direction_agreement_rate"] * 100
                                    if d["crosscheck"].get("direction_agreement_rate") is not None
                                    else None),
        "mean_abs_ratio_diff": d["crosscheck"].get("mean_abs_ratio_diff"),
        "n_flagged": len(d["crosscheck"].get("flagged_tanks") or []),
    } for d in days if d.get("crosscheck")])

    summary = {
        "mean_direction_agreement_pct_RAW": round(float(np.mean(da)) * 100, 2) if da else None,
        "last_direction_agreement_pct": round(da[-1] * 100, 2) if da else None,
        "mean_abs_ratio_diff_RAW": round(float(np.mean(diff)), 4) if diff else None,
        "tanks_flagged_every_day": [t for t, n in flag_counts.items() if n == len(cc)],
        "tanks_flagged_ge_75pct_days":
            [t for t, n in flag_counts.items() if n >= 0.75 * len(cc)],
    }

    # ---- corrected figure, excluding small-anchor tank-days -------------
    if output_dir is None or tank_params_csv is None or not os.path.exists(tank_params_csv):
        summary["anchor_correction"] = "not computed (output_dir/tank_params_csv not supplied)"
        return summary, per_day

    try:
        params = pd.read_csv(tank_params_csv).set_index("tank_id")
    except Exception as e:
        warn(f"could not load tank_params for the anchor correction ({e})")
        summary["anchor_correction"] = "not computed (failed to load tank_params)"
        return summary, per_day

    frames = []
    for p in sorted(glob.glob(os.path.join(output_dir, "*", "module3_crosscheck.csv"))):
        date = os.path.basename(os.path.dirname(p))
        df = pd.read_csv(p)
        df["date"] = date
        frames.append(df)
    if not frames:
        summary["anchor_correction"] = "not computed (no crosscheck CSVs found)"
        return summary, per_day
    cc_raw = pd.concat(frames, ignore_index=True)

    # Anchor storage per (date, tank) = that day's storage snapshot (day=1
    # in the crosscheck table is the anchor day itself, ratio == 1.0000).
    anchor_rows = cc_raw[cc_raw["day"] == 1][["date", "tank_id"]].copy()
    storage_by_date = {}
    for d in days:
        pt = (d.get("storage") or {}).get("per_tank_m3") or {}
        storage_by_date[d["date"]] = pt
    anchor_rows["anchor_storage"] = anchor_rows.apply(
        lambda r: storage_by_date.get(r["date"], {}).get(r["tank_id"]), axis=1)
    anchor_rows["s_min"] = anchor_rows["tank_id"].map(
        lambda t: params.loc[t, "S_min"] if t in params.index else np.nan)
    anchor_rows["reliable"] = (anchor_rows["anchor_storage"]
                              >= anchor_multiple * anchor_rows["s_min"])

    cc_last = cc_raw[cc_raw["day"] == cc_raw.groupby(["date", "tank_id"])["day"].transform("max")]
    cc_last = cc_last.merge(anchor_rows[["date", "tank_id", "reliable", "anchor_storage"]],
                            on=["date", "tank_id"], how="left")

    n_total = len(cc_last)
    n_unreliable = int((~cc_last["reliable"].fillna(False)).sum())

    reliable_rows = cc_last[cc_last["reliable"] == True]
    if len(reliable_rows):
        same_dir = (np.sign(reliable_rows["m4_volume_ratio"] - 1.0)
                   == np.sign(reliable_rows["m3_volume_ratio"] - 1.0))
        corrected_rate = float(same_dir.mean()) * 100
    else:
        corrected_rate = None

    summary["anchor_correction"] = {
        "threshold": f"anchor_storage >= {anchor_multiple:.1f} * S_min",
        "n_tank_days_total": n_total,
        "n_tank_days_excluded_small_anchor": n_unreliable,
        "pct_excluded": round(n_unreliable / n_total * 100, 1) if n_total else None,
        "direction_agreement_pct_CORRECTED": round(corrected_rate, 2) if corrected_rate is not None else None,
        "interpretation": "The RAW figure includes tank-days where the ratio "
                          "denominator (anchor storage) was small enough to "
                          "make the ratio numerically unstable, which is a "
                          "property of the comparison METHOD, not a "
                          "module-vs-module disagreement. CORRECTED excludes "
                          "those tank-days.",
    }

    # ---- magnitude figure, recomputed FRESH from the CSVs on every call ---
    # Unlike direction_agreement_pct_RAW above (cached from day_summary.json
    # at original simulation time), this is derived from module3_crosscheck.csv
    # every time the function runs - so a magnitude-only correction, such as
    # recompute_crosscheck.py's MODULE3_STORAGE_BASIS fix, becomes visible in
    # the standard report without a separate script's printed output being the
    # only record of it. Direction agreement is NOT duplicated here: it is
    # mathematically invariant to any monotonic transform of the ratio (a
    # sign-preserving transform, such as raising a positive ratio to a power,
    # can never change which side of 1 it falls on), so re-deriving it from
    # the same CSVs would always exactly reproduce mean_direction_agreement_
    # pct_RAW and add no information.
    summary["mean_abs_ratio_diff_CURRENT_CSV"] = round(
        float(cc_raw["abs_ratio_difference"].mean()), 4)
    summary["_note_on_magnitude_vs_direction"] = (
        "mean_abs_ratio_diff_RAW above is cached from day_summary.json at "
        "original simulation time and will NOT reflect a later magnitude-only "
        "fix (e.g. recompute_crosscheck.py). mean_abs_ratio_diff_CURRENT_CSV "
        "is recomputed fresh from module3_crosscheck.csv every run and always "
        "reflects the current files on disk. Direction-agreement figures are "
        "unaffected either way, by mathematical necessity - see the docstring."
    )

    return summary, per_day


# ======================================================================
# G. Per-tank equity + anomaly detection
# ======================================================================

def compute_per_tank_summary(decisions_df, demand_m3, tank_names, days,
                             c3_df, alpha_df, sink_tank_names=None):
    """
    Per-tank release/demand/satisfaction/zero-day statistics, with AUTOMATIC
    anomaly flagging. This directly operationalises how the Galkulama problem
    was found in this project: a tank releasing zero on a large majority of
    days, that is NOT explained by (a) being C3-held, (b) having near-zero
    demand, or (c) being a terminal-sink tank released by the fixed
    min(D, R_max, avail) rule instead of the optimiser, is flagged for manual
    investigation rather than silently passing.

    (c) was added after Kudaittikattiya_Wewa and Galwaduwawa_Wewa's ~0%
    2026 satisfaction showed up as neither C3-held nor low-demand - i.e. as
    UNEXPLAINED - when in fact both are TERMINAL_SINK_TANKS: Kudaittikattiya's
    R_max (86.4 m3/day) alone caps it near 7% of its ~1285 m3/day demand, and
    Galwaduwawa's near-zero release means sink_release_rule()'s `avail`
    (storage above S_min) was itself near zero for most of the season - a
    real, separate thing to verify against that tank's actual S trajectory,
    but not an optimiser bug, since the optimiser never touches either tank.
    """
    sink_tank_names = set(sink_tank_names or [])
    if decisions_df.empty:
        return pd.DataFrame()

    rel = decisions_df.groupby("tank_id")["release_m3"].agg(
        total_release="sum", mean_release="mean",
        zero_days=lambda x: int((x <= 1e-6).sum()),
        active_days=lambda x: int((x > 1e-6).sum()),
        n_days="count",
    ).reset_index()
    rel["zero_day_pct"] = (rel["zero_days"] / rel["n_days"] * 100).round(1)

    # Per-tank total demand across the run, if reconstructed.
    if demand_m3 is not None:
        offsets_by_date = {d["date"]: d.get("day_offset") for d in days}
        dec = decisions_df.copy()
        dec["day_offset"] = dec["date"].map(offsets_by_date)
        idx_map = {t: i for i, t in enumerate(tank_names)}
        dec["demand_m3"] = dec.apply(
            lambda r: demand_on_day(demand_m3, int(r["day_offset"]))[idx_map[r["tank_id"]]]
            if pd.notna(r["day_offset"]) and r["tank_id"] in idx_map else np.nan,
            axis=1,
        )
        dec["consumed_m3"] = np.minimum(dec["release_m3"], dec["demand_m3"])
        dem_summary = dec.groupby("tank_id").agg(
            total_demand="demand_m3", total_consumed="consumed_m3"
        ) if False else dec.groupby("tank_id").agg(
            total_demand=("demand_m3", "sum"),
            total_consumed=("consumed_m3", "sum"),
            mean_demand=("demand_m3", "mean"),
        ).reset_index()
        rel = rel.merge(dem_summary, on="tank_id", how="left")
        rel["satisfaction_pct"] = np.where(
            rel["total_demand"] > 0,
            rel["total_consumed"] / rel["total_demand"] * 100, np.nan)
    else:
        rel["total_demand"] = np.nan
        rel["mean_demand"] = np.nan
        rel["satisfaction_pct"] = np.nan

    if not c3_df.empty:
        rel = rel.merge(
            c3_df[["tank_id", "held_by_c3_season", "pct_days_below_floor",
                  "days_below_S_min"]],
            on="tank_id", how="left")
    else:
        rel["held_by_c3_season"] = np.nan
        rel["pct_days_below_floor"] = np.nan
        rel["days_below_S_min"] = np.nan

    if not alpha_df.empty:
        rel = rel.merge(
            alpha_df[["tank_id", "network_capture_fraction", "structural_terminus"]],
            on="tank_id", how="left")

    rel["is_sink_tank"] = rel["tank_id"].isin(sink_tank_names)

    # --- anomaly flag -------------------------------------------------
    def flag(row):
        reasons = []
        if row["zero_day_pct"] > 50:
            explained = False
            # (c) Terminal-sink tank: never a decision variable, released by
            # sink_release_rule()'s min(D, R_max, avail) instead of the
            # optimiser. Checked first since, when true, it fully accounts
            # for the release pattern regardless of what C3/demand say.
            if row.get("is_sink_tank"):
                explained = True
                reasons.append("expected: terminal-sink tank "
                              "(rule-based release, not optimised - check "
                              "R_max and storage-above-S_min for this tank "
                              "rather than the optimiser)")
            # SEASON-WIDE floor check, not day-1-only - see
            # compute_c3_decomposition's docstring for why day-1-only missed
            # Galkulama_Wewa, which crossed below its floor after day 1 and
            # was held for most of the season despite starting above it.
            # (held_by_c3_season is already forced False for sink tanks in
            # compute_c3_decomposition, so this branch cannot double-count
            # a sink tank as "C3-held".)
            if row.get("held_by_c3_season") is True:
                explained = True
                pct = row.get("pct_days_below_floor")
                reasons.append(f"mostly explained: C3-held "
                              f"({pct:.0f}% of days below floor)"
                              if pd.notna(pct) else "mostly explained: C3-held")
            if pd.notna(row.get("mean_demand")) and row["mean_demand"] < 1.0:
                explained = True
                reasons.append("mostly explained: near-zero demand")
            if not explained:
                reasons.append("UNEXPLAINED - high zero-release rate with "
                              "neither sink status, season-wide C3 status, "
                              "nor low demand accounting for it")
        return "; ".join(reasons) if reasons else ""

    rel["anomaly_flag"] = rel.apply(flag, axis=1)
    return rel.sort_values("zero_day_pct", ascending=False)


# ======================================================================
# Optional: baseline comparison
# ======================================================================

def compute_baseline_comparison(days, baseline_dir):
    if baseline_dir is None:
        return None
    if not os.path.isdir(baseline_dir):
        warn(f"--baseline-dir {baseline_dir} not found - skipping comparison.")
        return None
    base_days = load_day_summaries(baseline_dir)

    def agg(dlist):
        sel = [d.get("selected") for d in dlist if d.get("selected")]
        if not sel:
            return None
        return {
            "n_days": len(sel),
            "total_release_m3": sum(s["total_release"] for s in sel),
            "total_f1_shortage_m3": sum(s["f1_shortage"] for s in sel),
            "total_f2_overflow_m3": sum(s["f2_overflow"] for s in sel),
            "mean_f3_equity": float(np.mean([s["f3_equity"] for s in sel])),
            "total_f4_water_loss_m3": sum(s["f4_water_loss"] for s in sel),
            "mean_satisfaction_pct": float(np.mean(
                [s["satisfaction_pct_mean"] for s in sel])),
        }

    return {"proposed": agg(days), "baseline": agg(base_days)}


# ======================================================================
# Optional: external validation against real observed release
# ======================================================================

def compute_external_validation(decisions_df, real_release_csv, tank_id_filter=None):
    """
    Compares simulated release against an observed real-world record, if
    supplied. Not run by default - no real observed release file has yet
    been confirmed to exist for this project; this is here so that if/when
    one is obtained, the comparison is a one-flag addition rather than new
    code.
    """
    if real_release_csv is None:
        return None
    if not os.path.exists(real_release_csv):
        warn(f"--real-release-csv {real_release_csv} not found - skipping "
             f"external validation.")
        return None

    real = pd.read_csv(real_release_csv, parse_dates=["date"])
    real["date"] = real["date"].dt.strftime("%Y-%m-%d")

    sim = decisions_df
    if tank_id_filter:
        sim = sim[sim["tank_id"] == tank_id_filter]
    sim_daily = sim.groupby("date")["release_m3"].sum().reset_index()

    sim_daily = sim_daily.rename(columns={"release_m3": "release_m3_sim"})
    real = real.rename(columns={"release_m3": "release_m3_real"})
    merged = sim_daily.merge(real, on="date")
    if merged.empty:
        warn("no overlapping dates between simulated and real release records.")
        return None

    err = merged["release_m3_sim"] - merged["release_m3_real"]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    corr = float(np.corrcoef(merged["release_m3_sim"], merged["release_m3_real"])[0, 1])

    return {
        "n_days_compared": len(merged),
        "rmse_m3": round(rmse, 1),
        "mean_bias_m3": round(bias, 1),
        "pearson_correlation": round(corr, 4),
        "mean_sim_m3": round(float(merged["release_m3_sim"].mean()), 1),
        "mean_real_m3": round(float(merged["release_m3_real"].mean()), 1),
    }


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    out_dir = args.out or os.path.join(args.output_dir, "_evaluation")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {args.output_dir} ...")
    days = load_day_summaries(args.output_dir)
    weights_df = load_weights_log(args.output_dir)
    decisions_df = load_decisions(args.output_dir)
    crosscheck_raw = load_crosscheck(args.output_dir)
    tank_names = get_tank_names(days)
    print(f"  {len(days)} day(s), {len(tank_names)} tanks")

    season_days_needed = max((d.get("day_offset") or 0) for d in days) + 1
    demand_m3, season_days_reconstructed = try_reconstruct_demand(
        tank_names, season_days_needed, args)

    print("Computing metrics ...")
    # OPERATIONAL sink list (config.py's TERMINAL_SINK_TANKS for this run) -
    # these tanks are excluded from C1-C5/f1-f4 and released by
    # sink_release_rule(), NOT by the optimiser. This is a DIFFERENT concept
    # from alpha_df's "structural_terminus" (a topological property of
    # network_alpha.csv - which tanks have zero downstream conveyance - used
    # only by compute_channel_loss). The two sets overlap on Nachchaduwa_Wewa
    # but are not the same; using the wrong one to build "village storage"
    # silently mixes rule-based tanks into what's presented as the
    # controller's own trajectory. See --sink-tanks help text.
    sink_names = [t.strip() for t in args.sink_tanks.split(",") if t.strip()]
    feasibility = compute_feasibility_metrics(days, weights_df)
    delivery = compute_delivery_metrics(days)
    monthly = compute_monthly_breakdown(days, demand_m3, season_days_reconstructed)
    water_balance = compute_water_balance(days)
    alpha_df, channel_loss = compute_channel_loss(args.alpha_csv, decisions_df, tank_names)
    c3_df, c3_summary = compute_c3_decomposition(
        days, args.tank_params_csv, args.c3_buffer_fraction, tank_names,
        sink_tank_names=sink_names)
    storage_traj = compute_storage_trajectory(days, sink_tank_names=sink_names)
    crosscheck_summary, crosscheck_daily = compute_crosscheck_summary(
        days, output_dir=args.output_dir, tank_params_csv=args.tank_params_csv)
    per_tank = compute_per_tank_summary(decisions_df, demand_m3, tank_names,
                                        days, c3_df, alpha_df,
                                        sink_tank_names=sink_names)
    baseline_cmp = compute_baseline_comparison(days, args.baseline_dir)
    external_val = compute_external_validation(decisions_df, args.real_release_csv)

    # ---- write CSV artefacts ------------------------------------------
    season_metrics = pd.DataFrame([{
        "date": d["date"], "day_offset": d.get("day_offset"),
        "status": d.get("status"), "feasible": d.get("feasible"),
        "p_drought": d.get("p_drought"), "p_overflow": d.get("p_overflow"),
        "total_storage_m3": (d.get("storage") or {}).get("total_today_m3"),
        **{f"sel_{k}": v for k, v in (d.get("selected") or {}).items()
          if not isinstance(v, (list, dict))},
        **{f"cc_{k}": v for k, v in (d.get("crosscheck") or {}).items()
          if not isinstance(v, (list, dict))},
    } for d in days])
    season_metrics = season_metrics.merge(
        weights_df[["date"] + [c for c in weights_df.columns
                               if c not in ("date",) and c != "day"]],
        on="date", how="left", suffixes=("", "_w"))

    season_metrics.to_csv(os.path.join(out_dir, "season_metrics.csv"), index=False)
    monthly.to_csv(os.path.join(out_dir, "monthly_breakdown.csv"), index=False)
    c3_df.to_csv(os.path.join(out_dir, "c3_decomposition.csv"), index=False)
    per_tank.to_csv(os.path.join(out_dir, "per_tank_summary.csv"), index=False)
    alpha_df.to_csv(os.path.join(out_dir, "alpha_channel_loss.csv"), index=False)
    storage_traj.to_csv(os.path.join(out_dir, "storage_trajectory.csv"), index=False)
    crosscheck_daily.to_csv(os.path.join(out_dir, "crosscheck_daily.csv"), index=False)

    # ---- summary report -------------------------------------------------
    report = {
        "output_dir": args.output_dir,
        "date_range": [days[0]["date"], days[-1]["date"]],
        "feasibility": feasibility,
        "delivery": delivery,
        "water_balance": water_balance,
        "channel_loss": channel_loss,
        "c3_decomposition": c3_summary,
        "crosscheck": crosscheck_summary,
        "anomalous_tanks": per_tank[per_tank["anomaly_flag"].str.contains(
            "UNEXPLAINED", na=False)]["tank_id"].tolist() if not per_tank.empty else [],
        "baseline_comparison": baseline_cmp,
        "external_validation": external_val,
    }
    with open(os.path.join(out_dir, "summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # ---- human-readable text report --------------------------------------
    lines = []
    lines.append("=" * 78)
    lines.append(f"SEASON EVALUATION REPORT  -  {args.output_dir}")
    lines.append(f"  {days[0]['date']} .. {days[-1]['date']}   ({len(days)} days)")
    lines.append("=" * 78)

    lines.append("\nA. FEASIBILITY")
    for k, v in feasibility.items():
        lines.append(f"  {k:<32}: {v}")

    lines.append("\nB. WATER DELIVERY")
    for k, v in delivery.items():
        lines.append(f"  {k:<32}: {v}")

    lines.append("\nC. WATER BALANCE")
    for k, v in water_balance.items():
        lines.append(f"  {k:<32}: {v}")
    lines.append("\n   CHANNEL LOSS (corrected for structural network termini)")
    for k, v in channel_loss.items():
        lines.append(f"  {k:<32}: {v}")

    lines.append("\nD. C3 RESERVE DECOMPOSITION")
    for k, v in c3_summary.items():
        lines.append(f"  {k:<32}: {v}")

    lines.append("\nE. MODULE 3 CROSS-CHECK")
    for k, v in crosscheck_summary.items():
        lines.append(f"  {k:<32}: {v}")

    lines.append("\nG. PER-TANK ANOMALIES")
    anomalous = report["anomalous_tanks"]
    if anomalous:
        lines.append(f"  {len(anomalous)} tank(s) flagged UNEXPLAINED - see "
                     f"per_tank_summary.csv:")
        for t in anomalous:
            lines.append(f"    - {t}")
    else:
        lines.append("  none - every high-zero-release tank is explained by "
                     "C3 status or low demand.")

    if baseline_cmp:
        lines.append("\nF. BASELINE COMPARISON")
        lines.append(f"  proposed : {baseline_cmp['proposed']}")
        lines.append(f"  baseline : {baseline_cmp['baseline']}")
    else:
        lines.append("\nF. BASELINE COMPARISON: not requested (--baseline-dir not supplied)")

    if external_val:
        lines.append("\nEXTERNAL VALIDATION (vs real observed release)")
        for k, v in external_val.items():
            lines.append(f"  {k:<32}: {v}")
    else:
        lines.append("\nEXTERNAL VALIDATION: not requested "
                     "(--real-release-csv not supplied)")

    lines.append("\n" + "=" * 78)
    lines.append(f"Full artefacts written to: {out_dir}/")
    lines.append("=" * 78)

    text = "\n".join(lines)
    with open(os.path.join(out_dir, "summary_report.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    print()
    print(text)


if __name__ == "__main__":
    main()