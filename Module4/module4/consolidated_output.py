"""
consolidated_output.py
=======================
Optional, additive output mode for the day-by-day simulation runner.

By default (Config.CONSOLIDATED_OUTPUT = False) nothing here is called and
the existing output/<date>/ per-day folder pipeline (module4/output.py +
simulation_runner.py) behaves exactly as it always has.

When Config.CONSOLIDATED_OUTPUT is True, simulation_runner.run_one_day also
calls append_day() / append_day_summary() from here, which APPEND each
simulated day as rows onto a small set of run-wide CSVs sitting at the
output root, alongside (not instead of) the per-day folders:

  {output_root}/consolidated_mpc_decisions.csv       - one row per tank/day
  {output_root}/consolidated_topsis_weights_log.csv  - one row per day
  {output_root}/consolidated_module3_crosscheck.csv  - one row per tank/day
  {output_root}/consolidated_daily_summary.csv       - one row per day
                                                        (flattened day_summary)

Every row is tagged with an explicit "date" column, since - unlike the
per-day folders - the day is no longer implied by a directory name.
"""

import os
import pandas as pd


def _append_csv(df: pd.DataFrame, path: str):
    """Append df to path, writing the header only if the file is new."""
    if df is None or len(df) == 0:
        return
    file_exists = os.path.exists(path)
    df.to_csv(path, mode="a", header=not file_exists, index=False)


def append_day(output_root: str, sim_date, results, cfg) -> dict:
    """
    Append one simulated day's decisions / weights / crosscheck rows onto
    the consolidated CSVs under output_root.

    results : the same dict returned by mpc_loop.run_mpc() that
              output.save_all() already writes to the per-day folder -
              reused here as-is, just appended with a date column instead
              of overwritten into a fresh file.
    """
    os.makedirs(output_root, exist_ok=True)
    date_str = sim_date.isoformat()
    paths = {}

    decision_rows = []
    for dec in results["decisions"]:
        for i, tid in enumerate(dec["tank_ids"]):
            decision_rows.append({
                "date": date_str,
                "tank_id": tid,
                "release_m3": round(float(dec["R_today"][i]), 2),
            })
    if decision_rows:
        path = os.path.join(output_root, cfg.FILE_CONSOLIDATED_DECISIONS)
        _append_csv(pd.DataFrame(decision_rows), path)
        paths["decisions"] = path

    if results["weights_log"]:
        wl_rows = []
        for wl in results["weights_log"]:
            row = {"date": date_str}
            row.update(wl)
            wl_rows.append(row)
        path = os.path.join(output_root, cfg.FILE_CONSOLIDATED_WEIGHTS_LOG)
        _append_csv(pd.DataFrame(wl_rows), path)
        paths["weights"] = path

    frames = []
    for _, df in results["crosschecks"]:
        if df is not None and len(df) > 0:
            df = df.copy()
            df.insert(0, "date", date_str)
            frames.append(df)
    if frames:
        path = os.path.join(output_root, cfg.FILE_CONSOLIDATED_CROSSCHECK)
        _append_csv(pd.concat(frames, ignore_index=True), path)
        paths["crosscheck"] = path

    return paths


def append_day_summary(output_root: str, summary: dict, cfg) -> str:
    """
    Append one flattened row for this day's day_summary dict onto a single
    running CSV. Reuses simulation_runner._flatten_day_rows - the same
    flattening already used to build summary_NN/daily_metrics.csv - so the
    column layout matches what that periodic rollup already produces.
    """
    from .simulation_runner import _flatten_day_rows  # deferred: avoids a
    # circular import, since simulation_runner imports this module too.

    os.makedirs(output_root, exist_ok=True)
    df = _flatten_day_rows([summary])
    path = os.path.join(output_root, cfg.FILE_CONSOLIDATED_SUMMARY)
    _append_csv(df, path)
    return path
