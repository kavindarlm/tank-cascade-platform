"""
output.py
========
Presentation and export of MPC results: printed recommendations for the
operator, and CSV exports of daily decisions, applied TOPSIS weights, and the
Module 3 cross-check.
"""

import os
import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG


def print_recommendations(strategies, state, t_current, weights, cfg=DEFAULT_CONFIG):
    """
    Print the top strategies for one MPC step to the console.
    """
    print(f"\n{'='*66}")
    print(f"  MPC STEP - Day {t_current + 1}")
    print(f"  Risk (Module 3): P_drought={state['p_drought']:.2f}  "
          f"P_overflow={state['p_overflow']:.2f}")
    print(f"  Applied TOPSIS weights: shortage={weights[0]:.3f} "
          f"overflow={weights[1]:.3f} equity={weights[2]:.3f} loss={weights[3]:.3f}")
    print(f"{'='*66}")

    if not strategies:
        print("  No feasible strategy found this step.")
        return

    for s in strategies:
        print(f"\n  Rank {s['rank']}  (closeness={s['closeness']:.4f})")
        print(f"    f1 shortage   = {s['f1_shortage']:>12.2f} m3")
        print(f"    f2 overflow   = {s['f2_overflow']:>12.2f} m3")
        print(f"    f3 equity     = {s['f3_equity']:>12.4f}")
        print(f"    f4 water_loss = {s['f4_water_loss']:>12.2f} m3")
        print(f"    total release      = {s['total_release']:>12.2f} m3")
        print(f"    total consumed     = {s['total_consumed']:>12.2f} m3")
        print(f"    total return flow  = {s['total_return_flow']:>12.2f} m3")
        print(f"    total channel loss = {s['total_channel_loss']:>12.2f} m3")
        if s.get("tank_table"):
            headers = ["Tank", "Demand", "Release", "Consumed", "Return Flow", "Satisfaction %"]
            widths = [max(len(headers[i]), *(len(f"{row[i]}") for row in s["tank_table"]))
                      for i in range(len(headers))]
            print("    " + "  ".join(f"{headers[i]:<{widths[i]}}" if i == 0 else f"{headers[i]:>{widths[i]}}"
                                for i in range(len(headers))))
            print("    " + "  ".join("─" * widths[i] for i in range(len(headers))))
            for row in s["tank_table"]:
                print("    " + f"{row[0]:<{widths[0]}}  "
                      + f"{row[1]:>{widths[1]}.2f}  "
                      + f"{row[2]:>{widths[2]}.2f}  "
                      + f"{row[3]:>{widths[3]}.2f}  "
                      + f"{row[4]:>{widths[4]}.2f}  "
                      + f"{row[5]:>{widths[5]}.1f}%")
        if s.get("balance_table"):
            headers = ["Tank", "Storage Start", "Local Inflow", "Upstream Inflow", "Release", "Consumed", "Storage End"]
            widths = [max(len(headers[i]), *(len(f"{row[i]}") for row in s["balance_table"]))
                      for i in range(len(headers))]
            print("    " + "  ".join(f"{headers[i]:<{widths[i]}}" if i == 0 else f"{headers[i]:>{widths[i]}}"
                                for i in range(len(headers))))
            print("    " + "  ".join("─" * widths[i] for i in range(len(headers))))
            for row in s["balance_table"]:
                print("    " + f"{row[0]:<{widths[0]}}  "
                      + f"{row[1]:>{widths[1]}.2f}  "
                      + f"{row[2]:>{widths[2]}.2f}  "
                      + f"{row[3]:>{widths[3]}.2f}  "
                      + f"{row[4]:>{widths[4]}.2f}  "
                      + f"{row[5]:>{widths[5]}.2f}  "
                      + f"{row[6]:>{widths[6]}.2f}")
        sat = s["satisfaction_pct"]
        print(f"    satisfaction: mean={np.mean(sat):.1f}%  "
              f"min={np.min(sat):.1f}%  max={np.max(sat):.1f}%")


def save_decisions(all_decisions, cfg: Config = DEFAULT_CONFIG):
    """
    Save the day-0 release decisions across the whole season to CSV.

    all_decisions : list of dicts, each with 'day', 'tank_ids', 'R_today'
    """
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    rows = []
    for dec in all_decisions:
        for i, tid in enumerate(dec["tank_ids"]):
            rows.append({
                "day": dec["day"] + 1,
                "tank_id": tid,
                "release_m3": round(float(dec["R_today"][i]), 2),
            })
    df = pd.DataFrame(rows)
    path = os.path.join(cfg.OUTPUT_DIR, cfg.FILE_DECISIONS)
    df.to_csv(path, index=False)
    return path


def save_weights_log(weights_log, cfg: Config = DEFAULT_CONFIG):
    """
    Save the applied TOPSIS weights per MPC step.

    weights_log : list of dicts with 'day', 'p_drought', 'p_overflow', and w1..w4
    """
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(weights_log)
    path = os.path.join(cfg.OUTPUT_DIR, cfg.FILE_WEIGHTS_LOG)
    df.to_csv(path, index=False)
    return path


def save_crosscheck(comparison_dfs, cfg: Config = DEFAULT_CONFIG):
    """
    Concatenate and save the Module 3 cross-check tables across MPC steps.

    comparison_dfs : list of (day, DataFrame) tuples
    """
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    frames = []
    for day, df in comparison_dfs:
        if df is not None and len(df) > 0:
            df = df.copy()
            df.insert(0, "mpc_day", day + 1)
            frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    path = os.path.join(cfg.OUTPUT_DIR, cfg.FILE_VALIDATION)
    out.to_csv(path, index=False)
    return path
