"""
data_loader.py
=============
Load and validate all Module 4 inputs, assemble the SystemState dict passed to
every downstream module.

Inputs (all in DATA_DIR, see config for file names):
  tank_params.csv       - S_min, S_max, R_max, E_day*, catchment_area_km2  (static)
  tank_storage.csv      - storage_m3 today (from Module 1)                  (dynamic)
  demand.csv            - D_i(t) full season (from FAO-56 generator)        (static)
  network_alpha.csv     - alpha matrix, read directly from Module 2's live
                          output (Module2/development-history/outputs/
                          alpha_matrix.csv) - see config.FILE_NETWORK_ALPHA  (static)
  rainfall_history.csv  - daily rainfall history per tank (Open-Meteo)      (dynamic)
  module3_risk.csv      - Module 3 forecast + risk probs (Option A weights) (dynamic)
  mahaweli_schedule.csv - R_sched (ONLY if C6 enabled)                      (optional)

This module DOES NOT fabricate data. If a required file is missing it raises a
clear error telling you which file and which columns are expected. Data wiring is
the next phase after the code is in place.
"""

import os
import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from .demand_generator import demand_window
from .scenario_generator import generate_scenarios


def _require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required input file not found: {path}\n"
            f"  -> This file must be provided before running. See the data section "
            f"of the explanation .md for its exact column format."
        )
    return path


def load_static_inputs(cfg: Config = DEFAULT_CONFIG):
    """
    Load the inputs that do NOT change across MPC steps: tank parameters,
    demand (full season), network alpha, and rainfall history.

    Returns
    -------
    static : dict with keys N, S_min, S_max, R_max, E, catchment_km2, alpha,
             demand_full, rainfall_history, tank_ids
    """
    dd = cfg.DATA_DIR

    # ---- Tank parameters ----
    params = pd.read_csv(_require(os.path.join(dd, cfg.FILE_TANK_PARAMS)),
                         index_col="tank_id")
    N = len(params)
    S_min = params["S_min"].to_numpy(dtype=float)
    S_max = params["S_max"].to_numpy(dtype=float)
    R_max = params["R_max"].to_numpy(dtype=float)

    e_cols = [f"E_day{d}" for d in range(cfg.T)]
    missing_e = [c for c in e_cols if c not in params.columns]
    if missing_e:
        raise ValueError(
            f"tank_params.csv is missing evaporation columns {missing_e}. "
            f"Expected E_day0 .. E_day{cfg.T-1}."
        )
    E = params[e_cols].to_numpy(dtype=float)                    # (N, T)

    if "catchment_area_km2" not in params.columns:
        raise ValueError("tank_params.csv must include a 'catchment_area_km2' column "
                         "(needed to convert rainfall to inflow).")
    catchment_km2 = params["catchment_area_km2"].to_numpy(dtype=float)  # (N,)

    tank_ids = list(params.index)

    # ---- Guard: incomplete tank parameters --------------------------------
    # A single NaN in S_max / S_min / R_max / E silently propagates through the
    # mass balance and makes EVERY objective NaN, which destroys NSGA-II's
    # ability to rank candidates. Fail loudly here instead.
    incomplete = {}
    for label, arr in (("S_min", S_min), ("S_max", S_max),
                       ("R_max", R_max), ("evaporation", E)):
        bad = np.isnan(arr).any(axis=1) if arr.ndim > 1 else np.isnan(arr)
        if bad.any():
            incomplete[label] = [tank_ids[i] for i in np.where(bad)[0]]

    if incomplete:
        names = sorted({t for v in incomplete.values() for t in v})
        raise ValueError(
            "tank_params.csv has missing values for "
            f"{len(names)} tank(s):\n"
            + "\n".join(f"    - {t}" for t in names)
            + "\n\n  Affected columns: " + ", ".join(sorted(incomplete))
            + "\n\n  A single NaN makes every objective NaN and breaks the optimiser.\n"
              "  Fix by EITHER:\n"
              "    (a) sourcing water-spread-area + depth (or measured capacity) for\n"
              "        these tanks and re-running build_tank_params.py, OR\n"
              "    (b) excluding them from the run - drop their rows from every input\n"
              "        file (tank_params, demand, network_alpha rows AND columns).\n"
              "  Option (b) is valid for a first end-to-end test; document the exclusion."
        )

    # ---- Demand (full season) ----
    demand_df = pd.read_csv(_require(os.path.join(dd, cfg.FILE_DEMAND)),
                            index_col="tank_id")
    demand_full = demand_df.to_numpy(dtype=float)               # (N, season_days)

    # ---- Network alpha ----
    alpha = pd.read_csv(_require(os.path.join(dd, cfg.FILE_NETWORK_ALPHA)),
                        index_col=0).to_numpy(dtype=float)       # (N, N)
    if alpha.shape != (N, N):
        raise ValueError(f"network_alpha.csv must be ({N},{N}), got {alpha.shape}.")

    # ---- Rainfall history ----
    rain_df = pd.read_csv(_require(os.path.join(dd, cfg.FILE_RAINFALL_HISTORY)),
                          index_col="tank_id")
    rainfall_history = rain_df.to_numpy(dtype=float)            # (N, H)

    return {
        "N": N,
        "S_min": S_min,
        "S_max": S_max,
        "R_max": R_max,
        "E": E,
        "catchment_km2": catchment_km2,
        "alpha": alpha,
        "demand_full": demand_full,
        "rainfall_history": rainfall_history,
        "tank_ids": tank_ids,
    }


def load_module3_risk(cfg: Config = DEFAULT_CONFIG):
    """
    Load Module 3's forecast + risk output (used for Option A weights and the
    cross-check). Returns None if the file is absent (weights fall back to base).

    Expected columns (one row per tank):
        tank_id, drought_probability, overflow_probability, normal_probability,
        pred_t+1_storage .. pred_t+7_storage (optional, for cross-check)
    """
    path = os.path.join(cfg.DATA_DIR, cfg.FILE_MODULE3_RISK)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="tank_id")
    return df


def load_mahaweli(cfg: Config = DEFAULT_CONFIG):
    """
    Load the Mahaweli schedule (only needed if C6 is enabled). Returns
    (mah_ids, R_sched, delta_m) or (None, None, None) if not applicable.
    """
    if not cfg.ENABLE_C6_MAHAWELI:
        return None, None, None
    path = _require(os.path.join(cfg.DATA_DIR, cfg.FILE_MAHAWELI))
    mah_df = pd.read_csv(path, index_col="reservoir_id")
    day_cols = [c for c in mah_df.columns if c.startswith("day")][: cfg.T]
    R_sched = mah_df[day_cols].to_numpy(dtype=float)            # (M, T)
    # delta_m based on SCHEDULED release volume (mean over horizon), not capacity
    delta_m = cfg.DELTA_M_FRACTION * R_sched.mean(axis=1)       # (M,)
    mah_ids = list(mah_df.index)                               # tank indices
    return mah_ids, R_sched, delta_m


def load_current_storage(cfg: Config = DEFAULT_CONFIG):
    """
    Read today's storage for every tank from tank_storage.csv (the file Module 1
    writes / a satellite measurement).

    Used by the MPC loop when RELOAD_STORAGE_EACH_DAY is True (deployment mode:
    re-read the measured storage each day) and internally by build_state for the
    initial day-0 state.

    Returns
    -------
    S_current : (N,) np.ndarray of storage in m3, ordered as tank_params.csv.

    Raises
    ------
    ValueError if the 'storage_m3' column is missing or any storage is negative.
    """
    storage_path = _require(os.path.join(cfg.DATA_DIR, cfg.FILE_TANK_STORAGE))
    storage_df = pd.read_csv(storage_path, index_col="tank_id")
    if "storage_m3" not in storage_df.columns:
        raise ValueError("tank_storage.csv must include a 'storage_m3' column.")
    S_current = storage_df["storage_m3"].to_numpy(dtype=float)
    if np.any(S_current < 0):
        raise ValueError("Negative storage detected in tank_storage.csv "
                         "(check Module 1 output).")
    return S_current


def build_state(static, module3_df, t_current, cfg: Config = DEFAULT_CONFIG,
                S_current=None):
    """
    Assemble the per-MPC-step SystemState dict.

    Parameters
    ----------
    static     : dict from load_static_inputs
    module3_df : DataFrame from load_module3_risk (or None)
    t_current  : int  current day index in the season
    cfg        : Config
    S_current  : (N,) or None. If provided (e.g. the internally-advanced storage
                 from the previous MPC day in simulation mode), it is used directly.
                 If None, today's storage is read from tank_storage.csv.

    Returns
    -------
    state : dict with everything the optimiser/objectives/constraints need,
            plus scenarios and the risk probabilities for TOPSIS.
    """
    N = static["N"]

    # ---- Today's storage ----
    # Either supplied by the caller (simulation: advanced from yesterday) or read
    # from file (day 0, or deployment mode with a fresh Module 1 measurement).
    if S_current is None:
        S_current = load_current_storage(cfg)

    # ---- Demand window for this horizon ----
    D = demand_window(static["demand_full"], t_current, cfg.T)  # (N, T)

    # ---- Rainfall scenarios (Gamma fit to history) ----
    scenarios = generate_scenarios(static["rainfall_history"],
                                   static["catchment_km2"], cfg)  # (K, N, T)
    Q_expected = scenarios.mean(axis=0)                         # (N, T) scenario mean

    # ---- Module 3 risk probabilities (network-level, for Option A weights) ----
    if module3_df is not None:
        p_drought = float(module3_df["drought_probability"].mean())
        p_overflow = float(module3_df["overflow_probability"].mean())
    else:
        p_drought = 0.0
        p_overflow = 0.0

    # ---- Mahaweli (only if C6 active) ----
    mah_ids, R_sched, delta_m = load_mahaweli(cfg)

    # ---- Terminal-sink separation ----------------------------------------
    # Identify which array positions are terminal-sink reservoirs. These stay in
    # the simulation and validation but are removed from the optimisation.
    tank_ids = static["tank_ids"]
    sink_mask = np.array([tid in cfg.TERMINAL_SINK_TANKS for tid in tank_ids])
    sink_idx = list(np.where(sink_mask)[0])          # positions optimised OUT
    opt_idx = list(np.where(~sink_mask)[0])          # positions the optimiser decides
    for tid in cfg.TERMINAL_SINK_TANKS:
        if tid not in tank_ids:
            raise ValueError(
                f"TERMINAL_SINK_TANKS lists '{tid}' but it is not in tank_params.csv. "
                f"Check the exact tank_id spelling."
            )

    state = {
        "N": N,
        "T": cfg.T,
        "K": cfg.K,
        "S_current": np.asarray(S_current, dtype=float),
        "S_min": static["S_min"],
        "S_max": static["S_max"],
        "R_max": static["R_max"],
        "E": static["E"],
        "D": D,
        "alpha": static["alpha"],
        "scenarios": scenarios,
        "Q_expected": Q_expected,
        "p_drought": p_drought,
        "p_overflow": p_overflow,
        "tank_ids": tank_ids,
        # terminal-sink bookkeeping (empty sink_idx => original all-tank behaviour)
        "sink_mask": sink_mask,
        "sink_idx": sink_idx,
        "opt_idx": opt_idx,
    }
    if mah_ids is not None:
        state["mah_ids"] = mah_ids
        state["R_sched"] = R_sched
        state["delta_m"] = delta_m
    return state