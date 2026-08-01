"""
topsis.py
========
TOPSIS selection from the Pareto front, with Option A risk-driven weighting.

TOPSIS (Hwang & Yoon 1981): the best compromise is the solution simultaneously
closest to the ideal point and farthest from the anti-ideal point in weighted,
normalised objective space.

Option A - dynamic weight adjustment:
Module 3's next-day risk probabilities adjust the FOUR base weights before TOPSIS:

    w1 = w1_base + alpha_ds * P_drought          (shortage up under drought)
    w3 = w3_base + alpha_de * P_drought          (equity up under drought)
    w2 = w2_base + beta     * P_overflow         (overflow up under overflow risk)
    w4 = w4_base - gamma    * P_overflow         (loss down under overflow risk)

then every weight is floored at WEIGHT_FLOOR and the vector is renormalised to sum
to 1. The applied weights are returned so they can be logged per MPC step.

Module 3's probabilities enter ONLY here (not the optimiser, not the scenario
generator, not the constraints) - this keeps a single, clean point of influence
and avoids double-counting.
"""

import numpy as np
from .config import Config, DEFAULT_CONFIG
from .state_transition import simulate_cascade


def compute_risk_weights(p_drought: float, p_overflow: float,
                         cfg: Config = DEFAULT_CONFIG):
    """
    Compute the risk-adjusted TOPSIS weights for one MPC step (Option A).

    Parameters
    ----------
    p_drought  : float in [0,1]  network-level drought probability (e.g. mean over
                                 tanks, or the max, from Module 3) for tomorrow
    p_overflow : float in [0,1]  network-level overflow probability
    cfg        : Config

    Returns
    -------
    weights : np.ndarray shape (4,)  adjusted [w1, w2, w3, w4], summing to 1
    """
    if not cfg.USE_MODULE3_RISK_WEIGHTS:
        return np.array([cfg.W_SHORTAGE_BASE, cfg.W_OVERFLOW_BASE,
                         cfg.W_EQUITY_BASE, cfg.W_LOSS_BASE])

    w1 = cfg.W_SHORTAGE_BASE + cfg.ALPHA_DROUGHT_SHORTAGE * p_drought
    w3 = cfg.W_EQUITY_BASE   + cfg.ALPHA_DROUGHT_EQUITY   * p_drought
    w2 = cfg.W_OVERFLOW_BASE + cfg.BETA_OVERFLOW          * p_overflow
    w4 = cfg.W_LOSS_BASE     - cfg.GAMMA_OVERFLOW_LOSS    * p_overflow

    w = np.array([w1, w2, w3, w4])
    w = np.maximum(w, cfg.WEIGHT_FLOOR)     # floor so nothing is ever ignored
    w = w / w.sum()                          # renormalise to sum to 1
    return w


def topsis_rank(pareto_F, pareto_X, state, weights, top_n=3, cfg: Config = DEFAULT_CONFIG):
    """
    Rank Pareto solutions with TOPSIS and return the top_n strategies.

    Parameters
    ----------
    pareto_F : (n_sol, 4)   objective values
    pareto_X : (n_sol, N*T) decision variables
    state    : dict         system state (for N, T, D)
    weights  : (4,)         TOPSIS weights (from compute_risk_weights)
    top_n    : int
    cfg      : Config

    Returns
    -------
    strategies : list of dicts, best first, each with the full release plan,
                 day-0 releases, per-tank satisfaction, and the objective values.
    """
    if pareto_F.shape[0] == 0:
        return []

    n_sol = pareto_F.shape[0]
    N, T = state["N"], cfg.T

    # Step 1: vector normalisation of each objective column
    norms = np.linalg.norm(pareto_F, axis=0)          # (4,)
    norms = np.where(norms == 0, 1.0, norms)
    F_norm = pareto_F / norms                          # (n_sol, 4)

    # Step 2: apply weights
    V = F_norm * weights                               # (n_sol, 4)

    # Step 3: ideal (min, since all objectives are costs) and anti-ideal (max)
    ideal = np.min(V, axis=0)                          # (4,)
    anti_ideal = np.max(V, axis=0)                     # (4,)

    # Step 4: Euclidean separation from ideal and anti-ideal
    d_best = np.sqrt(np.sum((V - ideal) ** 2, axis=1))      # (n_sol,)
    d_worst = np.sqrt(np.sum((V - anti_ideal) ** 2, axis=1))# (n_sol,)

    # Step 5: closeness coefficient
    closeness = d_worst / (d_best + d_worst + 1e-10)   # (n_sol,)

    # Step 6: rank descending, unpack top_n
    order = np.argsort(-closeness)
    D = state["D"]

    # Reconstruct the FULL (N, T) release from the reduced decision vector, which
    # only spans the optimised tanks. The terminal-sink release is added back from
    # its operating rule so downstream reporting/simulation sees the full cascade.
    from .nsga2_optimizer import build_full_release

    strategies = []
    for rank, idx in enumerate(order[:top_n]):
        R = build_full_release(pareto_X[idx], state)
        sim = simulate_cascade(
            state["S_current"],
            R,
            state["Q_expected"],
            state["alpha"],
            state["E"],
            state["S_min"],
            state["S_max"],
            state["D"],
            storage_limited_evap=getattr(cfg, "STORAGE_LIMITED_EVAPORATION", True),
            upstream_iters=getattr(cfg, "UPSTREAM_FIXED_POINT_ITERS", 3),
        )
        demand = np.sum(state["D"], axis=1)
        release = np.sum(sim["R_effective"], axis=1)
        consumed = np.sum(sim["consumed"], axis=1)
        return_flow = np.sum(sim["return_flow"], axis=1)
        local_inflow = np.sum(state["Q_expected"], axis=1)
        upstream_inflow = np.sum(state["alpha"] @ sim["return_flow"], axis=1)
        storage_start = state["S_current"]
        storage_end = sim["S_clipped"][:, -1]

        # per-tank demand satisfaction averaged over the horizon (active days only)
        with np.errstate(divide="ignore", invalid="ignore"):
            sat = np.where(state["D"] > 0, sim["consumed"] / state["D"], np.nan)
        avg_sat = np.nanmean(sat, axis=1) * 100.0      # (N,) percent (nan-safe)
        avg_sat = np.nan_to_num(avg_sat, nan=100.0)    # tanks with no demand -> 100%

        total_release = float(np.sum(sim["R_effective"]))
        total_consumed = float(np.sum(sim["consumed"]))
        total_return_flow = float(np.sum(sim["return_flow"]))
        total_channel_loss = float(
            total_return_flow - np.sum(state["alpha"] @ sim["return_flow"])
        )

        strategies.append({
            "rank": rank + 1,
            "closeness": round(float(closeness[idx]), 4),
            "f1_shortage": round(float(pareto_F[idx, 0]), 2),
            "f2_overflow": round(float(pareto_F[idx, 1]), 2),
            "f3_equity": round(float(pareto_F[idx, 2]), 4),
            "f4_water_loss": round(float(pareto_F[idx, 3]), 2),
            "total_release": total_release,
            "total_consumed": total_consumed,
            "total_return_flow": total_return_flow,
            "total_channel_loss": total_channel_loss,
            "tank_table": [
                [
                    state["tank_ids"][i],
                    round(float(demand[i]), 2),
                    round(float(release[i]), 2),
                    round(float(consumed[i]), 2),
                    round(float(return_flow[i]), 2),
                    round(float(avg_sat[i]), 1),
                ]
                for i in range(len(state["tank_ids"]))
            ],
            "balance_table": [
                [
                    state["tank_ids"][i],
                    round(float(storage_start[i]), 2),
                    round(float(local_inflow[i]), 2),
                    round(float(upstream_inflow[i]), 2),
                    round(float(release[i]), 2),
                    round(float(consumed[i]), 2),
                    round(float(storage_end[i]), 2),
                ]
                for i in range(len(state["tank_ids"]))
            ],
            "R_matrix": R,                 # (N, T) full release plan
            "R_today": R[:, 0].copy(),     # (N,) day-0 releases (what gets applied)
            "satisfaction_pct": avg_sat,   # (N,)
        })
    return strategies