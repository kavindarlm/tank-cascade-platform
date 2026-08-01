"""
cross_validation.py
==================
Inter-module sanity check: compare Module 4's SIMULATED storage trajectory
(under the chosen release plan) against Module 3's independent 7-day forecast.

--------------------------------------------------------------------------
WHY THIS IS RATIO-BASED, NOT LEVEL-BASED  (Issue 2 fix)

Module 3 reports storage as a PERCENTAGE, but its 100% reference is Module 1's
`total_ha` - the maximum historical flood EXTENT, which is (a) an AREA, and
(b) a level real tanks never approach (no tank exceeded ~82% in 2021-2025).

Module 4 works in VOLUME (m3) relative to Full Supply Level.

Comparing those two percentages directly is meaningless - the denominators are
different quantities entirely, and area does not scale linearly with volume.

The fix: normalise EACH series to its own baseline day. The unknown denominator
cancels algebraically:

    F(t)/F(b) = [100*A(t)/total_ha] / [100*A(b)/total_ha] = A(t)/A(b)

Then convert the area ratio to a volume ratio using the conic reservoir
approximation (area ~ r^2, volume ~ r^3, so V ~ A^1.5):

    V(t)/V(b) = ( A(t)/A(b) ) ^ 1.5

...and compare against Module 4's own volume ratio S(t)/S(b). Both sides are now
dimensionless, denominator-free, and on the same volumetric basis.

WHAT THIS VALIDATES: whether the physics-based simulation and the learned
forecast agree on the DIRECTION and RATE of storage change.
WHAT IT DOES NOT VALIDATE: absolute level agreement (that would need Module 1's
total_ha, the FSL-to-total_ha relationship, and a real hypsometric curve).

If Module 3 confirms `storage_forecast` is volumetric rather than area-based,
set cfg.MODULE3_STORAGE_BASIS = 'volume' and the exponent is skipped.
--------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd

from .state_transition import simulate_cascade
from .config import Config, DEFAULT_CONFIG


# ----------------------------------------------------------------------
# Column parsing - tolerant of both the old flat CSV and the new field names
# ----------------------------------------------------------------------

def forecast_columns(module3_df, cfg: Config = DEFAULT_CONFIG):
    """
    Find Module 3's 7-day storage forecast columns, accepting either naming:
        old flat CSV : pred_t+1_storage .. pred_t+7_storage
        new export   : t+1 .. t+7   (flattened from storage_forecast{})

    Returns
    -------
    list of column names in day order, or None if not present.
    """
    for pattern in ("pred_t+{}_storage", "t+{}", "storage_forecast_t+{}"):
        cols = [pattern.format(d + 1) for d in range(cfg.T)]
        if all(c in module3_df.columns for c in cols):
            return cols
    return None


# ----------------------------------------------------------------------
# Module 4 side
# ----------------------------------------------------------------------

def simulated_trajectory(state, R, cfg: Config = DEFAULT_CONFIG):
    """
    Simulate the chosen plan under expected inflow and return the storage
    trajectory in m3 for forecast days 1..T.

    Parameters
    ----------
    state : dict   system state
    R     : (N, T) chosen release plan
    cfg   : Config

    Returns
    -------
    (N, T)  simulated storage volume (m3) at the end of each day
    """
    res = simulate_cascade(
        state["S_current"], R, state["Q_expected"],
        state["alpha"], state["E"], state["S_min"], state["S_max"],
        # Consumption must be applied here too, or the trajectory compared against
        # Module 3 would retain water the crops actually drank.
        state["D"] if getattr(cfg, "ENABLE_CROP_CONSUMPTION", True) else None,
        storage_limited_evap=getattr(cfg, "STORAGE_LIMITED_EVAPORATION", True),
        upstream_iters=getattr(cfg, "UPSTREAM_FIXED_POINT_ITERS", 3),
    )
    return res["S_clipped"][:, 1:]          # (N, T), days 1..T


# ----------------------------------------------------------------------
# Ratio construction
# ----------------------------------------------------------------------

def to_ratio(series, baseline_idx):
    """
    Normalise a 1-D series to its baseline element.

    Returns None if the baseline is non-positive (ratio undefined).
    """
    base = series[baseline_idx]
    if base is None or not np.isfinite(base) or base <= 0:
        return None
    return series / base


def module3_volume_ratio(forecast_pct, baseline_idx, cfg: Config = DEFAULT_CONFIG):
    """
    Convert Module 3's percentage forecast into a VOLUME ratio series.

    Parameters
    ----------
    forecast_pct : (T,)  Module 3's storage percentages for days 1..T
    baseline_idx : int   index of the anchor day
    cfg          : Config

    Returns
    -------
    (T,) volume ratio relative to the baseline day, or None if undefined.
    """
    ratio = to_ratio(np.asarray(forecast_pct, dtype=float), baseline_idx)
    if ratio is None:
        return None
    if cfg.MODULE3_STORAGE_BASIS == "area":
        # area ratio -> volume ratio via conic approximation
        return np.power(ratio, cfg.AREA_TO_VOLUME_EXPONENT)
    return ratio       # already volumetric; compare directly


# ----------------------------------------------------------------------
# Main comparison
# ----------------------------------------------------------------------

def compare_with_module3(state, R, module3_df, cfg: Config = DEFAULT_CONFIG):
    """
    Build a per-tank, per-day comparison of TRAJECTORY SHAPE between
    Module 4's simulation and Module 3's forecast.

    Both series are normalised to the same baseline forecast day, so the
    comparison is independent of each module's percentage/volume reference.

    Parameters
    ----------
    state      : dict     system state (needs tank_ids, S_current, S_max, ...)
    R          : (N, T)   chosen release plan
    module3_df : DataFrame indexed by tank_id with forecast columns
    cfg        : Config

    Returns
    -------
    DataFrame with columns:
        tank_id, day,
        m4_volume_ratio      - Module 4 simulated volume relative to baseline
        m3_volume_ratio      - Module 3 forecast converted to volume ratio
        ratio_difference     - m4 - m3 (signed)
        abs_ratio_difference - |m4 - m3|
    Returns None if Module 3 data or its forecast columns are unavailable.
    """
    if module3_df is None:
        return None
    cols = forecast_columns(module3_df, cfg)
    if cols is None:
        return None

    sim_m3vol = simulated_trajectory(state, R, cfg)        # (N, T) in m3
    tank_ids = state["tank_ids"]
    b = max(0, min(cfg.CROSSCHECK_BASELINE_DAY - 1, cfg.T - 1))   # 0-based anchor

    rows = []
    for i, tid in enumerate(tank_ids):
        if tid not in module3_df.index:
            continue

        m4_ratio = to_ratio(sim_m3vol[i, :], b)
        m3_ratio = module3_volume_ratio(
            module3_df.loc[tid, cols].to_numpy(dtype=float), b, cfg
        )
        if m4_ratio is None or m3_ratio is None:
            continue

        for d in range(cfg.T):
            diff = float(m4_ratio[d] - m3_ratio[d])
            rows.append({
                "tank_id": tid,
                "day": d + 1,
                "m4_volume_ratio": round(float(m4_ratio[d]), 4),
                "m3_volume_ratio": round(float(m3_ratio[d]), 4),
                "ratio_difference": round(diff, 4),
                "abs_ratio_difference": round(abs(diff), 4),
            })

    return pd.DataFrame(rows) if rows else None


# ----------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------

def divergence_summary(comparison_df, cfg: Config = DEFAULT_CONFIG):
    """
    Summarise agreement between Module 4 and Module 3 across all tanks/days.

    Returns
    -------
    dict with mean/max ratio difference, trend-direction agreement rate, the
    list of tanks exceeding the divergence threshold, or None if unavailable.
    """
    if comparison_df is None or len(comparison_df) == 0:
        return None

    d = comparison_df["abs_ratio_difference"]

    # Trend-direction agreement: on the final day, do both say rising or falling?
    last = comparison_df[comparison_df["day"] == comparison_df["day"].max()]
    same_direction = (
        np.sign(last["m4_volume_ratio"] - 1.0) ==
        np.sign(last["m3_volume_ratio"] - 1.0)
    )
    direction_agreement = float(same_direction.mean()) if len(last) else float("nan")

    # Tanks whose MEAN divergence exceeds the threshold
    per_tank = comparison_df.groupby("tank_id")["abs_ratio_difference"].mean()
    flagged = per_tank[per_tank > cfg.CROSSCHECK_DIVERGENCE_THRESHOLD]

    return {
        "mean_abs_ratio_diff": round(float(d.mean()), 4),
        "max_abs_ratio_diff": round(float(d.max()), 4),
        "direction_agreement_rate": round(direction_agreement, 3),
        "n_tanks_compared": int(comparison_df["tank_id"].nunique()),
        "n_comparisons": int(len(d)),
        "flagged_tanks": list(flagged.index),
        "divergence_threshold": cfg.CROSSCHECK_DIVERGENCE_THRESHOLD,
    }


def per_tank_divergence(comparison_df):
    """
    Per-tank mean and max divergence, sorted worst-first. Useful for spotting
    which tanks the physics and the forecast disagree about.
    """
    if comparison_df is None or len(comparison_df) == 0:
        return None
    g = comparison_df.groupby("tank_id")["abs_ratio_difference"]
    out = pd.DataFrame({
        "mean_abs_ratio_diff": g.mean().round(4),
        "max_abs_ratio_diff": g.max().round(4),
    }).sort_values("mean_abs_ratio_diff", ascending=False)
    return out.reset_index()