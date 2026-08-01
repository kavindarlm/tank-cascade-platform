"""
objectives.py
============
The four objective functions (all costs, lower is better).

    f1  shortage   : sum max(0, D - R)          - unmet demand
    f2  overflow   : sum spill                   - storage above capacity (pre-clip)
    f3  equity     : sum |rho_i - rho_bar|       - dispersion of satisfaction ratios
    f4  water_loss : sum max(0, R - D)           - over-release beyond demand

Fixes vs the original implementation:
  - f2 uses the pre-clip spill recorded by state_transition (Bug 1). It no longer
    reads a clipped trajectory that can never exceed capacity.
  - f3 EXCLUDES zero-demand tanks from both the mean and the sum (Bug 6). A tank
    that needs no water should not contribute artificial inequity.
  - f1/f4 use the EFFECTIVE release (what could actually be delivered), so a plan
    is scored on the water it truly moves, consistent with mass conservation.

Terminal-sink separation:
  Every objective takes an optional `rows` selector (a list of tank indices). When
  provided, the objective is summed over ONLY those tanks - the terminal-sink
  reservoir is left out so its order-of-magnitude-larger demand cannot dominate
  f1 or distort equity. The sink tank is still simulated and validated elsewhere;
  it simply does not contribute to the optimisation cost. Passing rows=None keeps
  the original all-tank behaviour.
"""

import numpy as np


def _select(arr, rows):
    """Return arr restricted to the given tank rows (or unchanged if rows is None)."""
    if rows is None:
        return arr
    return arr[rows, :]


def f1_shortage(R_eff, D, rows=None):
    """
    f1: total unmet demand (summed over the selected tanks).

    Parameters
    ----------
    R_eff : (N, T)  effective (deliverable) release
    D     : (N, T)  demand
    rows  : list or None  tank indices to include (None = all tanks)

    Returns
    -------
    float (lower is better)
    """
    shortfall = np.maximum(0.0, _select(D, rows) - _select(R_eff, rows))
    return float(np.sum(shortfall))


def f2_overflow(spill, rows=None):
    """
    f2: total overflow volume, taken directly from the pre-clip spill series
    recorded by the simulator (summed over the selected tanks).

    Parameters
    ----------
    spill : (N, T)  overflow volume each day
    rows  : list or None  tank indices to include (None = all tanks)

    Returns
    -------
    float (lower is better)
    """
    return float(np.sum(_select(spill, rows)))


def f3_equity(R_eff, D, rows=None):
    """
    f3: allocation inequity across tanks, measured as the mean-absolute-deviation
    of satisfaction ratios rho_i = R_i / D_i.

    Zero-demand tanks (D_i(t) == 0) are excluded from BOTH the network average and
    the deviation sum for that day. Equity is measured among the SELECTED tanks
    only (the terminal sink is excluded so its scale does not warp the ratios).

    Parameters
    ----------
    R_eff : (N, T)  effective release
    D     : (N, T)  demand
    rows  : list or None  tank indices to include (None = all tanks)

    Returns
    -------
    float (lower is better)
    """
    R_sel = _select(R_eff, rows)
    D_sel = _select(D, rows)
    N, T = R_sel.shape
    total_inequity = 0.0
    for t in range(T):
        D_t = D_sel[:, t]
        R_t = R_sel[:, t]
        active = D_t > 0                     # tanks that actually need water today
        if not np.any(active):
            continue                         # nobody needs water; no inequity defined
        rho = R_t[active] / D_t[active]      # satisfaction ratios of active tanks
        rho_bar = np.mean(rho)
        total_inequity += float(np.sum(np.abs(rho - rho_bar)))
    return total_inequity


def f4_water_loss(R_eff, D, rows=None):
    """
    f4: total over-release beyond demand (summed over the selected tanks).

    Parameters
    ----------
    R_eff : (N, T)  effective release
    D     : (N, T)  demand
    rows  : list or None  tank indices to include (None = all tanks)

    Returns
    -------
    float (lower is better)
    """
    over = np.maximum(0.0, _select(R_eff, rows) - _select(D, rows))
    return float(np.sum(over))


def evaluate_objectives_single(sim_result, D, rows=None):
    """
    Evaluate all four objectives for ONE simulated scenario.

    Parameters
    ----------
    sim_result : dict returned by simulate_cascade (has 'R_effective', 'spill')
    D          : (N, T) demand
    rows       : list or None  tank indices to include in the cost (None = all).
                 Pass the optimised-tank indices to exclude the terminal sink.

    Returns
    -------
    np.ndarray shape (4,) = [f1, f2, f3, f4]
    """
    R_eff = sim_result["R_effective"]
    spill = sim_result["spill"]
    return np.array([
        f1_shortage(R_eff, D, rows),
        f2_overflow(spill, rows),
        f3_equity(R_eff, D, rows),
        f4_water_loss(R_eff, D, rows),
    ])


def evaluate_objectives_expected(sim_results, D, rows=None):
    """
    Average the four objectives over all K scenarios (sample-average approximation
    of the expected cost).

    Parameters
    ----------
    sim_results : list of K simulate_cascade dicts
    D           : (N, T) demand
    rows        : list or None  tank indices to include (None = all). Pass the
                  optimised-tank indices to exclude the terminal sink from cost.

    Returns
    -------
    np.ndarray shape (4,) = expected [f1, f2, f3, f4]
    """
    K = len(sim_results)
    acc = np.zeros(4)
    for res in sim_results:
        acc += evaluate_objectives_single(res, D, rows)
    return acc / K