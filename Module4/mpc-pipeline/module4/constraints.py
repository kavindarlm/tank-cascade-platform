"""
constraints.py
=============
The six constraint checkers. Each returns a violation magnitude >= 0
(0 = satisfied). These are wired into pymoo's constraint-domination machinery
(out['G']) AND/OR added as a penalty, so infeasible plans are driven out.

Fixes vs the original implementation:
  - C1 uses the pre-clip spill/deficit from the simulator, so violations are
    actually detectable (Bug 1 / Bug 2).
  - C3 is reformulated as a DETERMINISTIC seasonal safety floor. Module 3's
    drought probability no longer enters here - under Option A it drives TOPSIS
    weights instead. This removes the dangling-risk-input problem.
  - C4 correctly includes inflow Q: sum(R) <= sum(S + Q) per day.
  - C6 tolerance delta_m is based on the SCHEDULED release volume, not tank
    capacity, and only evaluated when C6 is enabled (off for Mahakanumulla).
  - Constraints can be individually toggled from config.

Each constraint function returns a scalar total violation. `check_all_constraints`
assembles the active ones into a vector for the optimiser.
"""

import numpy as np
from .config import Config, DEFAULT_CONFIG


def _rows(arr, rows):
    """Restrict a (N, T) array to selected tank rows (or all if rows is None)."""
    return arr if rows is None else arr[rows, :]


# ======================================================================
# Baseline-relative floors  (see C3_REFORMULATION.md)
# ======================================================================
def build_baseline_trajectory(state, cfg: Config = DEFAULT_CONFIG, Q=None,
                              R_base=None):
    """
    Storage trajectory under the NO-ACTION plan: zero release on the optimised
    tanks, driven only by forecast inflow and evaporation.

    This is the reference C1's low side and C3 are measured against. The
    controller is responsible for the storage IT removes, not for the storage
    nature removes or for what it inherited from Modules 1/3.

    R_base defaults to all-zero. Pass state['_R_sink'] so the baseline includes
    the terminal sink's rule-based release: the guaranteed-feasible point the
    optimiser can actually reach is "R = 0 on the optimised tanks plus R_sink on
    the sink rows", not R = 0 everywhere. For a genuinely terminal sink the two
    coincide (its alpha column is zero), but making it explicit keeps the
    guarantee true for any TERMINAL_SINK_TANKS setting.

    IMPORTANT: S_base must be built on the SAME inflow the constraints are
    evaluated on. CascadeReleaseProblem._evaluate uses Q_expected, so that is
    the default here. If constraints ever move to per-scenario evaluation the
    baseline must move with them, or the feasibility guarantee silently fails.

    Returns
    -------
    (N, T) array - baseline storage for days 1..T.
    """
    from .state_transition import simulate_cascade

    N, T = int(state["N"]), int(state["T"])
    Q = state["Q_expected"] if Q is None else Q
    R_base = np.zeros((N, T)) if R_base is None else np.asarray(R_base, float)
    D_sim = state["D"] if getattr(cfg, "ENABLE_CROP_CONSUMPTION", True) else None

    sim = simulate_cascade(
        state["S_current"], R_base, Q,
        state["alpha"], state["E"], state["S_min"], state["S_max"], D_sim,
        storage_limited_evap=getattr(cfg, "STORAGE_LIMITED_EVAPORATION", True),
        upstream_iters=getattr(cfg, "UPSTREAM_FIXED_POINT_ITERS", 3),
    )
    # Unclipped: clipping at S_min would invent water and censor the baseline.
    return np.asarray(sim["S_unclipped"], dtype=float)[:, 1:]


def _baseline(state, cfg: Config = DEFAULT_CONFIG):
    """Cached baseline trajectory for the current MPC day."""
    S_base = state.get("S_base")
    if S_base is None:
        S_base = build_baseline_trajectory(state, cfg,
                                           R_base=state.get("_R_sink"))
        state["S_base"] = S_base
    return S_base


def c3_effective_floor(state, cfg: Config = DEFAULT_CONFIG):
    """
    Floor actually enforced by C3 on each tank and day:

        floor_eff(t) = min( S_base(t), S_min + buffer * S_max )

    A tank comfortably above the reserve level has S_base >= floor, so the FULL
    absolute floor binds and normal drawdown protection is unchanged. A tank
    below the reserve level, or falling below it naturally, is only required not
    to do worse than doing nothing.

    Because zero release reproduces S_base and lies inside [0, R_max], the
    feasible set is non-empty BY CONSTRUCTION - the recursive feasibility the
    absolute floor lacked.

    Set cfg.C3_BASELINE_FLOOR = False for the original absolute floor.
    """
    T = int(state["T"])
    floor_mat = np.repeat(_c3_floor_vector(state, cfg)[:, None], T, axis=1)
    if not getattr(cfg, "C3_BASELINE_FLOOR", True):
        return floor_mat
    return np.minimum(floor_mat, np.asarray(_baseline(state, cfg), dtype=float))


def c1_deficit_floor(state, cfg: Config = DEFAULT_CONFIG):
    """
    Lower bound enforced by C1: min( S_min, S_base(t) ).

    Same disease, same cure. If a tank starts below dead storage no release can
    raise it, so 'deficit' stays positive for every candidate and the feasible
    set is empty via C1 rather than C3.

    Only the LOW side is relaxed. The HIGH side (spill above S_max) stays hard:
    releasing more reduces spill, so the optimiser has genuine authority there
    and the constraint is legitimately binding.
    """
    T = int(state["T"])
    S_min_mat = np.repeat(np.asarray(state["S_min"], dtype=float)[:, None],
                          T, axis=1)
    if not getattr(cfg, "C1_BASELINE_DEFICIT", True):
        return S_min_mat
    return np.minimum(S_min_mat, np.asarray(_baseline(state, cfg), dtype=float))


def _cached_c3_floor(state, cfg: Config = DEFAULT_CONFIG):
    """C3 floor matrix, built once per MPC day (called POP_SIZE x N_GEN times)."""
    m = state.get("_c3_floor_mat")
    if m is None:
        m = c3_effective_floor(state, cfg)
        state["_c3_floor_mat"] = m
    return m


def _cached_c1_floor(state, cfg: Config = DEFAULT_CONFIG):
    """C1 lower-bound matrix, built once per MPC day."""
    m = state.get("_c1_floor_mat")
    if m is None:
        m = c1_deficit_floor(state, cfg)
        state["_c1_floor_mat"] = m
    return m


def c3_absolute_reserve_deficit(sim_result, state, cfg: Config = DEFAULT_CONFIG,
                                rows=None):
    """
    Shortfall against the ABSOLUTE seasonal floor, ignoring the baseline
    relaxation. Never enters out['G'] - a reported metric only.

    After the reformulation the enforced C3 violation reads 0 on most days.
    Logging this alongside it shows whether the reserve problem was solved or
    merely made non-binding, which is the honest presentation.
    """
    key = ("S_unclipped" if getattr(cfg, "C3_FLOOR_ON_UNCLIPPED", True)
           else "S_clipped")
    S = np.asarray(sim_result[key], dtype=float)[:, 1:]
    viol = np.maximum(0.0, _c3_floor_vector(state, cfg)[:, None] - S)
    if rows is not None:
        viol = viol[np.asarray(rows, dtype=int), :]
    return float(np.sum(viol))


def c1_storage_bounds(sim_result, rows=None, deficit_floor=None):
    """
    C1: S_min <= S(t) <= S_max for all tanks, all days (days 1..T).

    Uses the pre-clip spill (above S_max) and deficit (below S_min) recorded by
    the simulator, so the violation is genuinely non-zero when the plan pushes a
    tank out of bounds.

    With `rows`, only the selected (optimised) tanks are enforced; the terminal
    sink's bounds are monitored in the simulation rather than enforced here.

    Low side: measured against `deficit_floor` = min(S_min, S_base) when given,
    so an inherited or evaporation-driven shortfall is not charged to the
    optimiser. Omit it to recover the original behaviour.

    Returns
    -------
    float total violation
    """
    spill = _rows(sim_result["spill"], rows)          # (n, T) above S_max
    if deficit_floor is None:
        deficit = _rows(sim_result["deficit"], rows)  # (n, T) below S_min
    else:
        S = np.asarray(sim_result["S_unclipped"], dtype=float)[:, 1:]
        deficit = _rows(np.maximum(0.0, np.asarray(deficit_floor, dtype=float) - S),
                        rows)
    return float(np.sum(spill) + np.sum(deficit))


def c2_release_bounds(R, R_max):
    """
    C2: 0 <= R_i(t) <= R_max_i.

    Parameters
    ----------
    R     : (N, T)  requested release
    R_max : (N,)    max release per tank

    Returns
    -------
    float total violation
    """
    low = np.maximum(0.0, -R)                          # negative releases
    high = np.maximum(0.0, R - R_max[:, None])         # above sluice capacity
    return float(np.sum(low) + np.sum(high))


def c3_drought_floor(sim_result, S_min, S_max, cfg: Config, rows=None,
                     floor_mat=None):
    """
    C3: deterministic seasonal safety floor (Option A reformulation).

    Baseline-relative form (see c3_effective_floor):

        S(t) >= min( S_base(t), S_min + buffer_fraction * S_max )

    The absolute floor is a fixed hedging rule (You & Cai 2008) that does NOT
    depend on Module 3's probability - that drives TOPSIS weights instead. It is
    unsatisfiable whenever a tank starts below it, because release is
    unidirectional, so the enforced floor is capped at the no-action baseline.

    The trajectory is read from 'S_unclipped': clipping at S_min invents water
    and censors the violation at buffer*S_max per tank-day.

    Parameters
    ----------
    sim_result : dict with 'S_clipped'
    S_min      : (N,)
    S_max      : (N,)
    cfg        : Config

    Returns
    -------
    float total violation
    """
    key = ("S_unclipped" if getattr(cfg, "C3_FLOOR_ON_UNCLIPPED", True)
           else "S_clipped")
    S = np.asarray(sim_result[key], dtype=float)[:, 1:]   # (N, T) days 1..T
    if floor_mat is None:
        # Absolute floor - preserves the original signature for unit tests.
        floor_mat = (np.asarray(S_min, dtype=float)
                     + cfg.C3_SAFETY_BUFFER_FRACTION
                     * np.asarray(S_max, dtype=float))[:, None]
    viol = np.maximum(0.0, floor_mat - S)
    if rows is not None:
        viol = viol[rows, :]                           # enforce on optimised tanks
    return float(np.sum(viol))


def c4_mass_conservation(R, S_current, Q_expected):
    """
    C4: total release each day <= total available water (storage + inflow).

    Parameters
    ----------
    R          : (N, T)  requested release
    S_current  : (N,)    today's storage
    Q_expected : (N, T)  expected inflow over the horizon (scenario mean)

    Returns
    -------
    float total violation

    Note: uses a running storage estimate that adds expected inflow and subtracts
    releases day by day, so the bound reflects water genuinely available on each day.
    """
    N, T = R.shape
    S_run = S_current.copy()
    total_viol = 0.0
    for t in range(T):
        available = np.sum(S_run) + np.sum(Q_expected[:, t])
        released = np.sum(R[:, t])
        total_viol += max(0.0, released - available)
        # advance the running storage estimate (net of this day's release/inflow)
        S_run = S_run + Q_expected[:, t] - R[:, t]
        S_run = np.maximum(S_run, 0.0)
    return float(total_viol)


def c5_non_negativity(R, sim_result, rows=None):
    """
    C5: explicit non-negativity of releases and storage (defence in depth).
    R is already restricted to the optimised tanks by the caller; storage is
    restricted here via `rows`.

    Returns
    -------
    float total violation
    """
    r_neg = float(np.sum(np.maximum(0.0, -R)))
    s_unclipped = _rows(sim_result["S_unclipped"], rows)
    s_neg = float(np.sum(np.maximum(0.0, -s_unclipped)))
    return r_neg + s_neg


def c6_mahaweli(R, mah_ids, R_sched, delta_m):
    """
    C6: Mahaweli-fed tanks must track the scheduled release within +/- delta_m.

    Parameters
    ----------
    R       : (N, T)   requested release
    mah_ids : list     tank indices that are Mahaweli-fed
    R_sched : (M, T)   scheduled release per Mahaweli tank
    delta_m : (M,)     allowed deviation per Mahaweli tank (based on schedule)

    Returns
    -------
    float total violation
    """
    if not mah_ids:
        return 0.0
    total_viol = 0.0
    T = R.shape[1]
    for m_idx, tank_id in enumerate(mah_ids):
        for t in range(min(T, R_sched.shape[1])):
            diff = abs(R[tank_id, t] - R_sched[m_idx, t])
            total_viol += max(0.0, diff - delta_m[m_idx])
    return float(total_viol)


def check_all_constraints(R, sim_result, state, cfg: Config = DEFAULT_CONFIG,
                          rows=None):
    """
    Evaluate all ENABLED constraints and return a violation vector.

    Parameters
    ----------
    R          : (N, T)  requested release (full cascade)
    sim_result : dict    a representative simulate_cascade result (use scenario mean
                         or the expected-inflow run; violations are near-identical
                         across scenarios for bound constraints)
    state      : dict    system state from data_loader (S_min, S_max, R_max, ...)
    cfg        : Config
    rows       : list or None  tank indices the optimiser controls. When given,
                 constraints are enforced on these tanks only; the terminal-sink
                 reservoir's bounds are monitored in the simulation rather than
                 enforced (its release is rule-fixed, not a decision variable).

    Returns
    -------
    np.ndarray of violation magnitudes for the active constraints (order fixed:
    C1, C2, C3, C4, C5, C6 - inactive ones are omitted).
    """
    violations = []

    # Restrict the release matrix / per-tank params to optimised tanks where the
    # constraint acts on releases directly.
    if rows is None:
        R_c, R_max_c, S_cur_c, Q_c = (R, state["R_max"],
                                      state["S_current"], state["Q_expected"])
    else:
        R_c = R[rows, :]
        R_max_c = state["R_max"][rows]
        S_cur_c = state["S_current"][rows]
        Q_c = state["Q_expected"][rows, :]

    if cfg.ENABLE_C1_STORAGE_BOUNDS:
        violations.append(c1_storage_bounds(
            sim_result, rows, deficit_floor=_cached_c1_floor(state, cfg)))

    if cfg.ENABLE_C2_RELEASE_BOUNDS:
        violations.append(c2_release_bounds(R_c, R_max_c))

    if cfg.ENABLE_C3_DROUGHT_FLOOR:
        violations.append(c3_drought_floor(
            sim_result, state["S_min"], state["S_max"], cfg, rows,
            floor_mat=_cached_c3_floor(state, cfg)))

    if cfg.ENABLE_C4_MASS_CONSERVE:
        violations.append(c4_mass_conservation(R_c, S_cur_c, Q_c))

    if cfg.ENABLE_C5_NON_NEGATIVITY:
        violations.append(c5_non_negativity(R_c, sim_result, rows))

    if cfg.ENABLE_C6_MAHAWELI:
        violations.append(c6_mahaweli(R, state.get("mah_ids", []),
                                      state.get("R_sched", np.zeros((0, R.shape[1]))),
                                      state.get("delta_m", np.zeros(0))))

    return np.array(violations) if violations else np.zeros(1)


def constraint_labels(cfg: Config = DEFAULT_CONFIG):
    """
    Names of the ACTIVE constraints, in the same fixed order that
    `check_all_constraints` returns violations. Used for reporting only.
    """
    catalogue = [
        (cfg.ENABLE_C1_STORAGE_BOUNDS, "C1", "storage bounds",
         "S_min <= S <= S_max"),
        (cfg.ENABLE_C2_RELEASE_BOUNDS, "C2", "release bounds",
         "0 <= R <= R_max"),
        (cfg.ENABLE_C3_DROUGHT_FLOOR, "C3", "seasonal safety floor",
         (f"S >= min(S_base, S_min + {cfg.C3_SAFETY_BUFFER_FRACTION:.0%}*S_max)"
          if getattr(cfg, "C3_BASELINE_FLOOR", True)
          else f"S >= S_min + {cfg.C3_SAFETY_BUFFER_FRACTION:.0%}*S_max")),
        (cfg.ENABLE_C4_MASS_CONSERVE, "C4", "mass conservation",
         "sum R(t) <= sum S + sum Q(t)"),
        (cfg.ENABLE_C5_NON_NEGATIVITY, "C5", "non-negativity",
         "R >= 0 and S >= 0"),
        (cfg.ENABLE_C6_MAHAWELI, "C6", "Mahaweli tracking",
         f"|R - R_sched| <= {cfg.DELTA_M_FRACTION:.0%} of schedule"),
    ]
    return [(code, name, rule) for enabled, code, name, rule in catalogue if enabled]


def reference_plan(state, cfg: Config = DEFAULT_CONFIG):
    """
    Build the DIAGNOSTIC reference release plan: meet demand exactly, capped at
    each tank's sluice capacity.

        R_ref[i, t] = min( D[i, t], R_max[i] )

    This is the plan an operator would write down with no optimisation at all,
    which makes it a useful yardstick: if the reference plan already violates a
    constraint, that constraint is the binding one and NSGA-II will have to
    trade demand satisfaction away to satisfy it.

    NOT used by the optimiser, the objectives, or any exported file.
    """
    D = np.asarray(state["D"], dtype=float)
    R_max = np.asarray(state["R_max"], dtype=float)
    return np.minimum(D, R_max[:, None])


def _reference_plan_simulation(state, cfg: Config = DEFAULT_CONFIG, R_ref=None):
    """Build the reference plan and simulate it once for diagnostic use."""
    from .state_transition import simulate_cascade   # local: keeps import graph flat

    if R_ref is None:
        R_ref = reference_plan(state, cfg)

    sim = simulate_cascade(
        state["S_current"], R_ref, state["Q_expected"],
        state["alpha"], state["E"], state["S_min"], state["S_max"],
        # Same physics as the optimiser sees, so the diagnostic matches reality.
        state["D"] if getattr(cfg, "ENABLE_CROP_CONSUMPTION", True) else None,
        storage_limited_evap=getattr(cfg, "STORAGE_LIMITED_EVAPORATION", True),
        upstream_iters=getattr(cfg, "UPSTREAM_FIXED_POINT_ITERS", 3),
    )
    rows_idx = state.get("opt_idx")
    return R_ref, sim, rows_idx


def _c3_floor_vector(state, cfg: Config = DEFAULT_CONFIG):
    """Return the seasonal safety floor for each tank."""
    return (np.asarray(state["S_min"], dtype=float)
            + cfg.C3_SAFETY_BUFFER_FRACTION * np.asarray(state["S_max"], dtype=float))


def log_c3_initial_storage_diagnostic(state, cfg: Config = DEFAULT_CONFIG,
                                      rows=None):
    """
    Print a one-time-per-day diagnostic showing whether any tank already starts
    below the C3 seasonal safety floor before optimisation begins.
    """
    from . import logging_utils as lg

    log = lg.Log(cfg)
    if not log.enabled or not getattr(cfg, "LOG_CONSTRAINTS", True):
        return None

    tank_ids = np.asarray(state["tank_ids"])
    S_current = np.asarray(state["S_current"], dtype=float)
    S_min = np.asarray(state["S_min"], dtype=float)
    S_max = np.asarray(state["S_max"], dtype=float)
    floor = _c3_floor_vector(state, cfg)

    if rows is not None:
        rows = np.asarray(rows, dtype=int)
        tank_ids = tank_ids[rows]
        S_current = S_current[rows]
        S_min = S_min[rows]
        S_max = S_max[rows]
        floor = floor[rows]

    deficit = np.maximum(0.0, floor - S_current)
    pct_capacity = np.divide(S_current, S_max,
                             out=np.zeros_like(S_current), where=S_max > 0.0) * 100.0
    below = deficit > 1e-9

    scope = "all tanks" if rows is None else "optimised tanks"
    log.note(f"C3 initial-storage diagnostic ({scope})")
    log.table(
        ["tank", "S_current", "S_min", "S_max", "C3 floor", "% cap", "below floor", "deficit"],
        [[tid,
          lg.fmt_m3(s_cur, 0),
          lg.fmt_m3(s_lo, 0),
          lg.fmt_m3(s_hi, 0),
          lg.fmt_m3(s_floor, 0),
          lg.fmt_pct(pct, 1),
          "YES" if is_below else "NO",
          lg.fmt_m3(def_val, 0)]
         for tid, s_cur, s_lo, s_hi, s_floor, pct, is_below, def_val
         in zip(tank_ids, S_current, S_min, S_max, floor, pct_capacity, below, deficit)],
        aligns=["<", ">", ">", ">", ">", ">", "<", ">"],
    )

    below_count = int(np.count_nonzero(below))
    total_deficit = float(np.sum(deficit))
    if below_count:
        worst_idx = int(np.argmax(deficit))
        worst_tank = str(tank_ids[worst_idx])
        worst_deficit = float(deficit[worst_idx])
    else:
        worst_tank = "none"
        worst_deficit = 0.0

    log.kv("tanks below C3 floor", below_count, indent=2)
    log.kv("total initial C3 deficit", lg.fmt_m3(total_deficit, 0), indent=2)
    log.kv("worst offending tank", worst_tank, indent=2)
    log.kv("largest deficit", lg.fmt_m3(worst_deficit, 0), indent=2)
    return {
        "count": below_count,
        "total_deficit": total_deficit,
        "worst_tank": worst_tank,
        "largest_deficit": worst_deficit,
    }


def log_c3_horizon_diagnostic(sim_result, state, cfg: Config = DEFAULT_CONFIG,
                              rows=None):
    """
    Print the per-tank C3 violation over the prediction horizon for the
    reference plan, sorted by total violation descending.
    """
    from . import logging_utils as lg

    log = lg.Log(cfg)
    if not log.enabled or not getattr(cfg, "LOG_CONSTRAINTS", True):
        return None

    key = ("S_unclipped" if getattr(cfg, "C3_FLOOR_ON_UNCLIPPED", True)
           else "S_clipped")
    S = np.asarray(sim_result[key], dtype=float)[:, 1:]

    tank_ids = np.asarray(state["tank_ids"])
    viol = np.maximum(0.0, _cached_c3_floor(state, cfg) - S)      # enforced
    viol_abs = np.maximum(0.0, _c3_floor_vector(state, cfg)[:, None] - S)

    if rows is not None:
        rows = np.asarray(rows, dtype=int)
        tank_ids = tank_ids[rows]
        viol = viol[rows, :]
        viol_abs = viol_abs[rows, :]

    if viol.size == 0:
        log.note("C3 horizon diagnostic: no tanks to report")
        return None

    totals = np.sum(viol, axis=1)
    totals_abs = np.sum(viol_abs, axis=1)
    total_violation = float(np.sum(totals))
    total_absolute = float(np.sum(totals_abs))

    order = np.argsort(-totals_abs, kind="stable")
    tank_ids = tank_ids[order]
    viol = viol[order, :]
    totals = totals[order]
    totals_abs = totals_abs[order]
    shares = np.divide(totals_abs, total_absolute,
                       out=np.zeros_like(totals_abs),
                       where=total_absolute > 0.0) * 100.0

    days = [f"Day {day + 1}" for day in range(viol.shape[1])]
    scope = "optimised tanks only" if rows is not None else "all tanks"
    mode = ("baseline-relative" if getattr(cfg, "C3_BASELINE_FLOOR", True)
            else "absolute")

    log.note(f"C3 horizon diagnostic ({scope}, floor = {mode})")
    log.table(
        ["tank", *days, "enforced", "absolute", "share"],
        [[tid, *[lg.fmt_m3(value, 0) for value in day_viol],
          lg.fmt_m3(total, 0), lg.fmt_m3(total_abs, 0), lg.fmt_pct(share, 1)]
         for tid, day_viol, total, total_abs, share
         in zip(tank_ids, viol, totals, totals_abs, shares)],
        aligns=["<", *([">"] * len(days)), ">", ">", ">"],
    )

    top_n = min(10, len(tank_ids))
    log.note(f"Top {top_n} tanks by ABSOLUTE reserve deficit")
    log.table(
        ["tank", "enforced", "absolute", "share"],
        [[str(tank_ids[i]), lg.fmt_m3(totals[i], 0),
          lg.fmt_m3(totals_abs[i], 0), lg.fmt_pct(shares[i], 1)]
         for i in range(top_n)],
        aligns=["<", ">", ">", ">"],
    )
    log.kv("C3 violation (enforced)", lg.fmt_m3(total_violation, 0), indent=2)
    log.kv("reserve deficit (absolute)", lg.fmt_m3(total_absolute, 0), indent=2)
    if total_absolute > 0.0 and total_violation <= 1e-9:
        log.note("reserve deficit is entirely inherited / natural - no part of "
                 "it is attributable to the release plan", indent=2)
    return {
        "tank_ids": tank_ids,
        "violations": viol,
        "totals": totals,
        "totals_absolute": totals_abs,
        "shares": shares,
        "total_violation": total_violation,
        "total_absolute": total_absolute,
    }


def feasibility_snapshot(state, cfg: Config = DEFAULT_CONFIG, R_ref=None):
    """
    Evaluate every ACTIVE constraint on the reference plan and return a list of
    per-constraint dicts. Pure diagnosis - called once per MPC day, never per
    candidate (that would fire POP_SIZE x N_GEN times and flood the terminal).

    The snapshot is taken on the same tank set the optimiser is constrained on
    (`state['opt_idx']`, i.e. the terminal sink excluded), so the diagnosis
    matches what NSGA-II actually sees.

    Returns
    -------
    list of dicts: code, name, rule, violation (float), ok (bool)
    """
    R_ref, sim, rows_idx = _reference_plan_simulation(state, cfg, R_ref)
    g = check_all_constraints(R_ref, sim, state, cfg, rows=rows_idx)
    labels = constraint_labels(cfg)

    rows = []
    for j, (code, name, rule) in enumerate(labels):
        v = float(g[j]) if j < len(g) else float("nan")
        rows.append({"code": code, "name": name, "rule": rule,
                     "violation": v, "ok": v <= 1e-9})
    return rows


def log_feasibility_snapshot(state, cfg: Config = DEFAULT_CONFIG, R_ref=None):
    """
    Print the once-per-day constraint snapshot described in
    `feasibility_snapshot`. Observability only.
    """
    from . import logging_utils as lg

    log = lg.Log(cfg)
    if not log.enabled or not getattr(cfg, "LOG_CONSTRAINTS", True):
        return None

    log_c3_initial_storage_diagnostic(state, cfg)

    if getattr(cfg, "C3_BASELINE_FLOOR", True):
        log.note("C3 floor is baseline-relative: S(t) >= min(S_base(t), "
                 f"S_min + {cfg.C3_SAFETY_BUFFER_FRACTION:.0%}*S_max)")
        log.note("the no-action plan reproduces S_base, so a feasible plan "
                 "always exists", indent=2)

    R_ref, sim, rows_idx = _reference_plan_simulation(state, cfg, R_ref)

    if rows_idx is not None and len(rows_idx) != len(state.get("tank_ids", [])):
        log.note("C3 horizon total below uses optimised tanks only to match the constraint table")
    log_c3_horizon_diagnostic(sim, state, cfg, rows=rows_idx)

    g = check_all_constraints(R_ref, sim, state, cfg, rows=rows_idx)
    labels = constraint_labels(cfg)

    rows = []
    for j, (code, name, rule) in enumerate(labels):
        v = float(g[j]) if j < len(g) else float("nan")
        rows.append({"code": code, "name": name, "rule": rule,
                     "violation": v, "ok": v <= 1e-9})

    if not rows:
        log.warn("no constraints are enabled - the search is unconstrained")
        return rows

    log.note("reference plan = min(demand, R_max), i.e. 'release exactly what is asked'")
    if state.get("sink_idx"):
        log.note("constraints evaluated on the optimised tanks only "
                 "(terminal sink monitored, not enforced)")
    log.table(
        ["", "constraint", "rule", "reference plan", "violation"],
        [[r["code"], r["name"], r["rule"],
          "satisfied" if r["ok"] else "VIOLATED",
          "-" if r["ok"] else lg.fmt_num(r["violation"], 0)]
         for r in rows],
        aligns=["<", "<", "<", "<", ">"],
    )

    binding = [r for r in rows if not r["ok"]]
    if binding:
        log.warn("binding on the reference plan: "
                 + ", ".join(f"{r['code']} ({r['name']})" for r in binding)
                 + " -> NSGA-II must trade demand away here", indent=2)
    else:
        log.note("reference plan is fully feasible - constraints are not the bottleneck")
    return rows


def num_active_constraints(cfg: Config = DEFAULT_CONFIG) -> int:
    """Count how many constraints are enabled (for pymoo n_constr)."""
    return sum([
        cfg.ENABLE_C1_STORAGE_BOUNDS,
        cfg.ENABLE_C2_RELEASE_BOUNDS,
        cfg.ENABLE_C3_DROUGHT_FLOOR,
        cfg.ENABLE_C4_MASS_CONSERVE,
        cfg.ENABLE_C5_NON_NEGATIVITY,
        cfg.ENABLE_C6_MAHAWELI,
    ])