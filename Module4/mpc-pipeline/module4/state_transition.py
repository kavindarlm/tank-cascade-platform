"""
state_transition.py
==================
The simulation engine: propagate the mass balance forward over the horizon.

Mass balance (continuity equation, Loucks & van Beek 2017), with explicit
irrigation consumption:

    consumed_i(t)   = min( R_eff_i(t), D_i(t) )          <- crops drink
    return_i(t)     = R_eff_i(t) - consumed_i(t)         <- leftover drains on
    S_i(t+1) = S_i(t) + Q_i(t) + sum_j alpha_ij return_j(t) - R_eff_i(t) - E_i(t)

where alpha_ij is the fraction of tank j's RETURN FLOW that arrives at tank i
(row = destination, column = source), computed as the matrix product alpha @ return.

--------------------------------------------------------------------------
CRITICAL FIX vs the original implementation (Bugs 1 & 3):

The original code clipped storage to [S_min, S_max] and then evaluated overflow
and constraints on the CLIPPED trajectory - so overflow (f2) and the C1 violation
were always zero, and clipping at S_min silently created water from nothing.

This version records, BEFORE clipping:
    - spill_i(t)   = max(0, S_next - S_max)          -> used by f2 and C1-high
    - deficit_i(t) = max(0, S_min - S_next)          -> used by C1-low
    - R_effective  = release actually deliverable given available water,
                     so a plan cannot release more than the tank holds
                     (preserves conservation of mass, Bug 3).

--------------------------------------------------------------------------
PHYSICAL IRRIGATION CONSUMPTION (Bug 9 fix)

PREVIOUSLY: the whole release was routed downstream (alpha @ R_eff). Demand
appeared ONLY in the objective functions, so the crops never physically drank
anything - irrigation water was scored as "delivered" and then flowed on to the
next tank as if it had never been used. Water therefore over-circulated through
the cascade: nothing removed it at the fields, so releases had to be ~3x demand
to keep every tank supplied, and the simulated storage trajectory drifted above
reality (a likely contributor to the low Module 3 cross-check agreement).

NOW: each tank's command area consumes

        consumed_i(t) = min( R_eff_i(t), D_i(t) )

and ONLY the leftover (the field drainage / return flow) is routed downstream.

Two properties matter:

  * `min` (not D) means adding consumption does NOT force full demand
    satisfaction. If the optimiser releases 60 against a demand of 100, the crops
    drink 60, the shortage is 40, and nothing cascades. The optimiser is still
    free to leave demand unmet - that trade-off still lives in f1/f4.

  * consumed water is NOT multiplied by alpha. alpha routes water travelling
    between tanks; consumed water leaves the system vertically as crop
    evapotranspiration, so no downstream routing applies to it.

Mass is conserved exactly:
    consumed + alpha*(R-consumed) + (1-alpha)*(R-consumed)  ==  R_eff

Set D=None (or cfg.ENABLE_CROP_CONSUMPTION=False) to recover the previous
behaviour for ablation studies.
--------------------------------------------------------------------------
STORAGE-LIMITED EVAPORATION + UPSTREAM CONSISTENCY

E is now capped at the water actually present:
    E_actual = min(E, max(0, S + Q + upstream - S_min))
Previously a nearly-empty tank evaporated its full potential E and landed below
S_min even at R = 0, so the C1 low-side violation was uncontrollable.

Availability was also sized from the REQUESTED upstream release while the mass
balance routed the EFFECTIVE one, crediting tanks with water their neighbours
could not send. The two are now reconciled by a short monotone fixed point that
starts conservative, so stopping early under-estimates availability rather than
inventing water.
--------------------------------------------------------------------------
"""

import numpy as np


def simulate_cascade(S_init, R, Q_scenario, alpha, E, S_min, S_max, D=None,
                     storage_limited_evap=True, upstream_iters=3):
    """
    Simulate the cascade for ONE scenario over the horizon.

    Parameters
    ----------
    S_init     : (N,)    initial storage today
    R          : (N, T)  requested release matrix (decision variable)
    Q_scenario : (N, T)  one inflow trajectory (one scenario)
    alpha      : (N, N)  network transfer coefficients (row=dest, col=source)
    E          : (N, T)  potential evaporation losses
    S_min      : (N,)    dead storage limits
    S_max      : (N,)    capacity limits
    D          : (N, T) or None. Irrigation demand. When given, each tank's crops
                 consume min(R_eff, D) and only the leftover is routed downstream.
    storage_limited_evap : bool
                 True  -> E_actual = min(E, water above dead storage)  [physical]
                 False -> E_actual = E                                 [legacy]
    upstream_iters : int
                 Fixed-point sweeps reconciling availability with the effective
                 upstream return flow. 1 reproduces the old (inconsistent)
                 behaviour up to the starting estimate; 3 is ample in practice.

    Returns
    -------
    result : dict with
        'S_clipped'   : (N, T+1)  physical storage (clipped) carried forward
        'S_unclipped' : (N, T+1)  storage before clipping (for f2 / C1 / C3)
        'spill'       : (N, T)    overflow volume each day
        'deficit'     : (N, T)    below-dead-storage volume each day
        'R_effective' : (N, T)    release actually deliverable
        'R_shortfall' : (N, T)    requested minus deliverable
        'consumed'    : (N, T)    water drunk by the command area = min(R_eff, D)
        'return_flow' : (N, T)    leftover after consumption, routed downstream
        'E_actual'    : (N, T)    evaporation actually applied  [NEW]
    """
    N, T = R.shape
    S_clipped = np.zeros((N, T + 1))
    S_unclipped = np.zeros((N, T + 1))
    spill = np.zeros((N, T))
    deficit = np.zeros((N, T))
    R_effective = np.zeros((N, T))
    R_shortfall = np.zeros((N, T))
    consumed_all = np.zeros((N, T))
    return_all = np.zeros((N, T))
    E_actual_all = np.zeros((N, T))

    S_clipped[:, 0] = S_init
    S_unclipped[:, 0] = S_init

    n_iter = max(1, int(upstream_iters))

    for t in range(T):
        S_t = S_clipped[:, t]
        R_t = R[:, t]
        Q_t = Q_scenario[:, t]
        E_t = E[:, t]
        D_t = None if D is None else D[:, t]

        # ------------------------------------------------------------------
        # (B) Availability / upstream fixed point.
        #
        # available_i = S_i + Q_i + (alpha @ return)_i - E_actual_i - S_min_i
        # return_j    = R_eff_j - min(R_eff_j, D_j)
        # R_eff_j     = min(R_j, available_j)
        #
        # The map R_eff -> R_eff is monotone non-decreasing, so starting from
        # the conservative no-upstream estimate the iterates increase toward the
        # LEAST fixed point. Stopping early therefore under-estimates rather
        # than over-estimates availability: water is never invented.
        # ------------------------------------------------------------------
        # Evaporation is limited by the water PRESENT, floored at 0 - not at
        # S_min. Dead storage is an operational limit on what can be RELEASED
        # through the sluice, not a physical floor on what the tank holds. A
        # tank below dead storage still evaporates.
        if storage_limited_evap:
            E_act = np.minimum(E_t, np.maximum(0.0, S_t + Q_t))
        else:
            E_act = E_t
        # Release availability, by contrast, IS floored at S_min: water below
        # the sluice cannot be released.
        R_eff = np.minimum(R_t, np.maximum(0.0, S_t + Q_t - E_act - S_min))

        for _ in range(n_iter):
            consumed = np.zeros(N) if D_t is None else np.minimum(R_eff, D_t)
            return_flow = R_eff - consumed
            upstream_in = alpha @ return_flow

            if storage_limited_evap:
                E_act = np.minimum(
                    E_t, np.maximum(0.0, S_t + Q_t + upstream_in))
            else:
                E_act = E_t

            available = np.maximum(
                0.0, S_t + Q_t + upstream_in - E_act - S_min)
            R_new = np.minimum(R_t, available)

            if np.allclose(R_new, R_eff, rtol=0.0, atol=1e-9):
                R_eff = R_new
                break
            R_eff = R_new

        # Recompute the routed quantities from the converged R_eff so the mass
        # balance below is exactly consistent with the availability that
        # permitted it.
        consumed = np.zeros(N) if D_t is None else np.minimum(R_eff, D_t)
        return_flow = R_eff - consumed
        upstream_in_eff = alpha @ return_flow
        if storage_limited_evap:
            E_act = np.minimum(
                E_t, np.maximum(0.0, S_t + Q_t + upstream_in_eff))
        else:
            E_act = E_t

        R_effective[:, t] = R_eff
        R_shortfall[:, t] = R_t - R_eff
        consumed_all[:, t] = consumed
        return_all[:, t] = return_flow
        E_actual_all[:, t] = E_act

        # --- Mass balance ---------------------------------------------------
        S_next = S_t + Q_t + upstream_in_eff - R_eff - E_act
        S_unclipped[:, t + 1] = S_next

        # --- Record violations BEFORE clipping -------------------------------
        spill[:, t] = np.maximum(0.0, S_next - S_max)
        deficit[:, t] = np.maximum(0.0, S_min - S_next)

        # --- Clip to the PHYSICAL envelope for the carried-forward state -----
        # Floor at 0, NOT at S_min. Clamping up to S_min fabricates water for a
        # tank that is below dead storage: Todamaduwa starts at 416 m3 against a
        # 31,967 m3 floor, so the old clamp handed it ~31,500 m3 it does not
        # have, on day 1, every day. The invented water then propagates through
        # the whole planning horizon and inflates the reported storage.
        #
        # S_min is an OPERATIONAL limit (water below the sluice cannot be
        # released) and is already enforced through `available` above. It is not
        # a physical floor on tank contents. The only physical floor is empty.
        S_clipped[:, t + 1] = np.minimum(np.maximum(S_next, 0.0), S_max)

    return {
        "S_clipped": S_clipped,
        "S_unclipped": S_unclipped,
        "spill": spill,
        "deficit": deficit,
        "R_effective": R_effective,
        "R_shortfall": R_shortfall,
        "consumed": consumed_all,
        "return_flow": return_all,
        "E_actual": E_actual_all,
    }


def simulate_all_scenarios(S_init, R, Q_all, alpha, E, S_min, S_max, D=None,
                           storage_limited_evap=True, upstream_iters=3,
                           vectorised=True):
    """
    Run the cascade for all K scenarios.

    With vectorised=True the K scenarios are advanced simultaneously on a
    leading K axis - identical mathematics, one Python loop over T instead of
    K x T, and the alpha routing done as a single einsum. This is the hot path:
    _evaluate calls it POP_SIZE x N_GEN times, so it dominates total runtime.

    Set vectorised=False for the reference implementation (identical results;
    kept for verification).

    Parameters
    ----------
    Q_all : (K, N, T)  all inflow scenarios
    D     : (N, T) or None  irrigation demand (see simulate_cascade)

    Returns
    -------
    list of length K, each the dict returned by simulate_cascade.
    """
    K = Q_all.shape[0]

    if not vectorised:
        return [simulate_cascade(S_init, R, Q_all[k], alpha, E, S_min, S_max, D,
                                 storage_limited_evap=storage_limited_evap,
                                 upstream_iters=upstream_iters)
                for k in range(K)]

    N, T = R.shape
    S_clipped = np.zeros((K, N, T + 1))
    S_unclipped = np.zeros((K, N, T + 1))
    spill = np.zeros((K, N, T))
    deficit = np.zeros((K, N, T))
    R_effective = np.zeros((K, N, T))
    consumed_all = np.zeros((K, N, T))
    return_all = np.zeros((K, N, T))
    E_actual_all = np.zeros((K, N, T))

    S_clipped[:, :, 0] = S_init[None, :]
    S_unclipped[:, :, 0] = S_init[None, :]

    Smin = S_min[None, :]
    Smax = S_max[None, :]
    n_iter = max(1, int(upstream_iters))

    for t in range(T):
        S_t = S_clipped[:, :, t]                 # (K, N)
        R_t = R[None, :, t]                      # (1, N) broadcast over K
        Q_t = Q_all[:, :, t]                     # (K, N)
        E_t = E[None, :, t]                      # (1, N)
        D_t = None if D is None else D[None, :, t]

        if storage_limited_evap:
            E_act = np.minimum(E_t, np.maximum(0.0, S_t + Q_t - Smin))
        else:
            E_act = np.broadcast_to(E_t, S_t.shape).copy()
        R_eff = np.minimum(R_t, np.maximum(0.0, S_t + Q_t - E_act - Smin))

        for _ in range(n_iter):
            consumed = np.zeros_like(R_eff) if D_t is None else np.minimum(R_eff, D_t)
            return_flow = R_eff - consumed
            # alpha (N, N) applied to every scenario at once
            upstream_in = return_flow @ alpha.T          # (K, N)

            if storage_limited_evap:
                E_act = np.minimum(E_t, np.maximum(0.0, S_t + Q_t + upstream_in - Smin))
            else:
                E_act = np.broadcast_to(E_t, S_t.shape).copy()

            available = np.maximum(0.0, S_t + Q_t + upstream_in - E_act - Smin)
            R_new = np.minimum(R_t, available)
            if np.allclose(R_new, R_eff, rtol=0.0, atol=1e-9):
                R_eff = R_new
                break
            R_eff = R_new

        consumed = np.zeros_like(R_eff) if D_t is None else np.minimum(R_eff, D_t)
        return_flow = R_eff - consumed
        upstream_in_eff = return_flow @ alpha.T
        if storage_limited_evap:
            E_act = np.minimum(E_t, np.maximum(0.0, S_t + Q_t + upstream_in_eff - Smin))
        else:
            E_act = np.broadcast_to(E_t, S_t.shape).copy()

        R_effective[:, :, t] = R_eff
        consumed_all[:, :, t] = consumed
        return_all[:, :, t] = return_flow
        E_actual_all[:, :, t] = E_act

        S_next = S_t + Q_t + upstream_in_eff - R_eff - E_act
        S_unclipped[:, :, t + 1] = S_next
        spill[:, :, t] = np.maximum(0.0, S_next - Smax)
        deficit[:, :, t] = np.maximum(0.0, Smin - S_next)
        S_clipped[:, :, t + 1] = np.minimum(np.maximum(S_next, Smin), Smax)

    R_req = np.broadcast_to(R[None, :, :], (K, N, T))
    return [{
        "S_clipped": S_clipped[k],
        "S_unclipped": S_unclipped[k],
        "spill": spill[k],
        "deficit": deficit[k],
        "R_effective": R_effective[k],
        "R_shortfall": R_req[k] - R_effective[k],
        "consumed": consumed_all[k],
        "return_flow": return_all[k],
        "E_actual": E_actual_all[k],
    } for k in range(K)]