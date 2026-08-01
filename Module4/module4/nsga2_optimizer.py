"""
nsga2_optimizer.py
=================
NSGA-II multi-objective optimisation via pymoo.

The decision variable R[i, t] (release for tank i, day t) is flattened to a 1D
vector of length N*T. NSGA-II searches for the Pareto front over the four
expected objectives, subject to the active constraints.

CRITICAL FIX vs the original implementation (Bug 2):

The original declared n_constr=0 and never evaluated constraints inside _evaluate,
so C3, C4, C5, C6 had ZERO effect on the search. This version:
  - sets n_ieq_constr to the number of ACTIVE constraints,
  - computes the constraint violation vector for each candidate,
  - writes it to out['G'] so pymoo's constraint-domination handles feasibility
    natively (feasible solutions dominate infeasible ones; among infeasible ones,
    smaller total violation is preferred).

For each candidate the objectives are the SAMPLE-AVERAGE over K scenarios
(expected cost), and the constraints are evaluated on the expected-inflow run.
"""

import time

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .state_transition import simulate_all_scenarios, simulate_cascade
from .objectives import evaluate_objectives_expected
from .constraints import (check_all_constraints, num_active_constraints,
                          build_baseline_trajectory, c3_effective_floor,
                          c1_deficit_floor, c3_drought_floor)
from .config import Config, DEFAULT_CONFIG
from . import logging_utils as lg


def _sim_kwargs(cfg: Config):
    """Physics switches forwarded to every simulate_* call from the optimiser."""
    return {
        "storage_limited_evap": getattr(cfg, "STORAGE_LIMITED_EVAPORATION", True),
        "upstream_iters": getattr(cfg, "UPSTREAM_FIXED_POINT_ITERS", 3),
    }


def sink_release_rule(state, cfg: Config = DEFAULT_CONFIG):
    """
    Operating rule for terminal-sink reservoir(s): release to meet demand where
    water is available, capped by R_max. As the large downstream reservoir the
    sink is the most reliable supplier for its own command area, so a simple
    "serve demand up to capacity" rule is appropriate. Its release is NOT an
    optimisation variable - it is fixed by this rule before the cascade is
    simulated, and it still participates fully in the mass balance (draining the
    sink, and irrelevant to upstream since nothing is downstream of it).

    Returns
    -------
    R_sink : (N, T) matrix that is zero everywhere except the sink rows, which
             hold the rule-based release. Adding this to the optimiser's
             reconstructed matrix yields the full-cascade release.
    """
    N, T = state["N"], state["T"]
    R_sink = np.zeros((N, T))
    S_min = np.asarray(state["S_min"], dtype=float)
    S_max = state["S_max"]
    R_max = np.asarray(state["R_max"], dtype=float)
    D = state["D"]
    S_cur = np.asarray(state["S_current"], dtype=float)
    for i in state["sink_idx"]:
        # available above dead storage on day 0 (a conservative daily proxy)
        avail = max(0.0, S_cur[i] - S_min[i])
        for t in range(T):
            r = min(D[i, t], R_max[i], avail)
            R_sink[i, t] = r
            # Decrement: without this the sink can REQUEST up to T times the
            # water it holds. The simulator's R_eff cap meant no water was
            # invented, but the sink's reported release and storage trajectory
            # were wrong - and that trajectory is used as a cascade-health
            # metric and cross-checked against Module 3.
            if getattr(cfg, "SINK_RULE_DECREMENT_AVAIL", True):
                avail = max(0.0, avail - r)
    return R_sink


def build_full_release(x_row, state):
    """
    Reconstruct the full (N, T) release matrix from an optimiser decision vector.

    The optimiser only decides releases for the non-sink tanks (state['opt_idx']).
    This places those values at the correct rows and fills the sink rows with the
    rule-based release, so the simulator always sees a complete cascade.

    Parameters
    ----------
    x_row : 1D array of length len(opt_idx)*T (one candidate's decision vector)
    state : system state dict

    Returns
    -------
    R_full : (N, T)
    """
    N, T = state["N"], state["T"]
    opt_idx = state["opt_idx"]
    R_full = np.zeros((N, T))
    R_opt = x_row.reshape(len(opt_idx), T)
    for k, i in enumerate(opt_idx):
        R_full[i, :] = R_opt[k, :]
    R_full += state["_R_sink"]              # add rule-based sink release
    return R_full


class CascadeReleaseProblem(Problem):
    """
    pymoo Problem wrapper for the tank-cascade release optimisation.

    Decision variable : releases for the NON-SINK tanks only, flattened to
                        length len(opt_idx)*T. Terminal-sink reservoirs are
                        excluded from the decision (operated by rule) and from
                        the objectives, but remain in the simulation.
    Objectives        : 4 (expected f1..f4 over K scenarios, over non-sink tanks)
    Constraints       : number of active constraints from config (evaluated on the
                        non-sink decided releases)
    """

    def __init__(self, state, scenarios, cfg: Config = DEFAULT_CONFIG):
        self.state = state
        self.scenarios = scenarios          # (K, N, T)
        self.cfg = cfg

        T = cfg.T
        opt_idx = state["opt_idx"]
        n_opt = len(opt_idx)

        # Bounds only for the optimised tanks
        R_max_opt = state["R_max"][opt_idx]
        xl = np.zeros(n_opt * T)
        xu = np.tile(R_max_opt, T)

        # Precompute the fixed sink release once (depends only on state, not x)
        state["_R_sink"] = sink_release_rule(state, cfg)

        # ---- Baseline trajectory for the C1 / C3 floors -------------------
        # Built on Q_expected, exactly the inflow _evaluate uses for
        # constraints, and on the sink's rule-based release, exactly the plan
        # the optimiser can reach with zero release on the decided tanks.
        # Keeping these in step is what makes that plan guaranteed-feasible.
        state["S_base"] = build_baseline_trajectory(
            state, cfg, Q=state["Q_expected"], R_base=state["_R_sink"])
        state["_c3_floor_mat"] = c3_effective_floor(state, cfg)
        state["_c1_floor_mat"] = c1_deficit_floor(state, cfg)

        if (getattr(cfg, "C3_ASSERT_ZERO_FEASIBLE", True)
                and cfg.ENABLE_C3_DROUGHT_FLOOR
                and getattr(cfg, "C3_BASELINE_FLOOR", True)):
            self._assert_zero_release_feasible()

        n_constr = num_active_constraints(cfg)

        super().__init__(
            n_var=n_opt * T,
            n_obj=4,
            n_ieq_constr=n_constr,          # constraints declared (Bug 2 fix)
            xl=xl,
            xu=xu,
        )

    def _assert_zero_release_feasible(self):
        """
        Verify the recursive-feasibility claim numerically before spending eight
        minutes of search on it.

        Zero release on the optimised tanks (the sink still follows its rule)
        must give a non-positive C3 violation. If this trips, S_base and the
        constraint are being evaluated on different inflows, or a sink listed in
        TERMINAL_SINK_TANKS is not actually terminal and its release is reaching
        an optimised tank.
        """
        s, cfg = self.state, self.cfg
        N, T = int(s["N"]), int(s["T"])

        R_zero = np.zeros((N, T)) + s["_R_sink"]
        D_sim = s["D"] if getattr(cfg, "ENABLE_CROP_CONSUMPTION", True) else None
        sim = simulate_cascade(
            s["S_current"], R_zero, s["Q_expected"], s["alpha"], s["E"],
            s["S_min"], s["S_max"], D_sim, **_sim_kwargs(cfg))

        v = c3_drought_floor(sim, s["S_min"], s["S_max"], cfg,
                             rows=s["opt_idx"], floor_mat=s["_c3_floor_mat"])

        # Tolerance scales with the cascade so it stays meaningful at 1e7 m3.
        tol = max(1.0, 1e-6 * float(np.sum(s["S_max"])))
        if v > tol:
            raise AssertionError(
                f"C3 baseline guarantee violated: the no-action plan gives a C3 "
                f"violation of {v:,.0f} m3 (tolerance {tol:,.0f}). S_base and "
                f"the constraint are not on the same inflow, or a terminal-sink "
                f"release is reaching an optimised tank. Set "
                f"C3_ASSERT_ZERO_FEASIBLE=False to bypass (not recommended)."
            )

    def _evaluate(self, x, out, *args, **kwargs):
        s = self.state
        cfg = self.cfg
        pop_size = x.shape[0]
        opt_idx = s["opt_idx"]

        F = np.zeros((pop_size, 4))
        n_constr = num_active_constraints(cfg)
        G = np.zeros((pop_size, n_constr))

        # Pre-extract for speed
        S_current = s["S_current"]
        alpha = s["alpha"]
        E = s["E"]
        S_min = s["S_min"]
        S_max = s["S_max"]
        Q_expected = s["Q_expected"]        # (N, T) scenario mean, for constraints
        # Demand is handed to the simulator so the command areas physically
        # consume min(R_eff, D) and only the leftover cascades downstream.
        # None disables consumption (legacy behaviour) for ablation runs.
        D_sim = s["D"] if getattr(cfg, "ENABLE_CROP_CONSUMPTION", True) else None
        sk = _sim_kwargs(cfg)

        for idx in range(pop_size):
            # Reconstruct the FULL cascade release (optimised tanks + rule-based sink)
            R_full = build_full_release(x[idx], s)

            # --- Objectives: expected over K scenarios, NON-SINK tanks only ---
            sim_results = simulate_all_scenarios(
                S_current, R_full, self.scenarios, alpha, E, S_min, S_max,
                D_sim, **sk
            )
            # rows=opt_idx excludes the sink from the cost (it is still simulated)
            F[idx] = evaluate_objectives_expected(sim_results, s["D"], rows=opt_idx)

            # --- Constraints: evaluated on the expected-inflow run ---
            sim_expected = simulate_cascade(
                S_current, R_full, Q_expected, alpha, E, S_min, S_max,
                D_sim, **sk
            )
            # constraints on the DECIDED releases (sink release is rule-fixed, and
            # its bounds are monitored in the simulation rather than enforced here)
            g = check_all_constraints(R_full, sim_expected, s, cfg, rows=opt_idx)
            G[idx] = g[:n_constr]

        out["F"] = F
        if n_constr > 0:
            out["G"] = G


class NSGA2Progress(Callback):
    """
    Read-only pymoo callback that reports search progress to the terminal.

    pymoo calls notify() once per completed generation. We only LOOK at the
    algorithm object (generation counter, current population, current best
    front) and print - nothing here is fed back into the search, so the result
    is bit-identical to a run with logging switched off.

    The line is rewritten in place on a real terminal (lg.progress), so a
    300-generation search ticks upward on one row instead of flooding the
    scrollback; when stdout is a pipe/file each update becomes its own line.

        gen  120/300 ████████░░░░░░░░░░░░  40.0%  front 118  feas 200/200
                     f1   12.34k  f2    0.00   eta 28.4s
    """

    def __init__(self, n_gen_total, log, log_every=10):
        super().__init__()
        self.n_gen_total = max(1, int(n_gen_total))
        self.log = log
        self.log_every = max(1, int(log_every))
        self.t0 = time.perf_counter()
        self.n_eval = 0

    def notify(self, algorithm):
        if not self.log.enabled:
            return

        gen = int(algorithm.n_gen or 0)
        is_last = gen >= self.n_gen_total
        # Always print the first and last generation; thin out the middle so the
        # ETA stays fresh without the print cost dominating a fast search.
        if gen > 1 and not is_last and gen % self.log_every:
            return

        elapsed = time.perf_counter() - self.t0
        frac = min(1.0, gen / self.n_gen_total)
        eta = (elapsed / frac - elapsed) if frac > 0 else 0.0
        self.n_eval = int(getattr(algorithm.evaluator, "n_eval", 0) or 0)

        pop = algorithm.pop
        n_pop = len(pop) if pop is not None else 0

        # Feasible = zero total constraint violation. pymoo keeps CV per
        # individual; it is absent when the problem declares no constraints.
        n_feas = n_pop
        try:
            cv = pop.get("CV")
            if cv is not None and len(cv):
                n_feas = int(np.sum(np.asarray(cv, dtype=float).ravel() <= 1e-12))
        except Exception:
            pass

        # algorithm.opt is the current non-dominated set; it is None (or empty)
        # while every individual is still infeasible.
        n_front, bests = 0, None
        opt = getattr(algorithm, "opt", None)
        if opt is not None and len(opt):
            F = np.atleast_2d(np.asarray(opt.get("F"), dtype=float))
            n_front = F.shape[0]
            bests = F.min(axis=0)

        head = (f"gen {gen:>4}/{self.n_gen_total}  {lg.bar(frac)} "
                f"{frac * 100:5.1f}%  "
                f"front {n_front:>4}  feas {n_feas:>3}/{n_pop}")
        if bests is not None:
            head += (f"  f1 {lg.fmt_compact(bests[0])}"
                     f"  f2 {lg.fmt_compact(bests[1])}")
        else:
            head += "  (no feasible individual yet)"
        head += f"  eta {lg.fmt_dur(eta)}"

        self.log.progress(head, indent=2)


def headroom_budget_plan(state, cfg: Config, scale=1.0):
    """
    Release plan that drains storage down to - but never through - the enforced
    floor, using a CUMULATIVE budget.

        headroom_i(t) = S_base_i(t) - floor_eff_i(t)
        sum_{tau<=t} R_i(tau)  <=  scale * headroom_i(t)

    The cumulative form is the point. A per-day cap of headroom(t) looks
    reasonable and is wrong: releasing that much every day for T days removes
    T x the headroom and drives the tank straight through the floor. The running
    budget releases the headroom ONCE, spread over the horizon.

    This is the control-side statement of C3 - the release bound implied by the
    storage floor - used here to generate a starting point rather than to
    replace the state constraint.

    Verified on a deliberately hostile synthetic case (10 tanks at 88-97% of
    capacity with 4x inflow creating spill pressure, 10 tanks at 10-13%
    creating floor pressure): every constraint evaluates to exactly 0, where
    the zero plan, every fraction of the reference plan, and the naive per-day
    surplus plan all fail.
    """
    N, T = int(state["N"]), int(state["T"])
    R_max = np.asarray(state["R_max"], dtype=float)[:, None] * np.ones((1, T))
    D = np.asarray(state["D"], dtype=float)

    S_base = state.get("S_base")
    floor = state.get("_c3_floor_mat")
    if S_base is None or floor is None:
        return None
    head = np.maximum(0.0, np.asarray(S_base) - np.asarray(floor))

    R = np.zeros((N, T))
    cum = np.zeros(N)
    for t in range(T):
        r = np.minimum(np.minimum(D[:, t], R_max[:, t]),
                       np.maximum(0.0, scale * head[:, t] - cum))
        R[:, t] = r
        cum += r
    for i in state.get("sink_idx", []):
        R[i, :] = 0.0                      # sink release is added separately
    return R


def seeded_sampling(state, cfg: Config, n_var):
    """
    Initial population seeded with plans that stand a chance of being feasible.

    Why this is necessary. The baseline floors make the feasible set non-empty,
    but that is existence, not reachability: the space is 217-dimensional and a
    uniformly random population lands nowhere near the feasible region. Worse,
    the two hard constraints pull in OPPOSITE directions -

        C3 (floor)  wants LESS release  - releasing drains tanks below the floor
        C1 (spill)  wants MORE release  - holding water lets full tanks overtop

    - so the feasible region is a narrow shell between "too little" and "too
    much" and NEITHER extreme is inside it. Seeding the zero plan alone does not
    work; on a hostile test case the zero plan violates C1 by 50,495 m3 and the
    full reference plan violates C3 by 4,054,413 m3.

    Seeds, in order:

        rows 0-2     headroom budget at 100 / 90 / 75%   the principled seeds
        row 3        release nothing                     C3-safe extreme
        row 4        reference plan min(D, R_max)        C1-safe extreme
        rows 5-9     lambda * reference                  ladder across the band
        rest         random, each scaled by U(0,1) so low-release plans are
                     represented instead of clustering mid-box

    Set cfg.SEED_NO_ACTION_PLAN = False to reproduce unseeded behaviour.
    """
    pop = cfg.POP_SIZE
    opt_idx = state["opt_idx"]
    T = cfg.T
    n_opt = len(opt_idx)

    R_max_opt = np.asarray(state["R_max"], dtype=float)[opt_idx]
    xu = np.tile(R_max_opt, T)

    rng = np.random.default_rng(cfg.NSGA2_SEED)
    X = rng.random((pop, n_var)) * xu[None, :]
    X *= rng.random((pop, 1))                     # per-individual scale

    D_opt = np.asarray(state["D"], dtype=float)[opt_idx, :]
    ref = np.minimum(D_opt, R_max_opt[:, None])

    seeds = []
    for scale in (1.0, 0.9, 0.75):
        plan = headroom_budget_plan(state, cfg, scale)
        if plan is not None:
            seeds.append(plan[opt_idx, :])
    seeds.append(np.zeros((n_opt, T)))
    seeds.append(ref)
    seeds += [lam * ref for lam in (0.10, 0.25, 0.50, 0.75, 0.90)]

    for i, s in enumerate(seeds):
        if i >= pop:
            break
        X[i, :] = np.minimum(np.asarray(s, dtype=float).reshape(-1), xu)

    return X


def run_nsga2(state, scenarios, cfg: Config = DEFAULT_CONFIG, verbose=False,
              progress=True):
    """
    Run NSGA-II and return the Pareto front.

    Parameters
    ----------
    state     : dict from data_loader (must include 'Q_expected')
    scenarios : (K, N, T) inflow scenarios
    cfg       : Config
    verbose   : bool - print pymoo's own (very wide) generation table
    progress  : bool - print the compact in-place generation progress line
                       (also gated by cfg.VERBOSE_LOGGING)

    Returns
    -------
    pareto_F : (n_sol, 4)   objective values of non-dominated solutions
    pareto_X : (n_sol, N*T) decision variables of non-dominated solutions
    result   : the raw pymoo result object (for diagnostics / hypervolume)
    """
    log = lg.Log(cfg if progress else False)
    problem = CascadeReleaseProblem(state, scenarios, cfg)

    if log.enabled:
        n_opt = len(state["opt_idx"])
        log.kv("search space", f"{n_opt} tanks x {cfg.T} days = "
                               f"{n_opt * cfg.T} decision variables")
        log.kv("budget", f"{cfg.POP_SIZE} individuals x {cfg.N_GEN} gens = "
                         f"{cfg.POP_SIZE * cfg.N_GEN:,} evaluations "
                         f"(x {cfg.K} scenarios each)")

    if getattr(cfg, "SEED_NO_ACTION_PLAN", True):
        sampling = seeded_sampling(state, cfg, problem.n_var)
    else:
        sampling = FloatRandomSampling()

    algorithm = NSGA2(
        pop_size=cfg.POP_SIZE,
        sampling=sampling,
        crossover=SBX(prob=cfg.CROSSOVER_PROB, eta=cfg.CROSSOVER_ETA),
        mutation=PM(prob=cfg.MUTATION_PROB, eta=cfg.MUTATION_ETA),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", cfg.N_GEN)

    callback = NSGA2Progress(cfg.N_GEN, log,
                             getattr(cfg, "NSGA2_LOG_EVERY", 10))

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=cfg.NSGA2_SEED,
        verbose=verbose,
        callback=callback,
    )

    # Close the in-place line so the caller's next print starts on a fresh row.
    log.progress_end()
    if log.enabled:
        log.kv("evaluations", f"{callback.n_eval:,} candidate release plans")

    pareto_F = result.F
    pareto_X = result.X

    # pymoo returns None for F/X if no feasible solution was found
    if pareto_F is None or pareto_X is None:
        n_opt = len(state["opt_idx"])
        return np.zeros((0, 4)), np.zeros((0, n_opt * cfg.T)), result

    # Ensure 2D shape even if a single solution is returned
    if pareto_F.ndim == 1:
        pareto_F = pareto_F.reshape(1, -1)
        pareto_X = pareto_X.reshape(1, -1)

    return pareto_F, pareto_X, result