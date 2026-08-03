"""
mpc_loop.py
==========
The top-level receding-horizon MO-MPC controller.

For each day of the irrigation season:
    1. OBSERVE  - load today's storage (Module 1) and risk (Module 3)
    2. PREDICT  - generate K rainfall-inflow scenarios (Gamma fit to history)
    3. OPTIMISE - run NSGA-II to get the Pareto front over 4 expected objectives
    4. DECIDE   - compute Option A risk weights, rank with TOPSIS, take top-3
    5. CROSS-CHECK - compare the chosen plan's simulated trajectory vs Module 3
    6. ACT      - implement ONLY the day-0 release; discard days 1..T-1
    7. SHIFT    - advance one day and repeat (the horizon slides forward)

Only day 0 is ever applied; days 1..T-1 exist to prevent myopia and are
recomputed next step with fresh observations. This receding horizon is what makes
the controller robust to forecast error (Mayne et al. 2000).
"""

import time

from dataclasses import replace as _dc_replace
from datetime import date

import numpy as np

from .config import Config, DEFAULT_CONFIG
from .data_loader import (load_static_inputs, load_module3_risk,
                          load_module3_risk_live, build_state,
                          load_current_storage, storage_from_module3_t1)
from .state_transition import simulate_cascade
from .nsga2_optimizer import run_nsga2, sink_release_rule
from .topsis import compute_risk_weights, topsis_rank
from .cross_validation import compare_with_module3, divergence_summary
from .constraints import (log_feasibility_snapshot,
                          build_baseline_trajectory,
                          c3_absolute_reserve_deficit)
from . import logging_utils as lg
from . import output as out
from . import consolidated_output as co


def advance_one_day(state, R_today, cfg: Config = DEFAULT_CONFIG):
    """
    Propagate the cascade ONE day forward under the release that was actually
    applied, and return the storage the next MPC step starts from.

    This is what closes the loop. The optimiser plans T days ahead, but only day
    0 is implemented; the remaining T-1 days are discarded and replanned. So the
    state handed to tomorrow must reflect exactly one day of physics:

        S(t+1) = S(t) + Q(t) + alpha @ R(t) - R(t) - E(t),  clipped to [S_min, S_max]

    We reuse simulate_cascade with single-column inputs rather than reimplementing
    the balance, so the propagation is identical to the one the optimiser scored
    its candidates against (including effective-release mass conservation and the
    pre-clip spill/deficit bookkeeping).

    Inflow uses Q_expected - the scenario mean - for day 0. The K sampled
    scenarios exist to make the PLAN robust; the simulated world advances on the
    expected value, which is the standard certainty-equivalent closed-loop test.

    Parameters
    ----------
    state    : dict   the state the decision was made from
    R_today  : (N,)   the day-0 release that was applied
    cfg      : Config

    Returns
    -------
    S_next : (N,) storage at the start of the next day, and the one-day
             simulate_cascade result dict, as a tuple.
    """
    R_day0 = np.asarray(R_today, dtype=float).reshape(-1, 1)     # (N, 1)
    Q_day0 = state["Q_expected"][:, :1]                          # (N, 1)
    E_day0 = state["E"][:, :1]                                   # (N, 1)
    # Day-0 demand, so the command areas consume min(R_eff, D) exactly as they
    # did inside the optimiser. Without this the advanced state would keep water
    # the crops actually drank, and the simulated trajectory would drift upward.
    D_day0 = (state["D"][:, :1]
              if getattr(cfg, "ENABLE_CROP_CONSUMPTION", True) else None)

    step = simulate_cascade(
        state["S_current"], R_day0, Q_day0,
        state["alpha"], E_day0, state["S_min"], state["S_max"], D_day0,
        storage_limited_evap=getattr(cfg, "STORAGE_LIMITED_EVAPORATION", True),
        upstream_iters=getattr(cfg, "UPSTREAM_FIXED_POINT_ITERS", 3),
    )
    # Column 0 is today's storage; column 1 is tomorrow's.
    return step["S_clipped"][:, 1].copy(), step


def run_mpc(cfg: Config = DEFAULT_CONFIG, season_length=None, verbose=True):
    """
    Run the full MPC loop for one irrigation season.

    Parameters
    ----------
    cfg           : Config
    season_length : int   number of days to run (defaults to cfg.SEASON_LENGTH)
    verbose       : bool  print per-step recommendations

    Returns
    -------
    results : dict with
        'decisions'   : list of daily day-0 decisions
        'weights_log' : list of applied weights per day
        'crosschecks' : list of (day, comparison_df)
    """
    cfg.validate()
    season_length = season_length or cfg.SEASON_LENGTH

    log = lg.Log(cfg)
    season_t0 = time.perf_counter()

    log.banner("MODULE 4  -  MO-MPC RECEDING-HORIZON RUN",
               f"season={cfg.SEASON}  days={season_length}  horizon T={cfg.T}  "
               f"K={cfg.K}  pop={cfg.POP_SIZE}  gens={cfg.N_GEN}")

    # Static inputs are loaded once; dynamic inputs (storage, risk) reload each step.
    with log.timed("loading static inputs (tank params, demand, alpha, rainfall)"):
        static = load_static_inputs(cfg)
    log.kv("cascade", f"{static['N']} tanks")
    log.kv("rainfall history", f"{static['rainfall_history'].shape[1]} days")

    # Initial storage is read ONCE, for day 0. From then on the state either
    # advances internally from yesterday's release (closed-loop simulation) or is
    # re-measured from file each day (deployment) - see RELOAD_STORAGE_EACH_DAY.
    S_current = load_current_storage(cfg)
    log.kv("storage mode",
           "reload tank_storage.csv daily (deployment)"
           if cfg.RELOAD_STORAGE_EACH_DAY else
           "advance internally from day-0 releases (closed-loop simulation)")

    all_decisions = []
    weights_log = []
    crosschecks = []

    for t_current in range(season_length):
        day_t0 = time.perf_counter()
        log.banner(f"DAY {t_current + 1} / {season_length}",
                   f"season day {t_current + 1} of {cfg.SEASON_LENGTH} "
                   f"({cfg.SEASON})   |   planning days "
                   f"{t_current + 1}-{t_current + cfg.T}")

        # --- 1. OBSERVE ---
        # (build_state also runs stage 2 PREDICT and prints its own detail lines)
        log.stage("OBSERVE", "Module 1 storage + Module 3 risk")

        # module3_risk.csv is a STATIC placeholder unless MODULE3_LIVE_FORECAST
        # is set (config.py), in which case Module 3's live daily output is
        # read fresh instead - see data_loader.load_module3_risk_live.
        module3_df = (load_module3_risk_live(cfg) if cfg.MODULE3_LIVE_FORECAST
                      else load_module3_risk(cfg))          # reloaded each step

        if cfg.MODULE3_LIVE_FORECAST:
            # Live mode: today's storage comes from Module 3's own t+1 (% of
            # capacity), not the static tank_storage.csv - see
            # data_loader.storage_from_module3_t1 for why. Takes priority
            # over RELOAD_STORAGE_EACH_DAY (live mode is always exactly one
            # day, enforced by Config.validate()).
            S_current = storage_from_module3_t1(module3_df, static, cfg)
        elif cfg.RELOAD_STORAGE_EACH_DAY:
            # Deployment: Module 1 has written a fresh measurement overnight.
            S_current = load_current_storage(cfg)
        # else: S_current is whatever yesterday's release left behind.

        state = build_state(static, module3_df, t_current, cfg,
                            S_current=S_current)

        # --- 2. PREDICT is done inside build_state (scenarios) ---

        # --- 2c. No-action baseline for the C1 / C3 floors ---------------
        # Rebuilt every day: S_base depends on today's storage and today's
        # inflow forecast. Yesterday's cached floor matrices must be dropped or
        # the constraints would be enforced against a stale baseline.
        state.pop("_c3_floor_mat", None)
        state.pop("_c1_floor_mat", None)
        state["_R_sink"] = sink_release_rule(state)
        state["S_base"] = build_baseline_trajectory(
            state, cfg, Q=state["Q_expected"], R_base=state["_R_sink"])

        # --- 2b. Feasibility snapshot (diagnostic only, once per day) ---
        log.stage("CONSTRAINTS", "reference-plan feasibility")
        log_feasibility_snapshot(state, cfg)

        # --- 3. OPTIMISE ---
        log.stage("OPTIMISE", "NSGA-II")
        c3_tier = "hard"                  # which formulation produced the plan
        with log.timed("searching the Pareto front") as t_opt:
            pareto_F, pareto_X, _ = run_nsga2(state, state["scenarios"], cfg,
                                              verbose=False)

        # --- Tier 2 fallback --------------------------------------------
        # The baseline floor makes the no-action plan feasible by construction,
        # so this should essentially never fire. It exists so a numerical edge
        # case cannot cost a simulation day: C3 is dropped from out['G'] and the
        # day is flagged as relaxed rather than skipped. C1/C2/C4/C5 still bind.
        if (pareto_F.shape[0] == 0
                and getattr(cfg, "C3_SOFT_FALLBACK", True)
                and cfg.ENABLE_C3_DROUGHT_FLOOR):
            log.warn("no feasible plan under C3 - retrying with C3 relaxed "
                     "(day will be flagged, not skipped)", indent=2)
            cfg_relaxed = _dc_replace(cfg, ENABLE_C3_DROUGHT_FLOOR=False)
            with log.timed("re-searching with C3 relaxed"):
                pareto_F, pareto_X, _ = run_nsga2(state, state["scenarios"],
                                                  cfg_relaxed, verbose=False)
            if pareto_F.shape[0] > 0:
                c3_tier = "relaxed"

        if pareto_F.shape[0] == 0:
            log.fail("NSGA-II returned no feasible solution", indent=2)
        else:
            log.kv("C3 tier", c3_tier)
            log.kv("pareto front", f"{pareto_F.shape[0]} non-dominated solutions")
            log.kv("f1 shortage range", f"{lg.fmt_m3(pareto_F[:, 0].min())} … "
                                        f"{lg.fmt_m3(pareto_F[:, 0].max())}")
            log.kv("f2 overflow range", f"{lg.fmt_m3(pareto_F[:, 1].min())} … "
                                        f"{lg.fmt_m3(pareto_F[:, 1].max())}")
            log.kv("search time", lg.fmt_dur(t_opt.elapsed))

        # --- 4. DECIDE ---
        log.stage("DECIDE", "Option A weights + TOPSIS")
        weights = compute_risk_weights(state["p_drought"], state["p_overflow"], cfg)
        strategies = topsis_rank(pareto_F, pareto_X, state, weights,
                                 top_n=3, cfg=cfg)

        if strategies:
            b = strategies[0]
            log.done(f"selected rank 1 (closeness {b['closeness']:.4f})")
            log.kv("f1 shortage", lg.fmt_m3(b["f1_shortage"]), indent=3)
            log.kv("f2 overflow", lg.fmt_m3(b["f2_overflow"]), indent=3)
            log.kv("f3 equity", lg.fmt_num(b["f3_equity"], 4), indent=3)
            log.kv("f4 water loss", lg.fmt_m3(b["f4_water_loss"]), indent=3)
            log.kv("total release", lg.fmt_m3(b["total_release"]), indent=3)
            log.kv("total consumed", lg.fmt_m3(b["total_consumed"]), indent=3)
            log.kv("total return flow", lg.fmt_m3(b["total_return_flow"]), indent=3)
            log.kv("total channel loss", lg.fmt_m3(b["total_channel_loss"]), indent=3)
            log.table(
                ["Tank", "Demand", "Release", "Consumed", "Return Flow", "Satisfaction %"],
                b["tank_table"],
                indent=3,
                aligns=["<", ">", ">", ">", ">", ">"],
            )
            log.table(
                ["Tank", "Storage Start", "Local Inflow", "Upstream Inflow", "Release", "Consumed", "Storage End"],
                b["balance_table"],
                indent=3,
                aligns=["<", ">", ">", ">", ">", ">", ">"],
            )

        # The legacy block is kept for VERBOSE_LOGGING=False runs (and for anyone
        # parsing its format); when structured logging is on it would only
        # duplicate the DECIDE stage above.
        if verbose and not log.enabled:
            out.print_recommendations(strategies, state, t_current, weights, cfg)

        if not strategies:
            log.fail(f"day {t_current + 1}: no feasible strategy - "
                     f"logged as infeasible and skipped")
            # No feasible plan; log and continue (operator escalation in practice)
            weights_log.append({
                "day": t_current + 1,
                "p_drought": round(state["p_drought"], 4),
                "p_overflow": round(state["p_overflow"], 4),
                "w_shortage": round(float(weights[0]), 4),
                "w_overflow": round(float(weights[1]), 4),
                "w_equity": round(float(weights[2]), 4),
                "w_loss": round(float(weights[3]), 4),
                "feasible": False,
                "c3_tier": "none",
            })
            # The world still moves on a day with no plan: inflow arrives and
            # evaporation happens, there is simply no release. Advance with
            # R = 0 so an infeasible day does not freeze the season.
            if not cfg.RELOAD_STORAGE_EACH_DAY:
                S_current, _ = advance_one_day(state, np.zeros(state["N"]), cfg)
                log.note("state advanced with zero release (no feasible plan)")
            continue

        best = strategies[0]

        # --- 5. CROSS-CHECK against Module 3 ---
        log.stage("CROSS-CHECK", "simulation vs Module 3 forecast")
        with log.timed("comparing trajectory shape (ratio basis)"):
            comp = compare_with_module3(state, best["R_matrix"], module3_df, cfg)
        crosschecks.append((t_current, comp))

        summary = divergence_summary(comp, cfg)
        if summary is None:
            log.warn("no Module 3 forecast columns available - cross-check skipped",
                     indent=2)
        else:
            rate = summary["direction_agreement_rate"]
            log.kv("direction agreement", f"{rate * 100:.1f}%  "
                                          f"({summary['n_tanks_compared']} tanks, "
                                          f"{summary['n_comparisons']} comparisons)")
            log.kv("mean |ratio diff|", lg.fmt_num(summary["mean_abs_ratio_diff"], 4))
            log.kv("max  |ratio diff|", lg.fmt_num(summary["max_abs_ratio_diff"], 4))
            flagged = summary["flagged_tanks"]
            if flagged:
                log.warn(f"{len(flagged)} tank(s) diverge beyond "
                         f"{summary['divergence_threshold']:.0%}: "
                         + ", ".join(str(t) for t in flagged), indent=2)
            else:
                log.note(f"no tank exceeds the "
                         f"{summary['divergence_threshold']:.0%} divergence threshold")

        # --- 6. ACT (record day-0 release only) ---
        log.stage("ACT", "day-0 release (days 1..T-1 discarded)")
        R_today = best["R_today"]
        order = np.argsort(-np.asarray(R_today))
        top_n = max(1, int(getattr(cfg, "LOG_TOP_TANKS", 3)))
        log.kv("total release today", lg.fmt_m3(float(np.sum(R_today))))
        log.kv("tanks releasing", f"{int(np.sum(np.asarray(R_today) > 1e-6))} of {state['N']}")
        log.table(
            ["tank", "release", "share"],
            [[str(state["tank_ids"][i]),
              lg.fmt_m3(R_today[i]),
              f"{(R_today[i] / np.sum(R_today) * 100 if np.sum(R_today) > 0 else 0):.1f}%"]
             for i in order[:top_n]],
        )

        all_decisions.append({
            "day": t_current,
            "tank_ids": state["tank_ids"],
            "R_today": best["R_today"],
            "strategy": best,
        })

        weights_log.append({
            "day": t_current + 1,
            "p_drought": round(state["p_drought"], 4),
            "p_overflow": round(state["p_overflow"], 4),
            "w_shortage": round(float(weights[0]), 4),
            "w_overflow": round(float(weights[1]), 4),
            "w_equity": round(float(weights[2]), 4),
            "w_loss": round(float(weights[3]), 4),
            "feasible": True,
            "c3_tier": c3_tier,
            # Shortfall against the TRUE seasonal floor, not the enforced
            # baseline-relative one. Never a constraint - reported so the
            # results table shows whether the reserve problem was solved or
            # merely made non-binding.
            "reserve_deficit_m3": round(c3_absolute_reserve_deficit(
                best["sim"], state, cfg, rows=state["opt_idx"]), 1)
            if isinstance(best, dict) and "sim" in best else None,
        })

        # --- 7. SHIFT: advance the physical state by exactly one day. ---
        # In real deployment the tanks update overnight and Module 1 re-measures
        # them (RELOAD_STORAGE_EACH_DAY=True); in simulation we propagate the
        # mass balance ourselves so tomorrow starts from what today's release
        # actually left in the tanks.
        if not cfg.RELOAD_STORAGE_EACH_DAY:
            S_before = S_current
            S_current, step = advance_one_day(state, best["R_today"], cfg)
            delta = float(np.sum(S_current) - np.sum(S_before))
            log.stage("SHIFT", "one-day state propagation")
            log.kv("storage today", lg.fmt_m3(float(np.sum(S_before))))
            log.kv("storage tomorrow", lg.fmt_m3(float(np.sum(S_current))))
            log.kv("net change", f"{'+' if delta >= 0 else '−'}"
                                 f"{lg.fmt_m3(abs(delta))}")
            spilled = float(np.sum(step["spill"]))
            if spilled > 0:
                log.warn(f"{lg.fmt_m3(spilled)} spilled over S_max overnight", indent=2)

        day_elapsed = time.perf_counter() - day_t0
        remaining = (season_length - t_current - 1) * day_elapsed
        log.done(f"day {t_current + 1} complete", day_elapsed)
        if remaining > 0:
            log.note(f"~{lg.fmt_dur(remaining)} remaining at this pace "
                     f"({season_length - t_current - 1} days to go)")

    results = {
        "decisions": all_decisions,
        "weights_log": weights_log,
        "crosschecks": crosschecks,
    }
    results["runtime_seconds"] = time.perf_counter() - season_t0
    log_season_summary(results, season_length, cfg)
    return results


def log_season_summary(results, season_length, cfg: Config = DEFAULT_CONFIG, paths=None):
    """
    Closing report for a season run: how many days produced a feasible plan, the
    total wall-clock cost, and (once save_all has run) where the CSVs landed.
    """
    log = lg.Log(cfg)
    if not log.enabled:
        return

    wl = results["weights_log"]
    n_feasible = sum(1 for w in wl if w.get("feasible"))
    runtime = results.get("runtime_seconds", 0.0)

    log.banner("SEASON SUMMARY")
    log.kv("days simulated", f"{season_length}")
    log.kv("feasible decisions", f"{n_feasible} / {len(wl)} "
                                 f"({(n_feasible / len(wl) * 100) if wl else 0:.1f}%)")
    log.kv("infeasible days", f"{len(wl) - n_feasible}")
    n_hard = sum(1 for w in wl if w.get("c3_tier") == "hard")
    n_relaxed = sum(1 for w in wl if w.get("c3_tier") == "relaxed")
    log.kv("C3 hard / relaxed", f"{n_hard} / {n_relaxed}")
    rd = [w["reserve_deficit_m3"] for w in wl
          if w.get("reserve_deficit_m3") is not None]
    if rd:
        log.kv("mean reserve deficit", lg.fmt_m3(float(np.mean(rd))))
    log.kv("cross-checks stored", f"{len(results['crosschecks'])}")
    log.kv("total runtime", lg.fmt_dur(runtime))
    if season_length:
        log.kv("mean per day", lg.fmt_dur(runtime / season_length))

    if results["decisions"]:
        totals = [float(np.sum(d["R_today"])) for d in results["decisions"]]
        log.kv("mean daily release", lg.fmt_m3(float(np.mean(totals))))
        log.kv("season release total", lg.fmt_m3(float(np.sum(totals))))

    if paths:
        log.substep("output files")
        for name, p in paths.items():
            log.kv(name, p if p else "(nothing written)", indent=3)


def save_all(results, cfg: Config = DEFAULT_CONFIG):
    """
    Persist all MPC outputs to CSV and return the written paths.

    Always overwrites the flat cfg.OUTPUT_DIR/{mpc_decisions,
    topsis_weights_log, module3_crosscheck}.csv - a "latest run" snapshot,
    as before this function grew a second mode.

    Additionally, when cfg.CONSOLIDATED_OUTPUT is True (the Config default),
    ALSO appends this run as dated rows onto run-wide
    cfg.OUTPUT_DIR/consolidated_*.csv files - reusing
    consolidated_output.append_day() exactly as simulation_runner.py already
    does for the full-season path, just keyed by cfg.RUN_DATE (or real
    "today" if unset) instead of a simulated day offset. This is what makes
    repeated main.py runs (e.g. one per day in live/MODULE3_LIVE_FORECAST
    mode) build up a per-day history instead of each run overwriting the
    last one with no trace.
    """
    log = lg.Log(cfg)
    paths = {}
    with log.timed("writing CSV outputs"):
        paths["decisions"] = out.save_decisions(results["decisions"], cfg)
        paths["weights"] = out.save_weights_log(results["weights_log"], cfg)
        paths["crosscheck"] = out.save_crosscheck(results["crosschecks"], cfg)
        if cfg.CONSOLIDATED_OUTPUT:
            run_date = date.fromisoformat(cfg.RUN_DATE) if cfg.RUN_DATE else date.today()
            consolidated_paths = co.append_day(cfg.OUTPUT_DIR, run_date, results, cfg)
            for name, p in consolidated_paths.items():
                paths[f"consolidated_{name}"] = p
    for name, p in paths.items():
        log.kv(name, p if p else "(nothing written)")
    return paths