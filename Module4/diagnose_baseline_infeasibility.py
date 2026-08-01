"""
diagnose_baseline_infeasibility.py
====================================
Reconstructs the baseline policy's release plan for ONE date and prints the
exact per-constraint violation (C1..C5), so "feasible: False" can be
attributed to a specific constraint rather than left as a single bit.

Reuses baseline_comparator.baseline_release_plan() directly - not a
reimplementation - so this reproduces EXACTLY what that script computed for
the same date (scenario generation is seeded via cfg.SCENARIO_SEED, so
rebuilding state for the same date is deterministic and reproduces the same
Q_expected/scenarios the original run used).

USAGE
-----
    python diagnose_baseline_infeasibility.py 2025-06-01 \\
        --season yala --duration 135 \\
        --forecast "data/module4_forecasts_mar_aug_2025.csv"
"""
import argparse
import datetime

import numpy as np

from module4.config import Config
from module4 import data_loader as dl
from module4 import simulation_data as sd
from module4 import simulation_runner as sr
from module4 import state_transition as st
from module4 import constraints as ct
from module4.nsga2_optimizer import sink_release_rule
from baseline_comparator import baseline_release_plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", help="YYYY-MM-DD - the date to diagnose")
    ap.add_argument("--season-start", required=True,
                    help="YYYY-MM-DD - the FIRST day of the original run "
                         "(e.g. 2025-03-01 for the 2025 season), used to "
                         "compute the correct demand-slice offset. This must "
                         "match --start from the original baseline_comparator "
                         "/ run_simulation.py invocation exactly, or the "
                         "demand window will not match what was actually used.")
    ap.add_argument("--season", default="yala")
    ap.add_argument("--duration", type=int, required=True)
    ap.add_argument("--forecast", required=True)
    args = ap.parse_args()

    sim_date = datetime.date.fromisoformat(args.date)
    season_start = datetime.date.fromisoformat(args.season_start)
    day_offset = (sim_date - season_start).days
    if day_offset < 0:
        raise SystemExit(f"{args.date} is before --season-start {args.season_start}")

    names = sd.load_tank_index_map()
    fc = sd.load_forecast_dataset(path=args.forecast, tank_names=names)
    day_rows = sd.day_slice(fc, sim_date, n_tanks=len(names)).loc[names]

    cfg_season = sr._default_season_config(args.season, args.duration)

    import pandas as pd
    tank_params = pd.read_csv("data/tank_params.csv").set_index("tank_id")
    S_max_by_name = tank_params["S_max"]

    S_current = sd.storage_pct_to_volume(day_rows["t+1"], S_max_by_name)
    module3_frame = sd.build_module3_frame(day_rows)

    from module4.demand_generator import demand_window
    demand_m3, _ = sd.generate_season_demand(names, cfg_season)

    import tempfile, os, shutil
    static_dir = tempfile.mkdtemp()
    sd.stage_static_files(static_dir)
    sd.write_day_inputs(static_dir, names, S_current,
                        demand_window(demand_m3, day_offset, cfg_season.T),
                        module3_frame, cfg_season)

    day_cfg = Config(DATA_DIR=static_dir, RELOAD_STORAGE_EACH_DAY=True,
                     SEASON=args.season, SEASON_LENGTH=1,
                     PADDY_DURATION_DAYS=args.duration, VERBOSE_LOGGING=False)
    day_cfg.validate()

    static = dl.load_static_inputs(day_cfg)
    module3_df = dl.load_module3_risk(day_cfg)
    S_loaded = dl.load_current_storage(day_cfg)
    state = dl.build_state(static, module3_df, 0, day_cfg, S_current=S_loaded)

    R_sink = sink_release_rule(state, day_cfg)
    R_opt = baseline_release_plan(state, day_cfg)
    R_full = R_sink.copy()
    R_full[state["opt_idx"], :] = R_opt[state["opt_idx"], :]

    D_sim = state["D"] if day_cfg.ENABLE_CROP_CONSUMPTION else None
    sim = st.simulate_cascade(
        state["S_current"], R_full, state["Q_expected"], state["alpha"],
        state["E"], state["S_min"], state["S_max"], D_sim,
        storage_limited_evap=day_cfg.STORAGE_LIMITED_EVAPORATION,
        upstream_iters=day_cfg.UPSTREAM_FIXED_POINT_ITERS)

    g = ct.check_all_constraints(R_full, sim, state, day_cfg, rows=state["opt_idx"])
    labels = ct.constraint_labels(day_cfg)

    print(f"\nBASELINE constraint vector for {sim_date}:")
    print(f"{'code':<6}{'name':<20}{'violation':>16}")
    for (code, name, _), v in zip(labels, g):
        flag = "  <-- VIOLATED" if v > 1e-6 else ""
        print(f"{code:<6}{name:<20}{v:>16,.1f}{flag}")

    # ---- decompose C1 into spill vs deficit ------------------------------
    # C1's scalar bundles BOTH sides (spill above S_max, deficit below S_min)
    # into one number, AND - because C1_BASELINE_DEFICIT is active by default
    # (see constraints.c1_deficit_floor) - the low side is measured against
    # min(S_min, S_base), not literal S_min. sim['deficit'] from the raw
    # simulator is ALWAYS against literal S_min, so printing it directly
    # will NOT reconcile with C1's own reported violation whenever a tank's
    # no-action baseline was already below S_min coming into the day - the
    # baseline-relative floor correctly excludes that inherited portion. Both
    # numbers are printed below, clearly labelled, so the two are never
    # confused for each other.
    from module4.constraints import c1_deficit_floor, build_baseline_trajectory

    opt = state["opt_idx"]
    spill = sim["spill"][opt, :]
    total_spill = float(spill.sum())

    S_unclipped = sim["S_unclipped"][:, 1:]
    deficit_absolute = np.maximum(0.0, state["S_min"][:, None] - S_unclipped)[opt, :]
    total_deficit_absolute = float(deficit_absolute.sum())

    if "S_base" not in state:
        state["S_base"] = build_baseline_trajectory(
            state, day_cfg, Q=state["Q_expected"], R_base=R_sink)
    floor_enforced = c1_deficit_floor(state, day_cfg)
    deficit_enforced = np.maximum(0.0, floor_enforced - S_unclipped)[opt, :]
    total_deficit_enforced = float(deficit_enforced.sum())

    print(f"\nC1 decomposition (optimised tanks only):")
    print(f"  spill (above S_max)                    : {total_spill:>14,.1f} m3")
    print(f"  deficit, ENFORCED (vs min(S_min,S_base)): {total_deficit_enforced:>14,.1f} m3"
         f"   <- should equal C1's reported violation above (spill=0 here)")
    print(f"  deficit, ABSOLUTE (vs literal S_min)    : {total_deficit_absolute:>14,.1f} m3"
         f"   <- includes shortfall the tank already had before today's plan")
    print(f"  inherited (absolute - enforced)         : "
         f"{total_deficit_absolute - total_deficit_enforced:>14,.1f} m3"
         f"   <- NOT attributable to the baseline policy's decisions")

    if total_deficit_enforced > 1e-6:
        by_tank = deficit_enforced.sum(axis=1)
        worst = int(np.argmax(by_tank))
        print(f"  worst ENFORCED deficit tank            : "
             f"{state['tank_ids'][opt[worst]]}  ({by_tank[worst]:,.1f} m3)")
    if total_spill > 1e-6:
        by_tank = spill.sum(axis=1)
        worst = int(np.argmax(by_tank))
        print(f"  worst spilling tank                    : "
             f"{state['tank_ids'][opt[worst]]}  ({by_tank[worst]:,.1f} m3)")

    # Extra: how many optimised tanks are AT exactly S_min (dead storage)
    # after this plan - the direct fingerprint of the C3 hypothesis.
    S_next = sim["S_unclipped"][:, 1]
    S_min = state["S_min"]
    at_floor = np.isclose(S_next[state["opt_idx"]], S_min[state["opt_idx"]],
                          rtol=0, atol=1.0)
    print(f"\noptimised tanks at bare dead storage after this plan: "
         f"{int(at_floor.sum())} of {len(state['opt_idx'])}")

    import shutil
    shutil.rmtree(static_dir, ignore_errors=True)


if __name__ == "__main__":
    main()