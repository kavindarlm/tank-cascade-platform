"""
audit_day.py  (v2)
==================
Physical-realism audit of one MPC day.

    python audit_day.py                     # audits whatever is in cfg.DATA_DIR
    python audit_day.py --data _sim_scratch # explicit
    python audit_day.py --day 3             # t_current offset into the season

IMPORTANT - which day am I auditing?
    This reads tank_storage.csv and the Module 3 CSVs from --data. run_simulation
    OVERWRITES those files for each day it processes, so after a 15-day run they
    hold the LAST day, not the first. The banner below prints the total storage
    it loaded - compare it against the "Storage Start" column of the day you
    think you are auditing. If they differ, you are auditing a different day.

    To audit day 1, re-stage it:
        python run_simulation.py --start 2025-03-01 --end 2025-03-01

Checks:
  1. PHANTOM WATER   - storage fabricated by S_clipped clamping a
                       below-dead-storage tank up to S_min
  2. SEASONAL TREND  - Yala should EMPTY a rain-fed cascade
  3. INFLOW SANITY   - inflow vs live storage, and the Gamma tail
  4. SPILL PRESSURE  - which tanks force over-release, and how much of that
                       is driven by the scenario tail rather than the forecast
"""
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=None, help="directory holding the day's CSVs")
parser.add_argument("--day", type=int, default=0, help="t_current offset")
args = parser.parse_args()

from dataclasses import replace
from module4.config import DEFAULT_CONFIG
from module4.data_loader import (load_static_inputs, load_current_storage,
                                 load_module3_risk, build_state)
from module4.nsga2_optimizer import sink_release_rule
from module4.constraints import (build_baseline_trajectory, c3_effective_floor,
                                 c1_deficit_floor)
from module4.state_transition import simulate_cascade, simulate_all_scenarios

cfg = DEFAULT_CONFIG if args.data is None else replace(DEFAULT_CONFIG, DATA_DIR=args.data)

static = load_static_inputs(cfg)
state = build_state(static, load_module3_risk(cfg), args.day, cfg,
                    S_current=load_current_storage(cfg))
state["_R_sink"] = sink_release_rule(state)
state["S_base"] = build_baseline_trajectory(state, cfg, Q=state["Q_expected"],
                                            R_base=state["_R_sink"])
state["_c3_floor_mat"] = c3_effective_floor(state, cfg)
state["_c1_floor_mat"] = c1_deficit_floor(state, cfg)

N, T = int(state["N"]), int(state["T"])
ids = list(state["tank_ids"])
S0, S_min, S_max = state["S_current"], state["S_min"], state["S_max"]
opt = np.array(state["opt_idx"])

print("\n" + "=" * 78)
print("WHICH DAY IS THIS?")
print("=" * 78)
print(f"   data directory        : {cfg.DATA_DIR}")
print(f"   t_current offset      : {args.day}")
print(f"   total storage loaded  : {float(S0.sum()):,.0f} m3")
print(f"   village storage       : {float(S0[opt].sum()):,.0f} m3")
print("   -> cross-check this against the 'Storage Start' column of the day")
print("      you intended to audit. If it differs, run_simulation has already")
print("      overwritten the scratch files with a later day.")

# ------------------------------------------------------------------ 1
print("\n" + "=" * 78)
print("1. PHANTOM WATER")
print("=" * 78)
zero = np.zeros((N, T)) + state["_R_sink"]
sim0 = simulate_cascade(S0, zero, state["Q_expected"], state["alpha"],
                        state["E"], S_min, S_max,
                        state["D"] if cfg.ENABLE_CROP_CONSUMPTION else None,
                        storage_limited_evap=True, upstream_iters=3)
phantom = np.maximum(0.0, sim0["S_clipped"][:, 1:] - sim0["S_unclipped"][:, 1:]).sum(axis=1)
below = S0 < S_min
near = np.isclose(S0, S_min, rtol=1e-3)

if phantom.sum() > 1.0:
    print(f"   {'tank':<26}{'S_current':>13}{'S_min':>13}{'phantom m3':>14}")
    print("   " + "-" * 66)
    for i in np.argsort(-phantom):
        if phantom[i] <= 1.0:
            break
        print(f"   {ids[i]:<26}{S0[i]:>13,.0f}{S_min[i]:>13,.0f}{phantom[i]:>14,.0f}")
    print(f"\n   total phantom water : {phantom.sum():,.0f} m3 over {T} days")
else:
    print("   No tank starts below dead storage on THIS day.")
    print()
    print("   Read this carefully - it is not a clean bill of health. The clamp")
    print("   fabricates water only on the day a tank sits below S_min. Once")
    print("   that day has been simulated, the tank has been LIFTED to S_min and")
    print("   every later day reads as legitimately above dead storage. The")
    print("   fabrication is now laundered into the state.")
    print()
    print("   To test day 1 specifically:")
    print("       python run_simulation.py --start 2025-03-01 --end 2025-03-01")
    print("       python audit_day.py")

if near.any():
    print(f"\n   tanks sitting within 0.1% of S_min ({int(near.sum())}) - the")
    print("   fingerprint of a previous clamp:")
    for i in np.where(near)[0]:
        print(f"       {ids[i]:<26}{S0[i]:>13,.0f}  (S_min {S_min[i]:,.0f})")

# ------------------------------------------------------------------ 2
print("\n" + "=" * 78)
print("2. SEASONAL TREND  (Yala: a rain-fed cascade should be EMPTYING)")
print("=" * 78)
Su = sim0["S_unclipped"]
start, end = float(S0[opt].sum()), float(Su[opt, -1].sum())
print(f"   village storage, day 0 : {start:>16,.0f} m3")
print(f"   village storage, day {T} : {end:>16,.0f} m3   (no-action baseline)")
print(f"   change over {T} days     : {end - start:>+16,.0f} m3  ({(end/start-1)*100:+.1f}%)")
if end > start:
    wk = (end / start) ** (7.0 / T) - 1.0
    print(f"   -> FILLING at {wk*100:+.1f}%/week -> {((1+wk)**15-1)*100:+.0f}% over a 105-day Yala")

# ------------------------------------------------------------------ 3
print("\n" + "=" * 78)
print("3. INFLOW SANITY")
print("=" * 78)
Q = state["Q_expected"]
live = float((S_max[opt] - S_min[opt]).sum())
qd = float(Q[opt, :].sum()) / T
sc = state["scenarios"]
print(f"   expected inflow        : {qd:>14,.0f} m3/day into {len(opt)} village tanks")
print(f"   as % of live storage   : {qd/live*100:>14.2f} % per day")
print(f"   scenario max / mean    : {sc.max():>14,.0f} / {sc.mean():,.0f} "
      f"= {sc.max()/max(sc.mean(),1e-9):,.0f}x")

# ------------------------------------------------------------------ 4
print("\n" + "=" * 78)
print("4. SPILL PRESSURE  (what forces the optimiser to over-release)")
print("=" * 78)
print("   Spill under ZERO release. Expected column uses Q_expected (what the")
print("   constraints see); scenario column is the mean over K draws (what f2")
print("   sees). If scenario >> expected, the Gamma tail is driving decisions.")
print()
sims = simulate_all_scenarios(S0, zero, sc, state["alpha"], state["E"],
                              S_min, S_max,
                              state["D"] if cfg.ENABLE_CROP_CONSUMPTION else None)
sp_exp = sim0["spill"].sum(axis=1)
sp_sc = np.mean([s["spill"].sum(axis=1) for s in sims], axis=0)
tot_e, tot_s = float(sp_exp[opt].sum()), float(sp_sc[opt].sum())
print(f"   {'tank':<26}{'expected':>14}{'scenario mean':>16}{'ratio':>9}")
print("   " + "-" * 65)
for i in opt[np.argsort(-sp_sc[opt])][:8]:
    if sp_sc[i] <= 1.0:
        break
    r = sp_sc[i] / sp_exp[i] if sp_exp[i] > 1.0 else float("inf")
    rs = f"{r:,.0f}x" if np.isfinite(r) else "  inf"
    print(f"   {ids[i]:<26}{sp_exp[i]:>14,.0f}{sp_sc[i]:>16,.0f}{rs:>9}")
print(f"\n   village total   expected {tot_e:,.0f}   scenario mean {tot_s:,.0f}")
if tot_e > 1.0:
    print(f"   scenario/expected ratio : {tot_s/tot_e:,.1f}x")
    if tot_s / tot_e > 3.0:
        print("   -> f2 is dominated by scenarios the forecast does not predict.")
        print("      The optimiser is buying insurance against the Gamma tail,")
        print("      and paying for it in f4 (water loss) and f1 (shortage).")
elif tot_s > 1.0:
    print("   -> the forecast predicts NO spill at all, yet the scenarios do.")
    print("      Every m3 of spill-avoidance release is tail-driven.")
print()