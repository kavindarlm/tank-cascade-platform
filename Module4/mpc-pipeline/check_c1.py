"""
check_c1.py
===========
Diagnoses which tanks cause the residual C1 violation at zero release.

C1 has two components:
  HIGH side  spill above S_max  -> reduced by releasing MORE (controller has authority)
  LOW side   below dead storage -> covered by baseline floor (controller has no authority)

If the 5,457 violation is entirely HIGH-SIDE (spill), then zero release is
NOT feasible for C1 regardless of the floor patch - but the seeded headroom
budget plan (which DOES release surplus from near-full tanks) WILL satisfy it.
The fix is not to make zero release feasible for C1, but to ensure at least
one seeded plan is feasible for BOTH C1 and C3.

Run:  python check_c1.py
"""
import numpy as np
import pandas as pd
import datetime
from dataclasses import replace

from module4.config import DEFAULT_CONFIG as cfg
from module4.data_loader import load_static_inputs, build_state
import module4.simulation_data as sd
from module4.nsga2_optimizer import sink_release_rule, headroom_budget_plan
from module4.constraints import (build_baseline_trajectory, c3_effective_floor,
                                 c1_deficit_floor, _cached_c1_floor,
                                 check_all_constraints, constraint_labels)
from module4.state_transition import simulate_cascade

cfg2 = replace(cfg, DATA_DIR='_sim_scratch', RUNOFF_COEFFICIENT=0.16,
               PADDY_DURATION_DAYS=135, SEASON='yala')
static = load_static_inputs(cfg2)
names  = list(static['tank_ids'])
fc     = sd.load_forecast_dataset(path='data/module4_forecasts_mar_aug_2025.csv',
                                  tank_names=names)
rows_d = sd.day_slice(fc, datetime.date(2025, 3, 1))
m3     = sd.build_module3_frame(rows_d)
S_max_s = pd.Series(static['S_max'], index=names)
S_cur   = sd.storage_pct_to_volume(rows_d['t+1'], S_max_s)
state   = build_state(static, m3, 0, cfg2, S_current=S_cur)
state['_R_sink']      = sink_release_rule(state, cfg2)
state['S_base']       = build_baseline_trajectory(state, cfg2,
                                                   Q=state['Q_expected'],
                                                   R_base=state['_R_sink'])
state['_c3_floor_mat'] = c3_effective_floor(state, cfg2)
state['_c1_floor_mat'] = c1_deficit_floor(state, cfg2)

N, T   = int(state['N']), int(state['T'])
opt    = state['opt_idx']
floor  = _cached_c1_floor(state, cfg2)

def run(R_opt, label):
    R_full = np.zeros((N, T)) + state['_R_sink']
    R_full[opt, :] = R_opt[opt, :]
    sim = simulate_cascade(state['S_current'], R_full, state['Q_expected'],
                           state['alpha'], state['E'], state['S_min'],
                           state['S_max'], state['D'],
                           storage_limited_evap=True, upstream_iters=3)
    g   = check_all_constraints(R_full, sim, state, cfg2, rows=opt)
    tot = float(g.sum())
    spill_tot   = float(np.asarray(sim['spill'])[opt,:].sum())
    deficit_tot = float(np.maximum(0.0, floor[opt,:] -
                                   np.asarray(sim['S_unclipped'])[:,1:][opt,:]).sum())
    print(f"\n{'='*60}")
    print(f"Plan: {label}")
    print(f"  Total violation : {tot:>12,.1f}")
    print(f"  C1 spill (high) : {spill_tot:>12,.1f}  <- released MORE reduces this")
    print(f"  C1 deficit (low): {deficit_tot:>12,.1f}  <- baseline floor covers this")
    print(f"  Feasible        : {tot <= 1e-6}")
    for (c,n,_),v in zip(constraint_labels(cfg2), g):
        if v > 0:
            print(f"  {c} {n:<22} {v:>10,.1f}  VIOLATED")
    return sim

# Plan 1: zero release (tests C1 high-side)
R_zero_opt = np.zeros((N, T))
run(R_zero_opt, "zero release on optimised tanks")

# Plan 2: headroom budget (should satisfy both C1 and C3)
R_hb = headroom_budget_plan(state, cfg2, scale=0.9)
if R_hb is not None:
    run(R_hb, "headroom budget 90%")
    run(headroom_budget_plan(state, cfg2, scale=1.0), "headroom budget 100%")
else:
    print("\nheadroom_budget_plan returned None - S_base or floor not in state")

# Identify the spilling tanks at zero release
R_full = np.zeros((N, T)) + state['_R_sink']
sim0 = simulate_cascade(state['S_current'], R_full, state['Q_expected'],
                        state['alpha'], state['E'], state['S_min'],
                        state['S_max'], state['D'],
                        storage_limited_evap=True, upstream_iters=3)
spill_per_tank = np.asarray(sim0['spill'])[opt, :].sum(axis=1)
print("\n\nTanks that SPILL at zero release (must release to avoid C1 high-side):")
print(f"  {'tank':<30}  {'% cap':>8}  {'spill m3':>12}")
print("  " + "-"*55)
for i in np.argsort(-spill_per_tank):
    if spill_per_tank[i] <= 0:
        break
    ti = opt[i]
    pct = state['S_current'][ti] / state['S_max'][ti] * 100
    print(f"  {names[ti]:<30}  {pct:>7.1f}%  {spill_per_tank[i]:>12,.0f}")

# ── Extra: diagnose the spilling tanks ─────────────────────────────────
print("\n\nDETAIL: Why do Kudaittikattiya and Settikulama spill when nearly empty?")
print("Check upstream routing via alpha matrix:")
alpha = state['alpha']
Q_exp = state['Q_expected']
S_cur = state['S_current']
S_max = state['S_max']
S_min = state['S_min']

for tank_name in ['Kudaittikattiya_Wewa', 'Settikulama_Wewa']:
    ti = names.index(tank_name)
    upstream_cols = np.where(alpha[ti, :] > 0)[0]
    local_q = float(Q_exp[ti, 0])
    upstream_q_day1 = float(sum(
        alpha[ti, j] * max(0, S_cur[j] - S_min[j]) for j in upstream_cols
    ))
    print(f"\n  {tank_name}")
    print(f"    S_current : {S_cur[ti]:>12,.1f}  ({S_cur[ti]/S_max[ti]*100:.1f}% of {S_max[ti]:,.0f})")
    print(f"    local Q   : {local_q:>12,.1f} m3/day")
    print(f"    upstream sources: {[names[j] for j in upstream_cols]}")
    for j in upstream_cols:
        print(f"      {names[j]}: alpha={alpha[ti,j]:.3f}, S_upstream={S_cur[j]:,.0f}")
    total_day1 = S_cur[ti] + local_q + upstream_q_day1
    print(f"    total day1 inflow+storage: {total_day1:>12,.1f}")
    print(f"    overflow by              : {max(0, total_day1 - S_max[ti]):>12,.1f}")
    if total_day1 > S_max[ti]:
        print(f"    -> CONCLUSION: upstream routing delivers more than S_max on day 1")
        print(f"       The α network was calibrated for old S_max ({62216 if 'Kudaitti' in tank_name else 231438:,.0f})")
        print(f"       New S_max ({S_max[ti]:,.0f}) is too small to absorb upstream flow")

# ── Extra: satisfaction vs C3 status ───────────────────────────────────
print("\n\nSATISFACTION DIAGNOSIS")
print("R_max covers demand 3468% - shortage is NOT a capacity problem")
print("Controller is CHOOSING to withhold. Check which tanks and why.\n")

# Run the headroom budget plan and check per-tank satisfaction
from module4.demand_generator import demand_window
from dataclasses import replace as dc_replace
import datetime

cfg3 = dc_replace(cfg2, PADDY_DURATION_DAYS=135)
import module4.simulation_data as sd2
area = sd2.recover_command_area_acres()
from module4.demand_generator import generate_demand
dem_full, _ = generate_demand(area.reindex(names).to_numpy(), cfg3)
dwin = demand_window(dem_full, 0, cfg3.T)   # day 0 window

R0 = np.zeros((N, T)) + state['_R_sink']
sim0 = simulate_cascade(state['S_current'], R0, state['Q_expected'],
                        state['alpha'], state['E'], state['S_min'],
                        state['S_max'], state['D'],
                        storage_limited_evap=True, upstream_iters=3)

print(f"{'tank':<30} {'S_cur/S_max':>12} {'C3 held':>8} {'demand':>12} {'R_max':>12} {'sat%':>8}")
print("-"*90)
S_cur = state['S_current']
S_max = state['S_max']
S_min = state['S_min']
R_max = state['R_max']
floor_abs = S_min + cfg3.C3_SAFETY_BUFFER_FRACTION * S_max

for i in opt:
    pct_cap = S_cur[i] / S_max[i] * 100 if S_max[i] > 0 else 0
    below_floor = S_cur[i] < floor_abs[i]
    dem_i = float(dwin[i, :].mean())
    rmax_i = float(R_max[i])
    # effective release at zero release = R_eff = 0
    # so satisfaction = 0 for all tanks at zero release
    # but what's the MAXIMUM achievable satisfaction?
    avail = max(0, S_cur[i] - S_min[i])
    max_sat = min(dem_i, rmax_i, avail) / max(dem_i, 1) * 100
    print(f"  {names[i]:<28} {pct_cap:>11.1f}% {str(below_floor):>8} "
          f"{dem_i:>12,.0f} {rmax_i:>12,.0f} {max_sat:>7.1f}%")