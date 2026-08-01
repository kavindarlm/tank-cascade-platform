"""
sensitivity_runoff.py
=====================
Find the runoff coefficient C that closes the seasonal water balance.

C is the fraction of rainfall that reaches the tank:

    inflow = C  x  catchment_area_km2  x  rainfall_mm  x  1000

It is a straight multiplier on every inflow in the model, and it is currently
the only parameter in the chain with no source behind it. Rainfall is now a
three-season Yala climatology; catchment areas are predefined incremental
delineations. C = 0.25 is a default.

This script sweeps C and reports what happens to village storage over the
7-day horizon. It does NOT run NSGA-II - it simulates two fixed plans, so it
finishes in seconds rather than six minutes:

    no-action     zero release on the optimised tanks (pure inflow vs loss)
    reference     min(demand, R_max), i.e. release exactly what is asked

The no-action column is the physics: if storage rises with zero release, inflow
exceeds evaporation and the forcing is too strong. The reference column is
closer to what the controller actually does.

It also splits the change between tanks the C3 floor holds back and tanks free
to release, because those two groups move in opposite directions and the
aggregate hides it.

Run:
    python sensitivity_runoff.py
    python sensitivity_runoff.py --data _sim_scratch
    python sensitivity_runoff.py --c 0.08 0.10 0.12 0.15 0.18 0.20 0.25
"""
import argparse
from dataclasses import replace

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=None, help="directory holding the day's CSVs")
parser.add_argument("--day", type=int, default=0, help="t_current offset")
parser.add_argument("--c", type=float, nargs="+", default=None,
                    help="runoff coefficients to test")
args = parser.parse_args()

from module4.config import DEFAULT_CONFIG
from module4.data_loader import (load_static_inputs, load_current_storage,
                                 load_module3_risk, build_state)
from module4.nsga2_optimizer import sink_release_rule
from module4.constraints import reference_plan
from module4.state_transition import simulate_cascade

C_VALUES = args.c if args.c else [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25]

base = DEFAULT_CONFIG
if args.data is not None:
    base = replace(base, DATA_DIR=args.data)
base = replace(base, VERBOSE_LOGGING=False)

print(f"\ndata directory : {base.DATA_DIR}")
print(f"current C      : {DEFAULT_CONFIG.RUNOFF_COEFFICIENT}")
print(f"\nVillage storage change over the {base.T}-day horizon.")
print("Looking for the C where the no-action column stops being positive.\n")

hdr = (f"{'C':>6}{'inflow/day':>13}{'no-action':>13}{'':>8}"
       f"{'reference':>13}{'':>8}{'C3-held':>12}{'free':>12}")
print(hdr)
print("-" * len(hdr))

results = []
for C in C_VALUES:
    cfg = replace(base, RUNOFF_COEFFICIENT=float(C))

    static = load_static_inputs(cfg)
    state = build_state(static, load_module3_risk(cfg), args.day, cfg,
                        S_current=load_current_storage(cfg))
    state["_R_sink"] = sink_release_rule(state)

    N, T = int(state["N"]), int(state["T"])
    S0 = state["S_current"]
    S_min, S_max = state["S_min"], state["S_max"]
    opt = np.array(state["opt_idx"])
    D_sim = state["D"] if cfg.ENABLE_CROP_CONSUMPTION else None

    # Tanks the C3 floor will hold back: those starting below it.
    floor = S_min + cfg.C3_SAFETY_BUFFER_FRACTION * S_max
    held = np.array([i for i in opt if S0[i] < floor[i]], dtype=int)
    free = np.array([i for i in opt if S0[i] >= floor[i]], dtype=int)

    def run(R_opt):
        sim = simulate_cascade(S0, R_opt + state["_R_sink"], state["Q_expected"],
                               state["alpha"], state["E"], S_min, S_max, D_sim,
                               storage_limited_evap=True, upstream_iters=3)
        return sim["S_unclipped"]

    S_none = run(np.zeros((N, T)))
    R_ref = reference_plan(state, cfg).copy()
    R_ref[state["sink_idx"], :] = 0.0
    S_ref = run(R_ref)

    start = float(S0[opt].sum())
    d_none = float(S_none[opt, -1].sum()) - start
    d_ref = float(S_ref[opt, -1].sum()) - start
    d_held = float(S_ref[held, -1].sum() - S0[held].sum()) if held.size else 0.0
    d_free = float(S_ref[free, -1].sum() - S0[free].sum()) if free.size else 0.0
    q = float(state["Q_expected"][opt, :].sum()) / T

    results.append((C, d_none, d_ref))
    print(f"{C:>6.2f}{q:>13,.0f}{d_none:>+13,.0f}{d_none/start*100:>7.1f}%"
          f"{d_ref:>+13,.0f}{d_ref/start*100:>7.1f}%"
          f"{d_held:>+12,.0f}{d_free:>+12,.0f}")

print()

# ---- Where does the no-action column cross zero? --------------------------
xs = [r[0] for r in results]
ys = [r[1] for r in results]
cross = None
for k in range(len(xs) - 1):
    if ys[k] < 0.0 <= ys[k + 1] or ys[k] > 0.0 >= ys[k + 1]:
        t = -ys[k] / (ys[k + 1] - ys[k])
        cross = xs[k] + t * (xs[k + 1] - xs[k])
        break

if cross is not None:
    print(f"Balance closes at C = {cross:.3f}  (no-action storage change = 0)")
    print("Published values for dry-zone cascade catchments are typically")
    print("0.10-0.20, so a calibrated value in that range is defensible.")
    if cross > 0.20:
        print(f"NOTE: {cross:.3f} is above the usual literature range. If the")
        print("      balance only closes above 0.20, something other than C is")
        print("      still adding water - check evaporation and the phantom")
        print("      water clamp before settling on a value.")
else:
    if all(y > 0 for y in ys):
        print("Storage rises at EVERY C tested. Extend the sweep downward:")
        print("    python sensitivity_runoff.py --c 0.02 0.04 0.06 0.08 0.10")
        print("If it still rises at C = 0.05, C is not the cause - inflow is")
        print("not what is filling these tanks.")
    else:
        print("Storage falls at every C tested; the current value is already")
        print("high enough to deplete the cascade.")

print("\nThe C3-held and free columns are the decomposition worth reporting:")
print("tanks held by the reserve floor accumulate while free tanks draw down.")
print("An aggregate rise can be reserve recovery rather than excess inflow.")