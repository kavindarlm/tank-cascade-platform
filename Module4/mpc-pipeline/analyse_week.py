"""
analyse_week.py
===============
Read every output/<date>/day_summary.json and report what a multi-day run
tells you that a single day cannot.

    python analyse_week.py
    python analyse_week.py --out output --start 2025-03-01 --end 2025-03-07

Six questions:

  1. FEASIBILITY   did every day solve under the hard C3 tier?
  2. TRAJECTORY    is village storage declining, and how steadily?
  3. DECOMPOSITION are C3-held tanks recovering while free tanks draw down?
  4. OBJECTIVES    are f1-f4 stable day to day, or is the controller thrashing?
  5. EQUITY        which tanks are persistently starved across the week?
  6. CROSS-CHECK   is agreement with Module 3 stable or drifting?
"""
import argparse
import glob
import json
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="output", help="output directory")
parser.add_argument("--start", default=None, help="first date YYYY-MM-DD")
parser.add_argument("--end", default=None, help="last date YYYY-MM-DD")
args = parser.parse_args()

from module4.config import DEFAULT_CONFIG as cfg
from module4.data_loader import load_static_inputs

paths = sorted(glob.glob(os.path.join(args.out, "*", "day_summary.json")))
days = []
for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        s = json.load(f)
    if s.get("status") != "complete":
        continue
    if args.start and s["date"] < args.start:
        continue
    if args.end and s["date"] > args.end:
        continue
    days.append(s)

if not days:
    raise SystemExit(f"No completed days found under {args.out}/")
days.sort(key=lambda s: s["date"])

static = load_static_inputs(cfg)
ids = list(static["tank_ids"])
S_min = np.asarray(static["S_min"], dtype=float)
S_max = np.asarray(static["S_max"], dtype=float)
floor = S_min + cfg.C3_SAFETY_BUFFER_FRACTION * S_max
sinks = set(getattr(cfg, "TERMINAL_SINK_TANKS", []) or [])
village = [i for i, t in enumerate(ids) if t not in sinks]

def stor(s):
    per = s["storage"]["per_tank_m3"]
    return np.array([float(per.get(t, np.nan)) for t in ids])

S = np.vstack([stor(s) for s in days])            # (D, N)

# Tanks the C3 floor holds back on day 1
held = [i for i in village if S[0, i] < floor[i]]
free = [i for i in village if S[0, i] >= floor[i]]

W = 78
print("\n" + "=" * W)
print(f"WEEK ANALYSIS   {days[0]['date']} .. {days[-1]['date']}   ({len(days)} days)")
print("=" * W)

# ---------------------------------------------------------------- 1
n_feas = sum(1 for s in days if s.get("feasible"))
rt = [s.get("runtime_seconds", 0.0) for s in days]
print(f"\n1. FEASIBILITY")
print(f"   feasible days   : {n_feas}/{len(days)}")
print(f"   runtime         : mean {np.mean(rt)/60:.1f} min/day, "
      f"total {np.sum(rt)/60:.0f} min")
print(f"   projected season: {np.mean(rt)*184/3600:.1f} hours for 184 days")

# ---------------------------------------------------------------- 2
print(f"\n2. STORAGE TRAJECTORY  (village tanks, excluding terminal sink)")
v0 = S[0, village].sum()
print(f"   {'date':<12}{'village m3':>15}{'daily change':>15}{'cum %':>10}")
print("   " + "-" * 52)
for k, s in enumerate(days):
    v = S[k, village].sum()
    d = v - S[k - 1, village].sum() if k else 0.0
    print(f"   {s['date']:<12}{v:>15,.0f}{d:>+15,.0f}{(v/v0-1)*100:>9.1f}%")
tot = S[-1, village].sum() - v0
per_day = tot / max(len(days) - 1, 1)
print(f"\n   net over {len(days)} days : {tot:+,.0f} m3  ({tot/v0*100:+.1f}%)")
print(f"   mean per day      : {per_day:+,.0f} m3/day")
if per_day < 0:
    weeks_left = S[-1, village].sum() / abs(per_day) / 7.0
    print(f"   -> DEPLETING. At this rate the village tanks hold water for "
          f"about {weeks_left:.0f} more weeks.")
else:
    print("   -> FILLING. In Yala this is the wrong direction; check inflow.")

# ---------------------------------------------------------------- 3
print(f"\n3. DECOMPOSITION  ({len(held)} tanks below the C3 floor on day 1)")
dh = S[-1, held].sum() - S[0, held].sum() if held else 0.0
df_ = S[-1, free].sum() - S[0, free].sum() if free else 0.0
print(f"   C3-held tanks ({len(held):>2}) : {dh:>+14,.0f} m3")
print(f"   free tanks    ({len(free):>2}) : {df_:>+14,.0f} m3")
print(f"   net              : {dh + df_:>+14,.0f} m3")
if held:
    print(f"\n   {'tank':<28}{'day 1':>13}{'last':>13}{'change':>13}{'% cap':>8}")
    print("   " + "-" * 75)
    for i in sorted(held, key=lambda j: -(S[-1, j] - S[0, j])):
        print(f"   {ids[i]:<28}{S[0,i]:>13,.0f}{S[-1,i]:>13,.0f}"
              f"{S[-1,i]-S[0,i]:>+13,.0f}{S[-1,i]/S_max[i]*100:>7.1f}%")
    print("\n   Reserve recovery in depleted tanks is C3 working, not a fault.")
    print("   Report this split rather than the aggregate - the aggregate hides it.")

# ---------------------------------------------------------------- 4
print(f"\n4. OBJECTIVES")
sel = [s.get("selected") for s in days]
if any(sel):
    keys = [("f1_shortage", "f1 shortage"), ("f2_overflow", "f2 overflow"),
            ("f3_equity", "f3 equity"), ("f4_water_loss", "f4 loss")]
    print(f"   {'date':<12}" + "".join(f"{lbl:>14}" for _, lbl in keys)
          + f"{'closeness':>11}")
    print("   " + "-" * 79)
    for s in days:
        d = s.get("selected") or {}
        print(f"   {s['date']:<12}"
              + "".join(f"{d.get(k, float('nan')):>14,.0f}" for k, _ in keys)
              + f"{d.get('closeness', float('nan')):>11.4f}")
    for k, lbl in keys:
        vals = np.array([(s.get("selected") or {}).get(k, np.nan) for s in days],
                        dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size > 1 and vals.mean() != 0:
            cv = vals.std() / abs(vals.mean()) * 100
            flag = "   <- unstable" if cv > 50 else ""
            print(f"   {lbl:<14} mean {vals.mean():>12,.0f}   "
                  f"variation {cv:>5.1f}%{flag}")
    print("\n   High day-to-day variation in an objective means the controller is")
    print("   thrashing - re-deciding rather than following a consistent policy.")

# ---------------------------------------------------------------- 5
print(f"\n5. EQUITY  (mean satisfaction across the week)")
sm = [(s.get("selected") or {}).get("satisfaction_pct_mean") for s in days]
sn = [(s.get("selected") or {}).get("satisfaction_pct_min") for s in days]
sm = [x for x in sm if x is not None]
sn = [x for x in sn if x is not None]
if sm:
    print(f"   mean satisfaction : {np.mean(sm):.1f}%")
    print(f"   worst tank, worst day : {np.min(sn):.1f}%")
    print("   (per-tank detail is in each day's mpc_decisions.csv)")

# ---------------------------------------------------------------- 6
print(f"\n6. CROSS-CHECK vs MODULE 3")
cc = [s.get("crosscheck") for s in days if s.get("crosscheck")]
if cc:
    print(f"   {'date':<12}{'direction':>12}{'mean |diff|':>14}{'flagged':>10}")
    print("   " + "-" * 48)
    for s in days:
        c = s.get("crosscheck")
        if not c:
            continue
        print(f"   {s['date']:<12}{c['direction_agreement_rate']*100:>11.1f}%"
              f"{c['mean_abs_ratio_diff']:>14.4f}{len(c['flagged_tanks']):>10}")
    da = [c["direction_agreement_rate"] for c in cc]
    print(f"\n   direction agreement : mean {np.mean(da)*100:.1f}%, "
          f"range {np.min(da)*100:.1f}-{np.max(da)*100:.1f}%")
    from collections import Counter
    cnt = Counter(t for c in cc for t in c["flagged_tanks"])
    if cnt:
        print(f"   persistently diverging tanks (of {len(cc)} days):")
        for t, n in cnt.most_common(8):
            note = "  <- C3-held, divergence expected" if t in [ids[i] for i in held] else ""
            print(f"       {t:<28}{n:>3} days{note}")
    print("\n   Divergence in C3-held tanks is the controller deliberately")
    print("   departing from business-as-usual. Divergence elsewhere is not.")
print()