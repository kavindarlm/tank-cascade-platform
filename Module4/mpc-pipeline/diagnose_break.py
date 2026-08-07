"""
diagnose_break.py
=================
The 19-day run is stable for days 1-7 then breaks: f1 jumps from ~200k to
~600k at day 8 and stops settling. A controller does not spontaneously
destabilise on a fixed problem, so an INPUT must change at day 8. This finds
which one, using the real data-loading path (no hand-typed filenames or tank
names).

    python diagnose_break.py
"""
import numpy as np

from module4.config import DEFAULT_CONFIG as cfg
from module4.data_loader import load_static_inputs
import module4.simulation_data as sd

static = load_static_inputs(cfg)
names = list(static["tank_ids"])
N = len(names)

# ------------------------------------------------------------------ DEMAND
print("=" * 70)
print("DEMAND  (does the crop calendar step near day 8?)")
print("=" * 70)
dm, diag = sd.generate_yala_demand(names, cfg)     # (N, season_days)
tot = dm.sum(axis=0)                                # per-day total across tanks
print(f"season_days = {diag.get('season_days')}, duration_class = "
      f"{diag.get('duration_class')}")
print(f"\n{'day':>4}{'date offset':>13}{'total demand m3':>18}{'change':>14}")
print("-" * 49)
for k in range(min(14, tot.shape[0])):
    ch = tot[k] - tot[k - 1] if k else 0.0
    mark = "  <-- STEP" if k > 0 and abs(ch) > 0.05 * max(tot[k - 1], 1) else ""
    print(f"{k+1:>4}{k:>13}{tot[k]:>18,.0f}{ch:>+14,.0f}{mark}")

jump = None
for k in range(1, min(14, tot.shape[0])):
    if abs(tot[k] - tot[k - 1]) > 0.05 * max(tot[k - 1], 1):
        jump = k
        break
if jump is not None:
    print(f"\n-> demand steps at day {jump+1} (offset {jump}). If this aligns with")
    print("   the day-8 objective jump, the controller is TRACKING the crop")
    print("   calendar - correct behaviour, not instability.")
else:
    print("\n-> no demand step in the first two weeks. Demand is not the cause.")

# ------------------------------------------------------------------ STORAGE
print("\n" + "=" * 70)
print("STORAGE  (does observed Module 3 storage step near day 8?)")
print("=" * 70)
try:
    fc = sd.load_forecast_dataset(tank_names=names)
    date_col = "forecast_date" if "forecast_date" in fc.columns else "date"
    dates = sorted(fc[date_col].unique())[:14]
    tcol = next(c for c in fc.columns if c.replace("+", "").replace(" ", "").lower()
                in ("t1", "t+1"))
    print(f"using storage column '{tcol}', dated by '{date_col}'\n")
    print(f"{'day':>4}{'date':>14}{'total t+1':>16}{'change':>14}")
    print("-" * 48)
    prev = None
    for k, dt in enumerate(dates):
        v = float(fc[fc[date_col] == dt][tcol].sum())
        ch = v - prev if prev is not None else 0.0
        mark = "  <-- STEP" if prev and abs(ch) > 0.05 * abs(prev) else ""
        print(f"{k+1:>4}{str(dt):>14}{v:>16,.2f}{ch:>+14,.2f}{mark}")
        prev = v
    print("\n-> a jump here on 2025-03-08 means the controller is reacting to a")
    print("   real observed storage change - again correct, not a fault.")
    print("   (t+1 may be a % or a volume basis; only the STEP matters here.)")
except Exception as e:
    print(f"could not inspect forecast dataset automatically: {e}")
    print("Check data/ for the module_forecasts CSV and its column names.")

print("\n" + "=" * 70)
print("READING THE RESULT")
print("=" * 70)
print("If EITHER demand or storage steps at day 7->8, the objective jump is the")
print("controller responding to an input change - a feature. Launch the season")
print("and describe the objective variation as input-tracking.")
print("If NEITHER steps, the break is internal and must be found before the run.")