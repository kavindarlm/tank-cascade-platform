"""
find_714.py - pinpoint the exact tank causing the residual C1 violation.
Run from project root: python find_714.py
"""
import pandas as pd, numpy as np
from module4.config import Config
from module4.demand_generator import generate_demand
import os

cfg = Config()
DD = cfg.DATA_DIR
p = pd.read_csv(DD + cfg.FILE_TANK_PARAMS)
s = pd.read_csv(DD + cfg.FILE_TANK_STORAGE)
cmd_path = DD + 'command_area.csv'

df = p.merge(s, on='tank_id')
tanks = list(df['tank_id'])
S0 = df['storage_m3'].to_numpy()
S_min = df['S_min'].to_numpy()
S_max = df['S_max'].to_numpy()
R_max = df['R_max'].to_numpy()
E = df['E_day0'].to_numpy()

# Load demand day0-6
dem = pd.read_csv(DD + cfg.FILE_DEMAND)
dem = dem.set_index('tank_id').loc[tanks]
D7 = dem[[f'day{i}' for i in range(7)]].to_numpy()  # (N,7)

# Reference plan: release min(demand, R_max) each day
print("=== Simulating reference plan (min(demand,Rmax), no inflow) 7 days ===\n")
S = S0.copy().astype(float)
worst = {}
for day in range(7):
    R = np.minimum(D7[:, day], R_max)
    S = S - R - E  # no inflow, no alpha (worst case)
    for i, t in enumerate(tanks):
        deficit = S_min[i] - S[i]
        if deficit > 0:
            if t not in worst or deficit > worst[t][1]:
                worst[t] = (day+1, deficit, S[i], S_min[i])

if worst:
    print("Tanks breaching S_min during the horizon (the C1 violation source):")
    print(f"{'tank':<26}{'day':>5}{'deficit':>10}{'storage':>11}{'S_min':>10}")
    print("-"*64)
    total = 0
    for t,(d,defc,sv,sm) in sorted(worst.items(), key=lambda x:-x[1][1]):
        print(f"{t:<26}{d:>5}{defc:>10,.0f}{sv:>11,.0f}{sm:>10,.0f}")
        total += defc
    print(f"\nTotal deficit (approx the C1 violation): {total:,.0f} m3")
    print("\nFIX: raise these tanks' day-0 storage slightly, OR they'll release less")
    print("(the optimizer CAN choose lower releases - it just needs enough generations).")
else:
    print("No S_min breach found in reference plan - C1 violation may be from overflow side.")
    # check overflow
    S = S0.copy().astype(float)
    for day in range(7):
        R = np.minimum(D7[:,day], R_max)
        S = S - R - E
    over = [(tanks[i], S[i]-S_max[i]) for i in range(len(tanks)) if S[i]>S_max[i]]
    if over:
        print("Tanks ABOVE S_max:", over)