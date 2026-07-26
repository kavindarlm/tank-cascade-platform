"""
Sanity gate. Recomputes volume_mcm for every row currently marked
'validated_2021_2025' using config.final_volume() and asserts it still
matches. Run this after ANY edit to config.py. If it fails, the frozen
logic was changed and no longer reproduces the official dataset.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, config as C
df = pd.read_csv(C.TABLE_CSV)
v = df[df["record_status"] == "validated_2021_2025"].copy()
v["chk"] = v.apply(lambda r: C.final_volume(r["pond_name"], r["water_ha"], r["total_ha"]), axis=1)
err = (v["chk"] - v["volume_mcm"]).abs()
print(f"rows checked      : {len(v)}")
print(f"max abs error MCM : {err.max():.3e}")
assert err.max() < 1e-6, "FROZEN LOGIC CHANGED — volume no longer matches official CSV"
print("PASS — frozen volume logic reproduces the official dataset.")
