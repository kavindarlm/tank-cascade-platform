import json, glob
import pandas as pd
import numpy as np


import argparse
import os

_ap = argparse.ArgumentParser()
_ap.add_argument("output_dir", nargs="?", default="output",
                 help="output folder to check (default: output). "
                      "e.g. python %(prog)s output_25-03-08")
_args = _ap.parse_args()
OUTPUT_DIR = _args.output_dir

rows = []
for path in sorted(glob.glob(f'{OUTPUT_DIR}/*/day_summary.json')):
    with open(path) as f:
        s = json.load(f)
    sel = s.get('selected') or {}
    rows.append({
        'date':                  s['date'],
        'total_release_m3':      sel.get('total_release', 0),
        'total_consumed_m3':     sel.get('total_consumed', 0),
        'total_return_flow_m3':  sel.get('total_return_flow', 0),
        'total_channel_loss_m3': sel.get('total_channel_loss', 0),
        'f1_shortage':           sel.get('f1_shortage', 0),
        'f2_overflow':           sel.get('f2_overflow', 0),
        'satisfaction_pct_mean': sel.get('satisfaction_pct_mean', 0),
    })

df = pd.DataFrame(rows)

print('='*60)
print('SEASON WATER BALANCE  (2025-03-01 to 2025-08-31)')
print('='*60)
print(f'  Total released         : {df.total_release_m3.sum():>15,.0f} m3')
print(f'  Total consumed         : {df.total_consumed_m3.sum():>15,.0f} m3')
print(f'  Total return flow      : {df.total_return_flow_m3.sum():>15,.0f} m3')
print(f'  Total channel loss     : {df.total_channel_loss_m3.sum():>15,.0f} m3')
print(f'  Total f1 shortage      : {df.f1_shortage.sum():>15,.0f} m3')
print(f'  Total f2 overflow      : {df.f2_overflow.sum():>15,.0f} m3')
print()
total_rel = df.total_release_m3.sum()
print(f'  Consumed  / Released   : {df.total_consumed_m3.sum()/total_rel*100:.1f}%')
print(f'  Return    / Released   : {df.total_return_flow_m3.sum()/total_rel*100:.1f}%')
print(f'  Chan.loss / Released   : {df.total_channel_loss_m3.sum()/total_rel*100:.1f}%')
print()
print(f'  Mean daily release     : {df.total_release_m3.mean():>15,.0f} m3/day')
print(f'  Mean satisfaction      : {df.satisfaction_pct_mean.mean():.1f}%')
idx_min = df.satisfaction_pct_mean.idxmin()
idx_max = df.satisfaction_pct_mean.idxmax()
print(f'  Min satisfaction day   : {df.loc[idx_min,"date"]}  ({df.loc[idx_min,"satisfaction_pct_mean"]:.1f}%)')
print(f'  Max satisfaction day   : {df.loc[idx_max,"date"]}  ({df.loc[idx_max,"satisfaction_pct_mean"]:.1f}%)')