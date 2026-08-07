import json, glob
import pandas as pd


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
        'date':             pd.to_datetime(s['date']),
        'total_release_m3': sel.get('total_release', 0),
        'total_consumed':   sel.get('total_consumed', 0),
        'satisfaction_pct': sel.get('satisfaction_pct_mean', 0),
        'f1_shortage':      sel.get('f1_shortage', 0),
    })

df = pd.DataFrame(rows).set_index('date')
monthly = df.resample('ME').agg({
    'total_release_m3': 'sum',
    'total_consumed':   'sum',
    'satisfaction_pct': 'mean',
    'f1_shortage':      'sum',
})

print('='*75)
print('MONTHLY BREAKDOWN')
print(f'{"Month":<12} {"Release":>14} {"Consumed":>14} {"Sat%":>7} {"Shortage":>14}')
print('-'*75)
for dt, row in monthly.iterrows():
    print(f'{dt.strftime("%Y-%m"):<12} {row.total_release_m3:>14,.0f} {row.total_consumed:>14,.0f} {row.satisfaction_pct:>7.1f}% {row.f1_shortage:>14,.0f}')