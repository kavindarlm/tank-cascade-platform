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
    stor = s.get('storage', {})
    rows.append({
        'date':             s['date'],
        'total_storage_m3': stor.get('total_today_m3', 0),
        'total_release_m3': sel.get('total_release', 0),
        'satisfaction_pct': sel.get('satisfaction_pct_mean', 0),
    })

df = pd.DataFrame(rows)
df['storage_change'] = df['total_storage_m3'].diff()

rain_events = df[df['storage_change'] > 50000]
print('='*70)
print('RAINFALL EVENT DAYS  (storage gain > 50,000 m3)')
print(f'{"Date":<14} {"Storage gain":>18} {"Release":>18} {"Sat%":>8}')
print('-'*70)
for _, row in rain_events.iterrows():
    print(f'{row.date:<14} {row.storage_change:>18,.0f} {row.total_release_m3:>18,.0f} {row.satisfaction_pct:>8.1f}%')

print()
print('TOP 10 HIGHEST RELEASE DAYS')
print(f'{"Date":<14} {"Release":>14} {"Storage":>14} {"Sat%":>8}')
print('-'*60)
for _, row in df.nlargest(10, 'total_release_m3').iterrows():
    print(f'{row.date:<14} {row.total_release_m3:>14,.0f} {row.total_storage_m3:>14,.0f} {row.satisfaction_pct:>8.1f}%')