import pandas as pd, glob, os


import argparse
import os

_ap = argparse.ArgumentParser()
_ap.add_argument("output_dir", nargs="?", default="output",
                 help="output folder to check (default: output). "
                      "e.g. python %(prog)s output_25-03-08")
_args = _ap.parse_args()
OUTPUT_DIR = _args.output_dir

frames = []
for path in sorted(glob.glob(f'{OUTPUT_DIR}/*/mpc_decisions.csv')):
    date_str = os.path.basename(os.path.dirname(path))
    df = pd.read_csv(path)
    df['date'] = date_str
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)

summary = all_df.groupby('tank_id')['release_m3'].agg(
    total_release='sum',
    mean_daily='mean',
    zero_days=lambda x: (x == 0).sum(),
    active_days=lambda x: (x > 0).sum(),
).sort_values('total_release', ascending=False)

print('='*80)
print('PER-TANK SEASON RELEASE SUMMARY')
print(f'{"Tank":<35} {"Total":>12} {"Mean/day":>10} {"Active days":>12} {"Zero days":>10}')
print('-'*80)
for tid, row in summary.iterrows():
    print(f'{tid:<35} {row.total_release:>12,.0f} {row.mean_daily:>10,.1f} {row.active_days:>12} {row.zero_days:>10}')
print()
print(f'Tanks releasing ZERO all season : {(summary.active_days == 0).sum()}')
print(f'Tanks releasing every day       : {(summary.zero_days == 0).sum()}')