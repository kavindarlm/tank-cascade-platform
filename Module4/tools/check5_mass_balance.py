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
        'date':         s['date'],
        'release':      sel.get('total_release', 0),
        'consumed':     sel.get('total_consumed', 0),
        'return_flow':  sel.get('total_return_flow', 0),
        'channel_loss': sel.get('total_channel_loss', 0),
    })

df = pd.DataFrame(rows)
# --- The correct identity -----------------------------------------------
# channel_loss is NOT a fourth flow component. It is defined in topsis.py as
#     channel_loss = return_flow - sum(alpha @ return_flow)
# i.e. the PORTION of return_flow that alpha fails to deliver anywhere.
# channel_loss is therefore a SUBSET of return_flow, not additional to it.
# The exact per-tank identity enforced by the simulator (state_transition.py)
# is simply:
#     release = consumed + return_flow
# Adding channel_loss on top double-counts the lost portion, which is why the
# original version of this check always showed a NEGATIVE error (reconstructed
# too large) that scaled with channel_loss rather than with release.
df['reconstructed'] = df['consumed'] + df['return_flow']
df['error'] = df['release'] - df['reconstructed']
df['error_pct'] = df['error'] / df['release'].replace(0, np.nan) * 100

# channel_loss is reported separately as a diagnostic: what fraction of the
# routed return flow the network fails to deliver anywhere (conveyance loss).
df['channel_loss_pct_of_return'] = np.where(
    df['return_flow'] > 0,
    df['channel_loss'] / df['return_flow'] * 100,
    0.0,
)

print('='*60)
print('MASS BALANCE CLOSURE CHECK')
print('  release == consumed + return_flow   (exact identity)')
print('  channel_loss reported separately as % of return_flow lost in transit')
print('='*60)
print(f'  Max absolute error  : {df.error.abs().max():,.4f} m3')
print(f'  Mean absolute error : {df.error.abs().mean():,.4f} m3')
print(f'  Max error pct       : {df.error_pct.abs().max():.6f}%')
print()
bad = df[df['error'].abs() > 1.0]
if len(bad) == 0:
    print('  PASS - mass balance (release = consumed + return_flow) closes on every day.')
else:
    print(f'  WARNING - {len(bad)} days do not close within 1 m3:')
    print(bad[['date','release','reconstructed','error']].to_string())

print()
print('CHANNEL LOSS DIAGNOSTIC (informational - not part of the closure check)')
print(f'  mean channel_loss as % of return_flow : {df["channel_loss_pct_of_return"].mean():.2f}%')
print(f'  max  channel_loss as % of return_flow : {df["channel_loss_pct_of_return"].max():.2f}%')
print('  This is the share of routed return flow the network (alpha) fails to')
print('  deliver anywhere - i.e. conveyance loss in transit between tanks.')