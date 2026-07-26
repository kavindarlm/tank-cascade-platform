"""
Rebuild the final consolidated dataset using the FIXED S2 export,
to remove any contamination inherited from the buggy duplicate/zero rows
during the original sensor-priority consolidation.
"""
import pandas as pd
import numpy as np

# ── Load S1 (unchanged, already correct) ──────────────────────────
s1 = pd.read_csv('Module1_S1_AllTanks_6day_2021_2025.csv')
s1['date'] = pd.to_datetime(s1['date'])
if 'orbit' in s1.columns:
    s1 = s1[s1['orbit'] == 'DESCENDING'].copy()

# ── Load the FIXED S2 (replaces the buggy original) ────────────────
s2_fixed = pd.read_csv('Module1_S2_LargeTanks_FIXED.csv')
s2_fixed['date'] = pd.to_datetime(s2_fixed['date'])

large_tank_names = s2_fixed['pond_name'].unique().tolist()
small_tank_names = [t for t in s1['pond_name'].unique() if t not in large_tank_names]

print(f'Large tanks: {large_tank_names}')
print(f'Small tanks: {len(small_tank_names)}')

# ── Small tanks: S1 only, unchanged ────────────────────────────────
small_data = s1[s1['pond_name'].isin(small_tank_names)].copy()
small_data['sensor_used'] = 'S1_SAR'
small_data['s2_date'] = None
small_data['day_gap'] = 0

# ── Large tanks: S2-preferred (using FIXED file), S1 fallback ─────
large_records = []
for tank in large_tank_names:
    s1_tank = s1[s1['pond_name'] == tank].copy()
    s2_tank = s2_fixed[s2_fixed['pond_name'] == tank].copy()
    s1_dates = s1_tank['date'].sort_values().unique()

    for s1_date in s1_dates:
        date_diff = (s2_tank['date'] - s1_date).dt.days.abs()
        nearby_s2 = s2_tank[date_diff <= 3]

        if len(nearby_s2) > 0:
            best_s2 = nearby_s2.loc[date_diff[date_diff <= 3].idxmin()]
            large_records.append({
                'pond_name': tank,
                'date': s1_date,
                'year': pd.Timestamp(s1_date).year,
                'month': pd.Timestamp(s1_date).month,
                'water_ha': best_s2['water_ha'],
                'total_ha': best_s2['total_ha'],
                'fill_pct': best_s2['fill_pct'],
                'sensor_used': 'S2_MNDWI',
                's2_date': best_s2['date'],
                'day_gap': int(date_diff[date_diff <= 3].min())
            })
        else:
            s1_row = s1_tank[s1_tank['date'] == s1_date].iloc[0]
            large_records.append({
                'pond_name': tank,
                'date': s1_date,
                'year': pd.Timestamp(s1_date).year,
                'month': pd.Timestamp(s1_date).month,
                'water_ha': s1_row['water_ha'],
                'total_ha': s1_row['total_ha'],
                'fill_pct': s1_row['fill_pct'],
                'sensor_used': 'S1_SAR_fallback',
                's2_date': None,
                'day_gap': 0
            })

large_data = pd.DataFrame(large_records)

# ── Combine ─────────────────────────────────────────────────────────
keep_cols = ['pond_name', 'date', 'year', 'month', 'water_ha', 'total_ha',
             'fill_pct', 'sensor_used', 's2_date', 'day_gap']
consolidated = pd.concat([
    small_data[[c for c in keep_cols if c in small_data.columns]],
    large_data[[c for c in keep_cols if c in large_data.columns]]
], ignore_index=True)
consolidated = consolidated.sort_values(['date', 'pond_name']).reset_index(drop=True)

print(f'\nRebuilt consolidated records: {len(consolidated)}')
print(f'Unique dates: {consolidated["date"].nunique()}')

# ── Compare against the OLD (contaminated) final dataset ──────────
old_final = pd.read_csv('Module1_FINAL_WITH_VOLUME_STRATIFIED_2021_2025.csv')
old_final['date'] = pd.to_datetime(old_final['date'])

for tank in large_tank_names:
    old_t = old_final[old_final['pond_name'] == tank]
    new_t = consolidated[consolidated['pond_name'] == tank]
    merged_check = pd.merge(old_t[['date','water_ha']], new_t[['date','water_ha']],
                            on='date', suffixes=('_old','_new'))
    diff = (merged_check['water_ha_old'] - merged_check['water_ha_new']).abs()
    n_changed = (diff > 5).sum()
    if n_changed > 0:
        print(f'{tank}: {n_changed} dates changed by >5 ha after rebuild '
              f'(max change: {diff.max():.1f} ha)')

consolidated.to_csv('Module1_CONSOLIDATED_REBUILT_2021_2025.csv', index=False)
print('\nSaved: Module1_CONSOLIDATED_REBUILT_2021_2025.csv')
print('Next: re-run cleaning + volume calibration steps on this rebuilt file')