import pandas as pd
import numpy as np

# ── Load both datasets ────────────────────────────────────────────
s1 = pd.read_csv('/content/Module1_FINAL_Descending_2021_2025.csv')
s2 = pd.read_csv('/content/Module1_S2_LargeTanks_2021_2025.csv')

s1['date'] = pd.to_datetime(s1['date'])
s2['date'] = pd.to_datetime(s2['date'])

print('S1 records:', len(s1))
print('S2 records:', len(s2))
print('S1 unique tanks:', s1['pond_name'].nunique())
print('S2 unique tanks:', s2['pond_name'].nunique())

# ── Define large tanks based on area ─────────────────────────────
# These 7 tanks are above 50 ha in your dataset
large_tank_names = [
    'Nachchaduwa Wewa',
    'Periyakulama Wewa',
    'Wannanmaduwa Wewa',
    'Todamaduwa Wewa',
    'Tirappane Wewa',
    'Mahakanumulla Wewa',
    'Alittana Wewa'
]

small_tank_names = [t for t in s1['pond_name'].unique()
                    if t not in large_tank_names]

print(f'\nLarge tanks ({len(large_tank_names)}): {large_tank_names}')
print(f'Small tanks ({len(small_tank_names)}): {small_tank_names}')

# ── PART 1: Small tanks — S1 only ────────────────────────────────
small_data = s1[s1['pond_name'].isin(small_tank_names)].copy()
small_data['sensor_used'] = 'S1_SAR'
print(f'\nSmall tank records: {len(small_data)}')

# ── PART 2: Large tanks — S2 preferred, S1 fallback ──────────────
large_records = []

for tank in large_tank_names:
    s1_tank = s1[s1['pond_name'] == tank].copy()
    s2_tank = s2[s2['pond_name'] == tank].copy()

    # Get all unique dates from S1 for this tank
    s1_dates = s1_tank['date'].sort_values().unique()

    for s1_date in s1_dates:
        # Check if S2 image exists within 3 days of this S1 date
        date_diff = (s2_tank['date'] - s1_date).dt.days.abs()
        nearby_s2 = s2_tank[date_diff <= 3]

        if len(nearby_s2) > 0:
            # S2 available — use it (more accurate for large tanks)
            best_s2 = nearby_s2.loc[date_diff[date_diff <= 3].idxmin()]
            large_records.append({
                'pond_name':   tank,
                'pond_type':   best_s2.get('pond_type', 'Tank'),
                'date':        s1_date,  # keep S1 date as reference
                'year':        pd.Timestamp(s1_date).year,
                'month':       pd.Timestamp(s1_date).month,
                'water_ha':    best_s2['water_ha'],
                'total_ha':    best_s2['total_ha'],
                'fill_pct':    best_s2['fill_pct'],
                'sensor_used': 'S2_MNDWI',
                's2_date':     best_s2['date'],
                'day_gap':     int(date_diff[date_diff <= 3].min())
            })
        else:
            # No S2 available — use S1 (fallback)
            s1_row = s1_tank[s1_tank['date'] == s1_date].iloc[0]
            large_records.append({
                'pond_name':   tank,
                'pond_type':   s1_row.get('pond_type', 'Tank'),
                'date':        s1_date,
                'year':        pd.Timestamp(s1_date).year,
                'month':       pd.Timestamp(s1_date).month,
                'water_ha':    s1_row['water_ha'],
                'total_ha':    s1_row['total_ha'],
                'fill_pct':    s1_row['fill_pct'],
                'sensor_used': 'S1_SAR_fallback',
                's2_date':     None,
                'day_gap':     0
            })

large_data = pd.DataFrame(large_records)
print(f'Large tank records: {len(large_data)}')

# How many S2 vs S1 used for large tanks
sensor_counts = large_data['sensor_used'].value_counts()
print(f'\nSensor used for large tanks:')
print(sensor_counts.to_string())

# ── PART 3: Combine small + large ────────────────────────────────
# Align columns
small_data['sensor_used'] = 'S1_SAR'
small_data['s2_date']     = None
small_data['day_gap']     = 0

# Keep common columns
keep_cols = ['pond_name', 'pond_type', 'date', 'year',
             'month', 'water_ha', 'total_ha', 'fill_pct',
             'sensor_used', 's2_date', 'day_gap']

small_final = small_data[[c for c in keep_cols
                          if c in small_data.columns]]
large_final = large_data[[c for c in keep_cols
                          if c in large_data.columns]]

consolidated = pd.concat([small_final, large_final],
                         ignore_index=True)
consolidated = consolidated.sort_values(
    ['date', 'pond_name']).reset_index(drop=True)

# ── PART 4: Verify results ────────────────────────────────────────
print('\n=== CONSOLIDATED DATASET ===')
print(f'Total records    : {len(consolidated)}')
print(f'Unique dates     : {consolidated["date"].nunique()}')
print(f'Unique tanks     : {consolidated["pond_name"].nunique()}')
print(f'Records per date : {len(consolidated)/consolidated["date"].nunique():.0f}')
print()

print('Sensor distribution:')
print(consolidated['sensor_used'].value_counts().to_string())
print()

# Check one date has exactly 32 tanks
sample_date = consolidated['date'].iloc[0]
count_check  = len(consolidated[consolidated['date'] == sample_date])
print(f'Records on {sample_date.date()}: {count_check} (expected 32)')
print()

# Summary per large tank — how often S2 was used
print('=== S2 USAGE FOR LARGE TANKS ===')
for tank in large_tank_names:
    tank_data  = large_data[large_data['pond_name'] == tank]
    s2_count   = (tank_data['sensor_used'] == 'S2_MNDWI').sum()
    s1_count   = (tank_data['sensor_used'] != 'S2_MNDWI').sum()
    total      = len(tank_data)
    pct_s2     = s2_count / total * 100
    print(f'  {tank:<35s} S2:{s2_count:3d} ({pct_s2:4.1f}%)  '
          f'S1:{s1_count:3d} ({100-pct_s2:4.1f}%)')

# ── PART 5: Save final file ───────────────────────────────────────
consolidated.to_csv('Module1_CONSOLIDATED_Final_2021_2025.csv',
                    index=False)

print()
print('=== SAVED ===')
print('Module1_CONSOLIDATED_Final_2021_2025.csv')
print('This is your OFFICIAL final Module 1 dataset')
print()
print('Column guide:')
print('  sensor_used = S1_SAR         → Sentinel-1 used')
print('  sensor_used = S2_MNDWI       → Sentinel-2 used (large tank, clear sky)')
print('  sensor_used = S1_SAR_fallback → S1 used because S2 unavailable')
print('  day_gap     → days between S1 and S2 date when S2 was used')