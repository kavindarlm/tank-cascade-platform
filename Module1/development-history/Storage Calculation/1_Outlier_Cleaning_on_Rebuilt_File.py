"""
STEP 1 — Re-run outlier cleaning on the REBUILT consolidated dataset
"""
import pandas as pd
import numpy as np

df = pd.read_csv('Module1_CONSOLIDATED_REBUILT_2021_2025.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['pond_name', 'date']).reset_index(drop=True)

# ── Detect single-date drops (sandwiched between two high values) ─
def flag_outliers(group):
    group = group.sort_values('date').reset_index(drop=True)
    fill = group['fill_pct'].values
    is_outlier = np.zeros(len(fill), dtype=bool)
    for i in range(1, len(fill) - 1):
        if (fill[i-1] - fill[i]) > 30 and (fill[i+1] - fill[i]) > 30:
            is_outlier[i] = True
    group['is_outlier'] = is_outlier
    return group

df = df.groupby('pond_name', group_keys=False).apply(flag_outliers)
print(f'Single-date outliers flagged: {df["is_outlier"].sum()}')

# ── Detect multi-date runs (2-4 consecutive low values) ────────────
def flag_runs(group, low_thresh=10, high_thresh=40, max_run=4):
    group = group.sort_values('date').reset_index(drop=True)
    fill = group['fill_pct'].values
    n = len(fill)
    is_bad = group['is_outlier'].values.copy()
    i = 0
    while i < n:
        if fill[i] < low_thresh and not is_bad[i]:
            j = i
            while j < n and fill[j] < low_thresh:
                j += 1
            run_len = j - i
            before_val = fill[i-1] if i > 0 else None
            after_val = fill[j] if j < n else None
            if run_len <= max_run and before_val is not None and after_val is not None:
                if before_val > high_thresh and after_val > high_thresh:
                    is_bad[i:j] = True
            i = j
        else:
            i += 1
    group['is_outlier'] = is_bad
    return group

df = df.groupby('pond_name', group_keys=False).apply(flag_runs)
print(f'Total outliers flagged (single + multi-date runs): {df["is_outlier"].sum()}')

# ── Interpolate flagged values ──────────────────────────────────────
def interpolate_outliers(group):
    group = group.sort_values('date').reset_index(drop=True)
    group.loc[group['is_outlier'], 'water_ha'] = np.nan
    group.loc[group['is_outlier'], 'fill_pct'] = np.nan
    group['water_ha'] = group['water_ha'].interpolate(method='linear')
    group['fill_pct'] = group['fill_pct'].interpolate(method='linear')
    return group

df_clean = df.groupby('pond_name', group_keys=False).apply(interpolate_outliers)
df_clean = df_clean.drop(columns=['is_outlier'])

# ── Verify Nachchaduwa is clean ────────────────────────────────────
nach = df_clean[df_clean['pond_name']=='Nachchaduwa Wewa']
print(f'\nNachchaduwa fill % range after cleaning: {nach["fill_pct"].min():.1f}% - {nach["fill_pct"].max():.1f}%')

df_clean.to_csv('Module1_FULLYCLEANED_REBUILT_2021_2025.csv', index=False)
print('Saved: Module1_FULLYCLEANED_REBUILT_2021_2025.csv')