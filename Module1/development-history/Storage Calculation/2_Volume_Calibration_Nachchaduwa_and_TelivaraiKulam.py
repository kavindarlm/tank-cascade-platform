"""
STEP 2 — Apply the three-tier volume methodology to the cleaned, rebuilt dataset
"""
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

df = pd.read_csv('Module1_FULLYCLEANED_REBUILT_2021_2025.csv')
df['date'] = pd.to_datetime(df['date'])

# ── Tier 1: Nachchaduwa — recalibrate k against real gauge data ──
xl = pd.ExcelFile('water_level_manankattiya_2025.xlsx')
nach_raw = pd.read_excel(xl, sheet_name='Nachchaduwa', header=None)
gauge = nach_raw.iloc[3:, [0,1,2]].copy()
gauge.columns = ['date','water_head_ft','storage_acft']
gauge['date'] = pd.to_datetime(gauge['date'])
gauge['storage_acft'] = pd.to_numeric(gauge['storage_acft'], errors='coerce')
gauge = gauge.dropna(subset=['storage_acft'])
gauge['storage_mcm'] = gauge['storage_acft'] * 1233.48 / 1e6

nach_sat = df[df['pond_name']=='Nachchaduwa Wewa'][['date','water_ha']].copy()
merged = pd.merge_asof(nach_sat.sort_values('date'), gauge[['date','storage_mcm']].sort_values('date'),
                       on='date', direction='nearest', tolerance=pd.Timedelta('3 days')).dropna()
merged = merged[merged['water_ha']>50]

N_LIEBE = 1.43
def power_law_fixed_n(area, k): return k * np.power(area, N_LIEBE)
k_nach, _ = curve_fit(power_law_fixed_n, merged['water_ha'], merged['storage_mcm'])
k_nach = k_nach[0]
pred = power_law_fixed_n(merged['water_ha'], k_nach)
r2_nach = 1 - np.sum((merged['storage_mcm']-pred)**2)/np.sum((merged['storage_mcm']-merged['storage_mcm'].mean())**2)
print(f'Nachchaduwa recalibrated: k={k_nach:.6f}, R²={r2_nach:.3f}')

# ── Tier 2: Telivarai Kulam power law (unchanged — real SL bathymetry) ──
telivarai_raw = pd.DataFrame({
    'area_acres': [0.09,0.95,2.48,5.64,9.97,13.76,17.96,21.69,24.58,27.68,
                   30.85,35.02,43.70,52.29,65.03,86.46,103.55,106.94,107.83],
    'capacity_acft': [0.09,1.04,3.52,9.15,19.12,32.89,50.84,72.53,97.11,124.79,
                      155.64,190.66,234.36,286.66,351.68,438.14,541.69,648.63,756.45]
})
telivarai_raw['area_ha'] = telivarai_raw['area_acres'] * 0.404686
telivarai_raw['vol_mcm'] = telivarai_raw['capacity_acft'] * 1233.48 / 1e6
fit_data = telivarai_raw[telivarai_raw['area_ha']>0.3]

def power_law(area,k,n): return k*np.power(area,n)
params,_ = curve_fit(power_law, fit_data['area_ha'], fit_data['vol_mcm'], p0=[0.01,1.3], maxfev=10000)
K_TELIVARAI, N_TELIVARAI = params

def telivarai_volume(area_ha):
    return K_TELIVARAI * (max(area_ha,0) ** N_TELIVARAI)

# ── Tier 3: Imbulana ceiling check (unchanged) ────────────────────
imbulana_table1 = pd.DataFrame({
    'label':['Maximum','Minimum','Average'], 'depth_m':[3.54,0.76,1.93],
    'area_ha':[55.74,0.20,11.33], 'capacity_m3':[784493,1067,114215]
})
imbulana_table1['theta_linear'] = imbulana_table1['capacity_m3']/(imbulana_table1['area_ha']*imbulana_table1['depth_m'])
THETA = imbulana_table1['theta_linear'].mean()
D_AVG = 1.93

def imbulana_ceiling_mcm(area_ha):
    return (THETA * area_ha * D_AVG) / 1e6

# ── Apply final stratified volume ─────────────────────────────────
def final_volume(row):
    if row['pond_name'] == 'Nachchaduwa Wewa':
        return power_law_fixed_n(row['water_ha'], k_nach)
    tv = telivarai_volume(row['water_ha'])
    ceiling = imbulana_ceiling_mcm(row['total_ha'])
    fill_frac = row['water_ha']/row['total_ha'] if row['total_ha']>0 else 0
    return min(tv, ceiling * fill_frac * 1.5)

df['volume_mcm'] = df.apply(final_volume, axis=1)
df['volume_confidence'] = np.where(
    df['pond_name']=='Nachchaduwa Wewa',
    f'Validated (R²={r2_nach:.3f} vs gauge)',
    'Sri Lanka small-tank power law (Telivarai Kulam), Imbulana ceiling-capped')

print('\n=== FINAL SANITY CHECK ===')
print(df.groupby('pond_name')['volume_mcm'].max().sort_values(ascending=False).round(3).head(10).to_string())

df.to_csv('Module1_FINAL_CORRECTED_2021_2025.csv', index=False)
print('\nSaved: Module1_FINAL_CORRECTED_2021_2025.csv — this is your new official dataset')