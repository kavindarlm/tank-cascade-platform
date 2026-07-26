import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

xl = pd.ExcelFile('/content/water_level_manankattiya_2025.xlsx')
nach_raw = pd.read_excel(xl, sheet_name='Nachchaduwa', header=None)
gauge = nach_raw.iloc[3:, [0,1,2]].copy()
gauge.columns = ['date','water_head_ft','storage_acft']
gauge['date'] = pd.to_datetime(gauge['date'])
gauge['storage_acft'] = pd.to_numeric(gauge['storage_acft'], errors='coerce')
gauge = gauge.dropna(subset=['storage_acft'])
gauge['storage_mcm'] = gauge['storage_acft'] * 1233.48 / 1e6

sat = pd.read_csv('/content/Module1_FULLYCLEANED_Final_2021_2025.csv')
sat['date'] = pd.to_datetime(sat['date'])
nach_sat = sat[sat['pond_name']=='Nachchaduwa Wewa'][['date','water_ha']].copy()
merged = pd.merge_asof(nach_sat.sort_values('date'), gauge[['date','storage_mcm']].sort_values('date'),
                        on='date', direction='nearest', tolerance=pd.Timedelta('3 days')).dropna()

# Published Liebe et al. (2005) exponent
N_LIEBE = 1.43
def power_law(area, k):
    return k * np.power(area, N_LIEBE)

valid = merged[merged['water_ha'] > 1]
k, _ = curve_fit(power_law, valid['water_ha'], valid['storage_mcm'])
k = k[0]
print(f'Volume (MCM) = {k:.6f} * Area(ha)^1.43   [Liebe et al. 2005 exponent, locally calibrated k]')

all_df = pd.read_csv('/content/Module1_FULLYCLEANED_Final_2021_2025.csv')
all_df['volume_mcm'] = power_law(all_df['water_ha'].clip(lower=0), k)
all_df.loc[all_df['water_ha'] <= 0, 'volume_mcm'] = 0

all_df['volume_confidence'] = np.where(
    all_df['pond_name']=='Nachchaduwa Wewa', 'Validated (R²=0.65 vs gauge)',
    'Indicative only — extrapolated, not field-validated')

# --- NEW: attach tank_id for downstream module calls ---
tanks = pd.read_csv('/content/tanks.csv')[['tank_id', 'tank_name']]
all_df = all_df.merge(tanks, left_on='pond_name', right_on='tank_name', how='left').drop(columns=['tank_name'])
cols = ['tank_id'] + [c for c in all_df.columns if c != 'tank_id']
all_df = all_df[cols]

assert all_df['tank_id'].isna().sum() == 0, "Some pond_name values didn't match tanks.csv — check spelling"

all_df.to_csv('Module1_FINAL_WITH_VOLUME_2021_2025.csv', index=False)
print('Saved: Module1_FINAL_WITH_VOLUME_2021_2025.csv')