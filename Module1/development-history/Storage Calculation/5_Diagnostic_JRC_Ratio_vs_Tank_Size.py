"""
Diagnostic: is the JRC ratio driven by tank SIZE (small tanks under-resolved
by 30m Landsat), confirming a resolution limitation rather than a bug?
"""
import pandas as pd
import numpy as np

df = pd.read_csv('/content/Module1_FINAL_CORRECTED_2021_2025.csv')
jrc = pd.read_csv('/content/Module1_JRC_Validation_PerTank.csv')

merged_jrc = pd.merge(
    df.groupby('pond_name')['water_ha'].max().reset_index(),
    jrc, on='pond_name'
)
merged_jrc['ratio'] = merged_jrc['water_ha'] / merged_jrc['jrc_max_ha'].replace(0, np.nan)
merged_jrc = merged_jrc.sort_values('water_ha', ascending=False)

print('=== RATIO BY TANK SIZE (largest to smallest) ===')
print(merged_jrc[['pond_name','water_ha','jrc_max_ha','ratio']].to_string(index=False))

# Correlation between tank size and ratio — if strongly negative,
# confirms small tanks drive the inflated ratio (resolution effect)
from scipy import stats
r, p = stats.pearsonr(np.log(merged_jrc['water_ha']), merged_jrc['ratio'].fillna(merged_jrc['ratio'].mean()))
print(f'\nCorrelation between log(tank size) and ratio: r={r:.3f}')
print('Strong negative r = small tanks drive the inflated ratio (30m JRC resolution limit)')
print('Weak/no correlation = something else is going on, worth a deeper look')

# Isolate large tanks only for a fairer comparison
large = merged_jrc[merged_jrc['water_ha'] > 50]
print(f'\n=== LARGE TANKS ONLY (>50 ha, well-resolved by both sensors) ===')
print(f'Mean ratio: {large["ratio"].mean():.2f} ± CV={large["ratio"].std()/large["ratio"].mean():.2f}')
print('(should be much closer to 1.0 if the resolution theory is correct)')