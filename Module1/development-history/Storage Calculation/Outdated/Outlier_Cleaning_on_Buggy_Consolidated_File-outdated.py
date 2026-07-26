import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Load Dataset
# ============================================================
df = pd.read_csv('/content/Module1_CONSOLIDATED_Final_2021_2025.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['pond_name', 'date']).reset_index(drop=True)

# ============================================================
# STEP 1 - Detect Single-Date Drop Outliers
# ============================================================
def flag_single_outliers(group):
    group = group.sort_values('date').reset_index(drop=True)
    fill = group['fill_pct'].values
    is_outlier = np.zeros(len(fill), dtype=bool)

    for i in range(1, len(fill)-1):
        prev_val = fill[i-1]
        curr_val = fill[i]
        next_val = fill[i+1]

        if (prev_val - curr_val > 30) and (next_val - curr_val > 30):
            is_outlier[i] = True

    group['is_outlier'] = is_outlier
    return group

df = df.groupby('pond_name', group_keys=False).apply(flag_single_outliers)

n1 = df['is_outlier'].sum()

print("="*60)
print("STEP 1 - SINGLE DROP OUTLIERS")
print("="*60)
print(f"Total Outliers : {n1}")

# ============================================================
# Interpolate Step 1 Outliers
# ============================================================
def interpolate_outliers(group):
    group = group.sort_values('date').reset_index(drop=True)

    group.loc[group['is_outlier'], 'water_ha'] = np.nan
    group.loc[group['is_outlier'], 'fill_pct'] = np.nan

    group['water_ha'] = group['water_ha'].interpolate(method='linear')
    group['fill_pct'] = group['fill_pct'].interpolate(method='linear')

    return group

df = df.groupby('pond_name', group_keys=False).apply(interpolate_outliers)
df = df.drop(columns=['is_outlier'])

# ============================================================
# STEP 2 - Detect Short Runs of Low Values
# ============================================================
def flag_run_outliers(group, low_thresh=10, high_thresh=40, max_run=4):
    group = group.sort_values('date').reset_index(drop=True)

    fill = group['fill_pct'].values
    n = len(fill)

    is_bad = np.zeros(n, dtype=bool)

    i = 0
    while i < n:

        if fill[i] < low_thresh:

            j = i

            while j < n and fill[j] < low_thresh:
                j += 1

            run_len = j - i

            before = fill[i-1] if i > 0 else None
            after = fill[j] if j < n else None

            if (
                run_len <= max_run
                and before is not None
                and after is not None
                and before > high_thresh
                and after > high_thresh
            ):
                is_bad[i:j] = True

            i = j

        else:
            i += 1

    group['is_outlier'] = is_bad
    return group

df = df.groupby('pond_name', group_keys=False).apply(flag_run_outliers)

n2 = df['is_outlier'].sum()

print()
print("="*60)
print("STEP 2 - LOW RUN OUTLIERS")
print("="*60)
print(f"Additional Outliers : {n2}")

# ============================================================
# Interpolate Step 2 Outliers
# ============================================================
df = df.groupby('pond_name', group_keys=False).apply(interpolate_outliers)
df = df.drop(columns=['is_outlier'])

# ============================================================
# Final Verification (Nachchaduwa)
# ============================================================
nach = df[df['pond_name'] == 'Nachchaduwa Wewa'].sort_values('date')

print()
print("="*60)
print("FINAL CHECK")
print("="*60)
print(f"Remaining values <5% : {(nach['fill_pct'] < 5).sum()}")
print(f"Minimum fill %       : {nach['fill_pct'].min():.2f}%")

# ============================================================
# Plot Final Result
# ============================================================
fig, ax = plt.subplots(figsize=(16,5))

ax.plot(
    nach['date'],
    nach['fill_pct'],
    color='darkgreen',
    linewidth=1.5
)

ax.set_title("Nachchaduwa Wewa - Fully Cleaned")
ax.set_ylabel("Fill Percentage (%)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Fully_Cleaned_Nachchaduwa.png", dpi=150)
plt.show()

# ============================================================
# Save Final Dataset
# ============================================================
df.to_csv("Module1_FULLYCLEANED_Final_2021_2025.csv", index=False)

print()
print("="*60)
print("DONE")
print("="*60)
print(f"Step 1 Outliers Fixed : {n1}")
print(f"Step 2 Outliers Fixed : {n2}")
print(f"Total Fixed           : {n1 + n2}")
print("Saved: Module1_FULLYCLEANED_Final_2021_2025.csv")