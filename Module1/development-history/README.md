# Module 1 — File Guide for Evaluators

This folder contains every script and data file used to produce Module 1
(satellite-based water surface area and storage volume monitoring for 32
Anuradhapura cascade tanks, 2021–2025). Files are organised into three
folders — **GEE**, **Exported CSVs**, **Storage Calculation** — each with
current, numbered files at the top level and an **Outdated/** subfolder
containing superseded versions kept for transparency and audit trail.

---

## How to read this guide

Files at the top level of each folder are numbered in the order they were
used. Follow the numbers in sequence to reproduce the pipeline exactly.
Everything in an `Outdated/` subfolder was a real, legitimate step taken
during the project but was later superseded — kept here so the full
development history is visible, not hidden.

---

## 1. GEE/ — Google Earth Engine scripts (run in code.earthengine.google.com)

| File | Purpose | Status |
|---|---|---|
| `1_SMART_SENSOR_6DAY_TIME_SERIES_S1+S2_raw_export.txt` | Master script: loads 32 tank boundaries, collects Sentinel-1 (descending+ascending) and Sentinel-2 imagery 2021–2025, computes MNDWI/SAR water detection, exports raw S1 and S2 CSVs | **Current** — the S1 portion of this script's output remains in use; its S2 portion was superseded by script 2 below |
| `2_FIX_S2_Tile_Mosaicking_produces_S2_LargeTanks_FIXED.txt` | Corrects a data bug: mosaics same-date Sentinel-2 tile granules before water detection, discards low-validity observations instead of recording false zeros | **Current — this is the fix that produced the final, trustworthy S2 data** |
| `3_JRC_Validation_Export.txt` | Exports JRC Global Surface Water occurrence/max-extent per tank for independent cross-validation | **Current** |
| `Outdated/Adaptive_Otsu_Threshold_...txt` | Explored an adaptive per-image Otsu SAR threshold as a possible fix for an S1-vs-S2 disagreement | **Outdated** — diagnostic investigation showed the real root cause was the S2 tile-mosaicking bug (fixed in script 2), not the SAR threshold. This script was a reasonable hypothesis that turned out not to be the actual cause; kept for transparency. |

---

## 2. Exported CSVs/ — Data files produced by the GEE scripts and Colab pipeline

| File | Produced by | Status |
|---|---|---|
| `0_water_level_manankattiya_2025_RAW_SOURCE_DATA.xlsx` | Irrigation Department of Sri Lanka | **Raw input** — real daily gauge storage records for Nachchaduwa, used as ground truth |
| `1_Module1_S1_AllTanks_6day_2021_2025.csv` | GEE script 1 | **Current** — raw Sentinel-1 export, still the input to the final pipeline |
| `2_Module1_S2_LargeTanks_FIXED.csv` | GEE script 2 | **Current** — corrected Sentinel-2 export for the 7 large tanks |
| `3_Module1_CONSOLIDATED_REBUILT_2021_2025.csv` | Storage Calc script 3 | **Current** — S1+S2(fixed) merged, sensor-priority rule applied |
| `4_Module1_FULLYCLEANED_REBUILT_2021_2025.csv` | Storage Calc script 1 | **Current** — outlier-cleaned version of file 3 |
| `5_Module1_FINAL_CORRECTED_2021_2025_OFFICIAL_DATASET.csv` | Storage Calc script 2 | **★ FINAL OFFICIAL DELIVERABLE ★** — 32 tanks, cleaned, stratified volume calibration |
| `6_Module1_JRC_Validation_PerTank.csv` | GEE script 3 | **Current** — supporting validation data |
| `7_Module1_Validation_Summary_FINAL.csv` | Storage Calc script 4 | **★ FINAL VALIDATION TABLE ★** |
| `Outdated/Module1_S2_LargeTanks_2021_2025_BUGGY...csv` | GEE script 1 (before fix) | **Outdated** — contained duplicate/zero rows from unmosaicked Sentinel-2 tiles; superseded by file 2 |
| `Outdated/Module1_FINAL_Descending_2021_2025.csv` | Early cleaning step | **Outdated** — superseded by file 3 |
| `Outdated/Module1_CONSOLIDATED_Final_2021_2025_built_on_buggy_S2.csv` | Old consolidation script | **Outdated** — built using the buggy S2 file above |
| `Outdated/Module1_FULLYCLEANED_Final_2021_2025_built_on_buggy_S2.csv` | Old cleaning script | **Outdated** — same buggy S2 lineage |
| `Outdated/Module1_FINAL_WITH_VOLUME_2021_2025_Liebe_only...csv` | Early volume script | **Outdated** — used only the Ghana-derived Liebe et al. (2005) coefficients, later replaced by the Sri Lanka-derived stratified approach, and built on buggy S2 data |
| `Outdated/Module1_FINAL_WITH_VOLUME_STRATIFIED_2021_2025_buggy_S2_base.csv` | Stratified volume script (pre-fix) | **Outdated** — correct volume *methodology* but applied on top of the still-buggy S2 base data; superseded once the S2 bug was found and fixed |

---

## 3. Storage Calculation/ — Python scripts run in Google Colab

| File | Input → Output | Status |
|---|---|---|
| `1_Outlier_Cleaning_on_Rebuilt_File.py` | `Module1_CONSOLIDATED_REBUILT...csv` → `Module1_FULLYCLEANED_REBUILT...csv` | **Current** |
| `2_Volume_Calibration_Nachchaduwa_and_TelivaraiKulam.py` | `Module1_FULLYCLEANED_REBUILT...csv` → `Module1_FINAL_CORRECTED...csv` | **Current — produces the official final dataset** |
| `3_Rebuild_Consolidated_Dataset_using_FIXED_S2.py` | S1 + `Module1_S2_LargeTanks_FIXED.csv` → `Module1_CONSOLIDATED_REBUILT...csv` | **Current** |
| `4_Validation_Summary_Table_Generator_FINAL.py` | `Module1_FINAL_CORRECTED...csv` (+ gauge Excel, S1/S2 files, JRC file) → `Module1_Validation_Summary_FINAL.csv` | **Current — final validation table generator** |
| `5_Diagnostic_JRC_Ratio_vs_Tank_Size.py` | `Module1_FINAL_CORRECTED...csv` + JRC file → console diagnostic | **Current** — tested whether JRC discrepancy was driven by tank size (30m resolution limit); result was inconclusive on its own, which motivated the manual satellite-imagery comparison confirming real tank rehabilitation under CRIWMP (see Validation Report Section 2.7) |
| `Outdated/Consolidation_Script_uses_buggy_S2-outdated.py` | Old S1+S2(buggy) → old consolidated file | **Outdated** — superseded by script 3 |
| `Outdated/Outlier_Cleaning_on_Buggy_Consolidated_File-outdated.py` | Old consolidated (buggy S2) → old cleaned file | **Outdated** — superseded by script 1 |
| `Outdated/Volume_LiebeOnly_ClearlyCaveated-outdated.py` | Old cleaned file → volume using Liebe et al. (2005) only | **Outdated** — superseded once the Sri Lanka-specific Telivarai Kulam / Imbulana stratified approach was adopted |
| `Outdated/Volume_FullyDerived_PreS2Fix-outdated.py` | Old cleaned file (buggy S2 base) → stratified volume, fully derived from source data | **Outdated** — correct methodology, but run on the pre-bugfix data; superseded by script 2 once S2 was corrected |
| `Outdated/Rebuild_Consolidated_DUPLICATE_of_current_file_3-outdated.py` | — | **Outdated — exact duplicate.** Byte-for-byte identical to current file `3_Rebuild_Consolidated_Dataset_using_FIXED_S2.py`. Kept only to preserve the original submission record; safe to delete. |

---

## Summary — the shortest path through the pipeline

```
GEE 1 (raw S1+S2) ──┬── GEE 2 (S2 tile-mosaic fix) ──► S2_LargeTanks_FIXED.csv
                     │
                     ▼
        Storage Calc 3 (consolidate S1 + fixed S2)
                     │
                     ▼
        Storage Calc 1 (outlier cleaning)
                     │
                     ▼
        Storage Calc 2 (volume calibration)
                     │
                     ▼
   ★ Module1_FINAL_CORRECTED_2021_2025.csv ★  ← OFFICIAL MODULE 1 OUTPUT
                     │
                     ▼
        Storage Calc 4 (validation summary)
                     │
                     ▼
   ★ Module1_Validation_Summary_FINAL.csv ★
```

GEE 3 (JRC export) and Storage Calc 5 (JRC diagnostic) run in parallel as
supporting validation evidence, feeding into the discussion of tank
rehabilitation under the CRIWMP programme documented in the Validation
and Methodology Report.
