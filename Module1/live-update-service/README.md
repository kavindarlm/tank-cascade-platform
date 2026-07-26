# Tank Storage Service (Module 1 — live serving layer)

Keeps a date-wise **tank storage table** current from satellite data and
serves it to Module 2 / Module 3. It reuses Module 1's *frozen* detection
and volume logic — nothing is re-calibrated here.

## Relationship to `../development-history/`
This service is a **verified port**, not a rewrite. `config.py` documents,
line by line, which file in `../development-history/` each rule and
coefficient came from, and `scripts/verify_frozen_logic.py` proves the
port reproduces the official 2021–2025 dataset exactly. The historical
GEE `.txt` scripts and Colab `.py` scripts are **not executed by this
service** — the GEE scripts are JavaScript (Code Editor only) and the
Colab scripts are one-shot batch jobs over the whole CSV (global outlier
cleaning, `curve_fit` recalibration), neither of which fits an incremental
daily run. They stay in `development-history/` as the audit trail that
justifies why the numbers in `config.py` are what they are.

## The table
`data/tank_storage.csv` is the shared source of truth (committed to git).
It is the official 2021–2025 dataset plus two additive columns:
`record_status` (`validated_2021_2025` | `live`) and `last_updated`. The
original 13 columns are unchanged, so existing readers keep working.

New rows are only ever written for dates on which a real acquisition
existed. Missing dates stay missing — **interpolation is Module 3's job.**

## How the GEE work runs
`gee_detect.py` uses the `earthengine-api`: every pixel operation executes
on Google's GEE cloud, not locally. So teammates and the scheduler run it
without a heavy geoprocessing stack — they only need GEE access.

## Setup
```bash
pip install -r requirements.txt
earthengine authenticate        # one-time, interactive (local dev)
```

## Run order
1. **Verify the freeze** (do this first, and after any config edit):
   ```bash
   python scripts/verify_frozen_logic.py      # must print PASS
   ```
2. **Backfill the gap** 2025-12-21 → today (one-time):
   ```bash
   python backfill.py
   ```
3. **Daily update** (checks for a new acquisition; appends only if one exists):
   ```bash
   python run_daily.py
   ```
4. **Automate** — `.github/workflows/daily_update.yml` runs step 3 every day
   and commits the CSV if it changed. Needs repo secrets `GEE_SA_EMAIL` and
   `GEE_KEY_JSON` (see `service_account_setup.md`).

## Reading the table (Module 2 / 3)
```python
from storage_table import get_tank_volume
get_tank_volume("Nachchaduwa Wewa", "2026-02-14")
# -> {'status':'observed','row':{...}}  exact date exists
# -> {'status':'bracketed','before':{...},'after':{...}}  raw neighbours, no interpolation
```
Or just `pandas.read_csv("data/tank_storage.csv")`.
