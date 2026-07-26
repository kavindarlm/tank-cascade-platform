# Module 1 — Satellite-Based Water Resource Monitoring

Water surface area detection and storage volume estimation for 32 village
irrigation tanks (Anuradhapura district, Sri Lanka, 2021–present). This
repo has two parts, kept together so the full trail — from raw satellite
processing to the daily-updating table other modules read — lives in one
place.

```
Module1/
├── development-history/     ← HOW the 2021–2025 baseline dataset was built
│   ├── README.md               (file-by-file guide, current vs outdated)
│   ├── GEE/                    Google Earth Engine scripts (JavaScript)
│   ├── Exported CSVs/          raw + intermediate + official outputs
│   └── Storage Calculation/    Colab Python: cleaning, volume calibration
│
└── live-update-service/      ← the table that stays current, day to day
    ├── README.md                setup, run order, serving contract
    ├── config.py                frozen rules/coefficients (traces back
    │                            to specific files in development-history/)
    ├── data/tank_storage.csv    the shared table — commit target for CI
    └── .github/workflows/       daily GitHub Action
```

## Which folder do I want?

- **Verifying methodology, checking what was tried and superseded, writing
  up the thesis/validation report** → `development-history/`. Nothing in
  here is touched by automation; it's the permanent record.
- **Getting today's tank volume, building Module 2/3 against live data,
  running or debugging the daily update** → `live-update-service/`.

## How they connect

`live-update-service/config.py` is a verified port of the detection rules
and volume coefficients from `development-history/GEE/...` and
`development-history/Storage Calculation/...` — not a re-derivation. Every
constant is traced to its source file in `config.py`'s docstring, and
`live-update-service/scripts/verify_frozen_logic.py` checks the port still
reproduces `development-history/Exported CSVs/5_..._OFFICIAL_DATASET.csv`
exactly. If you ever change a detection rule or volume coefficient, change
it in the history first (documenting why, per that folder's own
conventions), then port it into `config.py` and re-run the verification
script before trusting the daily table again.
