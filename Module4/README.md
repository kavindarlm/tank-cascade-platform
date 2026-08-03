# Module 4 — Multi-Objective Model Predictive Control (MO-MPC)

Daily sluice-release decision support for the tank cascade. Consumes Module 1
(today's storage), Module 2 (cascade alpha matrix), and Module 3 (risk/
forecast), runs NSGA-II across sampled rainfall scenarios, and TOPSIS-ranks
the result into a release decision.

For the full methodology (objectives, constraints, NSGA-II/TOPSIS design) see
[`EXPLANATION.md`](EXPLANATION.md). This file is only about **how to run it**.

## Directory layout

```
Module4/
├── EXPLANATION.md
├── .venv/                  the one Python environment for everything below
├── mpc-pipeline/           the actual pipeline - cd here before running anything
│   ├── main.py                 single-day entry point (see Setting B below)
│   ├── run_simulation.py       full-season entry point (see Setting A below)
│   ├── evaluate_season.py, baseline_comparator.py, audit_day.py, ...
│   ├── module4/                 the package (NSGA-II, TOPSIS, objectives, ...)
│   ├── data/, data_/            static inputs (tank_params, demand, rainfall, ...)
│   ├── _sim_scratch/, _baseline_scratch/   scratch dirs auto-populated per run
│   ├── output/                  live output of the season run you're actively on
│   └── tests/
├── tools/                   standalone diagnostics (check1-5, generate_plots.py)
└── runs-archive/            past/completed season runs, kept for comparison
```

## Setup (once per shell)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& G:\Reserch_Project_Final_Year_Repo\tank-cascade-platform\Module4\.venv\Scripts\Activate.ps1
cd G:\Reserch_Project_Final_Year_Repo\tank-cascade-platform\Module4\mpc-pipeline
```

Every command below assumes you're sitting in `mpc-pipeline/` with the venv
active. All paths in this file are relative to that directory unless noted.

---

## Setting A — Full-season simulation (`run_simulation.py`)

Runs day-by-day over a date range, one full NSGA-II/TOPSIS cycle per day,
sourcing Module 3's risk/forecast from a single **bulk CSV** covering the
whole season (`--forecast`). This is the research/back-testing mode. **This
mechanism is untouched by the Module 3 live-mode work below** — it always
reads the bulk CSV, never Module 3's live daily output.

### Command

```powershell
python run_simulation.py --start <YYYY-MM-DD> --end <YYYY-MM-DD> `
    --season yala --duration 135 `
    --forecast data\module4_forecasts_mar_aug_2025.csv `
    --output <output-folder-name> `
    --consolidated-output
```

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--start` / `--end` | `2025-03-01` / `2025-08-31` | Inclusive date range to simulate |
| `--season` | `yala` | `yala` (Mar-Aug) or `maha` (Oct-Feb) — sets the demand curve |
| `--duration` | 135 (season default) | Paddy duration class: 105, 135, or 90 |
| `--forecast` | **broken default — always pass this explicitly, see Known Issues** | Path to the bulk Module 1+3 forecast CSV |
| `--output` | `output` | Root folder for day folders + summaries (relative to `mpc-pipeline/`) |
| `--static-dir` | `_sim_scratch` | Scratch dir for staged per-day inputs — leave as default |
| `--consolidated-output` | **no-op — see Known Issues** | Intended to toggle appending every day to run-wide `consolidated_*.csv` files at the output root, but `Config.CONSOLIDATED_OUTPUT` already defaults `True` and nothing resets it, so this happens regardless of the flag |
| `--fast-test` | off | Shrinks NSGA-II (pop=20, gen=10, K=5) to smoke-test wiring in seconds — **never use for real results** |

### Examples

```powershell
# Resume an interrupted run (completed days are skipped automatically)
python run_simulation.py --start 2026-03-01 --end 2026-07-26 --duration 135 `
    --forecast data\module4_forecasts_mar_aug_2025.csv --consolidated-output

# Quick smoke test (seconds, not real results)
python run_simulation.py --fast-test --start 2025-03-01 --end 2025-03-01 `
    --forecast data\module4_forecasts_mar_aug_2025.csv

# A fresh run into its own output folder
python run_simulation.py --start 2025-03-01 --end 2025-08-31 `
    --forecast data\module4_forecasts_mar_aug_2025.csv --output my_run
```

### Evaluating a completed run

```powershell
python evaluate_season.py output --season yala --duration 135 `
    --demand-csv data\demand.csv --tank-params-csv data\tank_params.csv
python ..\tools\generate_plots.py output\_evaluation
```
(`--alpha-csv` no longer needs to be passed — it defaults to Module 2's live
`alpha_matrix.csv` output. Pass it explicitly only to pin an older matrix.)

To inspect an archived run instead of the live `output/`, point at
`..\runs-archive\<folder-name>` (e.g. `output_25-03-08`, `output_26`, ...).

---

## Setting B — Single day, live Module 3 forecast (`main.py`)

Runs **exactly one day**, reading Module 3's live daily output directly
(`Module3/forecast-service/outputs/forecast_<date>.csv`) instead of a bulk
CSV or a static placeholder — the deployment/production mode. Uses the exact
same optimisation engine as Setting A (`mpc_loop.run_mpc`) — only the input
data source differs.

### Step 1 — make sure Module 3 has today's forecast

Module 4 never generates this itself. From `Module3/forecast-service/` (a
**different Python environment** — needs `torch`, `pandas`, `python-dotenv`;
Module 4's `.venv` does not have `torch` installed):

```powershell
python batch_forecast.py --date 2026-07-29
```

This writes `Module3/forecast-service/outputs/forecast_2026-07-29.csv`
(one row per tank; a rejected tank that day gets a blank row and is handled
automatically by Module 4 — see below).

### Step 2 — turn on live mode and run

`main.py` takes no CLI flags — per `config.py`'s own convention, you switch
behaviour by editing `Config`, not the algorithm files. Two ways to do it:

**A. Persistent toggle** — edit `module4/config.py`:
```python
MODULE3_LIVE_FORECAST: bool = True   # was False
RUN_DATE: Optional[str] = "2026-07-29"   # or leave None to always mean "today"
```
then just run:
```powershell
python main.py
```

**B. One-off, without editing the file:**
```powershell
python -c "
from module4.config import Config
from module4.mpc_loop import run_mpc, save_all

cfg = Config(MODULE3_LIVE_FORECAST=True, RUN_DATE='2026-07-29').validate()
results = run_mpc(cfg, season_length=cfg.SEASON_LENGTH, verbose=True)
paths = save_all(results, cfg)
print(paths)
"
```

### Config keys for this mode

| Key | Default | Meaning |
|---|---|---|
| `MODULE3_LIVE_FORECAST` | `False` | `True` → read Module 3's live daily file instead of `data/module3_risk.csv`, **and** today's storage from that same file's `t+1` instead of `data/tank_storage.csv` (see below) |
| `RUN_DATE` | `None` | Which date's `forecast_<date>.csv` to read, and the date consolidated rows get tagged with. `None` → real "today" |
| `MODULE3_FORECAST_OUTPUTS_DIR` | `Module3/forecast-service/outputs/` (auto-resolved) | Normally leave alone |
| `SEASON_LENGTH` | `1` | **Must stay `1`** when `MODULE3_LIVE_FORECAST=True` — enforced by `Config.validate()`, since Module 3 only guarantees a file for the one date you generated |
| `RELOAD_STORAGE_EACH_DAY` | `False` | Separate, pre-existing flag. Ignored when `MODULE3_LIVE_FORECAST=True` (storage always comes from Module 3's `t+1` in that mode instead) |

If Module 3 rejected a tank that day (no forecast/no storage available),
Module 4 doesn't fail — it prints a warning naming the tank(s), uses a
neutral 1/3-1/3-1/3 drought/normal/overflow probability for its risk, and
falls back to `data/tank_storage.csv` for its storage, for that tank only,
that day only.

### Why storage comes from `t+1`, not `tank_storage.csv`

Reading storage as an absolute m³ number from the static `data/tank_storage.csv`
turned out to silently drift out of sync with `data/tank_params.csv`'s S_max
(capacity) for several tanks — e.g. a tank whose file said 29,967 m³ current
storage against a real capacity of only 576 m³. `MODULE3_LIVE_FORECAST=True`
instead derives today's storage from Module 3's own `t+1` (% of capacity, the
same convention `run_simulation.py`'s bulk path already uses), which can't
mathematically exceed 100% of capacity. This is handled automatically —
nothing to configure — but if you ever see storage numbers that don't look
right, check whether the tank in question was a Module-3-rejected fallback
(printed as a warning) before assuming it's a new bug.

### Output — overwritten snapshot *and* per-day history

Every run writes the same flat "latest run" files as before
(`outputs/mpc_decisions.csv`, `topsis_weights_log.csv`, `module3_crosscheck.csv`
— overwritten each time). **Additionally**, since `Config.CONSOLIDATED_OUTPUT`
defaults `True`, each run also *appends* a dated row per tank onto:
```
outputs/consolidated_mpc_decisions.csv
outputs/consolidated_topsis_weights_log.csv
outputs/consolidated_module3_crosscheck.csv
```
tagged with a `date` column (`RUN_DATE`, or real "today" if unset) — this is
what lets repeated `main.py` runs (one per day, in real deployment) build up
a per-day history in a few flat CSVs instead of each run silently overwriting
the last one. Re-running the same `RUN_DATE` twice appends a second copy of
that date's rows rather than deduplicating — same behaviour
`run_simulation.py`'s equivalent consolidated files already have.

There is currently no `consolidated_daily_summary.csv` equivalent for this
mode (that 4th file, on the `run_simulation.py` side, is built from a
`day_summary` shape that only `simulation_runner.py` constructs) — ask if you
want that added too.

---

## Data sources at a glance

| Input | Source today | Config |
|---|---|---|
| Tank storage | `data/tank_storage.csv` (static) **or** Module 3's live `t+1` | `MODULE3_LIVE_FORECAST` (Setting B); `RELOAD_STORAGE_EACH_DAY` otherwise |
| Cascade alpha matrix | **Live** — `Module2/development-history/outputs/alpha_matrix.csv`, read directly, no copy | Nothing to configure; automatic |
| Module 3 risk/forecast | `data/module3_risk.csv` (static) **or** live per-day file | `MODULE3_LIVE_FORECAST` + `RUN_DATE` (Setting B only) |

---

## Known issues (pre-existing, not introduced by this README)

- **`run_simulation.py` without `--forecast` fails.** The built-in default
  (`simulation_data.FORECAST_DATASET_PATH`) points at a filename that doesn't
  exist on disk. Always pass `--forecast` explicitly (examples above do).
- **`data/module4_forecasts_mar_jul_2026_tft.csv` cannot currently be used**
  with `run_simulation.py` — it has a `date` column, but
  `load_forecast_dataset()` hardcodes `forecast_date`, so it raises
  `KeyError: 'forecast_date'`. Use `module4_forecasts_mar_aug_2025.csv`
  (verified working) until this is fixed.
- **`--consolidated-output` is a no-op.** `run_simulation()` only ever sets
  `cfg_season.CONSOLIDATED_OUTPUT = True` when the flag is passed — it never
  sets it `False` — and the `Config` dataclass field it's toggling already
  defaults to `True`. So the `consolidated_*.csv` files get written on every
  run regardless of whether you pass the flag. Verified empirically.
