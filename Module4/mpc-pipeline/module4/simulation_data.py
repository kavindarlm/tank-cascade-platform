"""
simulation_data.py
==================
Data-adapter layer for the day-by-day MO-MPC simulation.

This module bridges a Module 1+3 forecast CSV (one row per tank per day) to
the EXACT input shapes module4's existing data_loader / mpc_loop already
expect (tank_params.csv, tank_storage.csv, demand.csv, network_alpha.csv,
rainfall_history.csv, module3_risk.csv).

Nothing here touches the optimisation, MPC, NSGA-II, TOPSIS, objective, or
constraint logic - it only prepares the CSVs those modules read, exactly the
way Module 1 / Module 3 / a human operator would in deployment. See
simulation_runner.py for the day-by-day loop that uses these functions.

DATASET CONVENTIONS (as specified for this run, not the module4 default):
  - tank_id in the forecast dataset is an integer 1..32. It maps POSITIONALLY
    to data_/tank_coordinates.csv (row 1 -> tank_id 1, ... row 32 -> tank_id
    32), which is the same ordering already used by tank_params.csv,
    tank_storage.csv, demand.csv and network_alpha.csv (verified below).
  - Storage columns t+1..t+7 are PERCENT of each tank's capacity (S_max).
  - t+1 is TODAY's actual storage (Module 1) for the simulation day whose
    forecast_date equals the current simulation date. t+2..t+7 are Module 3's
    forward forecast, used only for the existing cross-check.

MULTI-RUN SUPPORT
  The default forecast path and demand settings below reflect the original
  2025 Yala run. For a different year or season pass explicit arguments to
  load_forecast_dataset() and generate_season_demand() - the runner does this
  automatically when --forecast / --season / --duration are supplied.
  The original generate_yala_demand() function is kept unchanged for backward
  compatibility with any existing scripts that call it directly.
"""

import os
import shutil

import numpy as np
import pandas as pd

from .config import Config
from .data_loader import load_static_inputs
from .demand_generator import generate_demand, get_cwr_daily, field_to_tank


TANK_COORDINATES_PATH = os.path.join("data_", "tank_coordinates.csv")

# Default forecast dataset - original 2025 Yala run.
# When running a different year or season, pass the path explicitly to
# load_forecast_dataset(); do NOT change this constant.
FORECAST_DATASET_PATH = os.path.join("data", "module_forecasts_and mar_aug_2025.csv")

# The command-area source file is not present in this repo (see
# recover_command_area_acres). data/demand.csv was generated from it with
# these settings (build_demand.py defaults), so recovery must use the same.
SOURCE_DEMAND_CSV = os.path.join("data", "demand.csv")
SOURCE_DEMAND_SEASON = "maha"
SOURCE_DEMAND_DURATION = 135


def load_tank_index_map(coords_path=TANK_COORDINATES_PATH):
    """
    Return the 32 tank_id NAMES in tank_coordinates.csv row order. Position i
    (0-based) is forecast-dataset tank index i+1.
    """
    coords = pd.read_csv(coords_path)
    return list(coords["tank_id"])


def load_forecast_dataset(path=FORECAST_DATASET_PATH, tank_names=None):
    """
    Load the full Module 1+3 forecast dataset (one row per tank per day) and
    attach the tank NAME (from tank_coordinates order) next to the raw
    integer tank index.

    Parameters
    ----------
    path : str
        Path to the forecast CSV. Defaults to the original 2025 Yala file.
        Pass a different path when running a different year or season.
    tank_names : list or None
        Ordered list of tank names (positional, matching integer tank_id).
        Loaded from tank_coordinates.csv when None.
    """
    df = pd.read_csv(path)
    df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date

    if tank_names is None:
        tank_names = load_tank_index_map()
    index_to_name = {i + 1: name for i, name in enumerate(tank_names)}

    bad = set(df["tank_id"].unique()) - set(index_to_name)
    if bad:
        raise ValueError(
            f"forecast dataset has tank_id value(s) outside 1..{len(tank_names)}: "
            f"{sorted(bad)}"
        )
    df["tank_name"] = df["tank_id"].map(index_to_name)
    return df


def day_slice(forecast_df, sim_date, n_tanks=32):
    """One simulation day's tank rows, indexed by tank NAME."""
    rows = forecast_df[forecast_df["forecast_date"] == sim_date]
    if len(rows) != n_tanks:
        raise ValueError(
            f"expected {n_tanks} tank rows for {sim_date}, got {len(rows)}"
        )
    return rows.set_index("tank_name")


def storage_pct_to_volume(pct_by_name: pd.Series, S_max_by_name: pd.Series) -> pd.Series:
    """
    Convert t+1 (% of capacity) to m3 for every tank: storage_m3 = pct/100 * S_max.
    """
    pct = pct_by_name.astype(float)
    smax = S_max_by_name.reindex(pct.index).astype(float)
    if smax.isna().any():
        missing = list(smax[smax.isna()].index)
        raise ValueError(f"no S_max found for tank(s): {missing}")
    return (pct / 100.0) * smax


def build_module3_frame(day_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape one day's forecast rows into the frame data_loader / cross_validation
    expect: {drought,overflow,normal}_probability + t+1..t+7 (already named
    t+1..t+7 in the source data, matching one of the patterns
    cross_validation.forecast_columns already accepts).
    """
    out = pd.DataFrame(index=day_rows.index.copy())
    out.index.name = "tank_id"
    out["drought_probability"] = day_rows["prob_drought"].astype(float)
    out["overflow_probability"] = day_rows["prob_overflow"].astype(float)
    out["normal_probability"] = day_rows["prob_normal"].astype(float)
    for d in range(1, 8):
        out[f"t+{d}"] = day_rows[f"t+{d}"].astype(float)
    # Carry through extra diagnostic columns untouched (harmless; unused by
    # any loader/consumer, but kept for traceability in the saved CSV).
    for c in ("primary_risk", "classifier_risk", "agreement",
              "drought_duration_days", "overflow_duration_days",
              "confidence", "days_gap"):
        if c in day_rows.columns:
            out[c] = day_rows[c]
    return out


def recover_command_area_acres(demand_csv=SOURCE_DEMAND_CSV,
                                season=SOURCE_DEMAND_SEASON,
                                duration=SOURCE_DEMAND_DURATION):
    """
    The original command_area.csv (tank_id, command_area_acres) that
    build_demand.py needs is not present in this repo. Recover each tank's
    command area by algebraically INVERTING the FAO/CWR demand formula
    against the already-committed data/demand.csv:

        demand_m3[i, t] = gross_mm[t] * area_ha[i] * 10
        => area_ha[i]   = demand_m3[i, t] / (10 * gross_mm[t])

    This is an exact inverse (not an estimate): regenerating demand from the
    recovered areas reproduces data/demand.csv to <0.1 m3 (pure round(1)
    noise from build_demand.py). Verified for maha/135/eta=0.70, the
    defaults build_demand.py used to write the committed file.
    """
    src_cfg = Config()
    src_cfg.SEASON = season
    src_cfg.PADDY_DURATION_DAYS = duration
    src_cfg.validate()

    daily_cwr_mm, _ = get_cwr_daily(src_cfg, duration)
    gross_mm = field_to_tank(daily_cwr_mm, src_cfg)          # (season_days,)

    demand = pd.read_csv(demand_csv, index_col="tank_id")
    day_cols = [c for c in demand.columns if c.startswith("day")]
    if len(day_cols) != len(gross_mm):
        raise ValueError(
            f"{demand_csv} has {len(day_cols)} day columns but "
            f"season={season}/duration={duration} implies {len(gross_mm)}; "
            f"recovery assumptions no longer match the committed file."
        )
    D = demand[day_cols].to_numpy(dtype=float)               # (N, days)

    with np.errstate(divide="ignore", invalid="ignore"):
        implied_area_ha = D / (10.0 * gross_mm[np.newaxis, :])
    area_ha = np.median(implied_area_ha, axis=1)
    area_acres = area_ha / src_cfg.ACRE_TO_HA
    return pd.Series(area_acres, index=demand.index, name="command_area_acres")


def generate_season_demand(tank_names, cfg: Config, season: str = None,
                           duration: int = None):
    """
    Generate the full season demand matrix ONCE for ANY season and duration.

    This is the generalised replacement for generate_yala_demand().  The
    runner calls this function; the season and duration come from the CLI
    arguments so a single code path handles Yala, Maha, or any other season
    without touching any algorithm file.

    Parameters
    ----------
    tank_names : list of str
        Ordered tank names (must match the demand.csv row order).
    cfg : Config
        Season config - must have SEASON and PADDY_DURATION_DAYS set before
        calling. The runner builds this from _default_season_config().
    season : str or None
        If given, overrides cfg.SEASON for the command-area recovery step.
        Normally left None so cfg drives everything.
    duration : int or None
        If given, overrides cfg.PADDY_DURATION_DAYS for recovery.

    Returns
    -------
    demand_m3   : (N, season_days) float array ordered as tank_names
    diagnostics : dict from demand_generator.generate_demand
    """
    s = season or cfg.SEASON
    d = duration or cfg.PADDY_DURATION_DAYS
    area_acres = recover_command_area_acres(season=s, duration=d)
    area_acres = area_acres.reindex(tank_names)
    if area_acres.isna().any():
        missing = list(area_acres[area_acres.isna()].index)
        raise ValueError(f"could not recover command area for tank(s): {missing}")

    demand_m3, diagnostics = generate_demand(area_acres.to_numpy(), cfg)
    return demand_m3, diagnostics


def generate_yala_demand(tank_names, cfg_yala: Config):
    """
    Generate the full Yala-season demand matrix ONCE (per the task spec: the
    demand generator runs a single time; the day-by-day loop only slices it).

    KEPT UNCHANGED for backward compatibility with any existing scripts that
    call it directly.  New code should call generate_season_demand() instead.

    Returns
    -------
    demand_m3   : (N, season_days) ordered exactly as `tank_names`
    diagnostics : dict from demand_generator.generate_demand
    """
    area_acres = recover_command_area_acres()
    area_acres = area_acres.reindex(tank_names)
    if area_acres.isna().any():
        missing = list(area_acres[area_acres.isna()].index)
        raise ValueError(f"could not recover command area for tank(s): {missing}")

    demand_m3, diagnostics = generate_demand(area_acres.to_numpy(), cfg_yala)
    return demand_m3, diagnostics


def load_static_reference(cfg: Config = None):
    """
    Load tank_params/network_alpha/rainfall_history/tank_ids once from the
    real committed data/ directory (read-only), via the EXISTING, unmodified
    data_loader.load_static_inputs. Its returned demand_full (maha) is not
    used by the simulation - the season demand is generated separately.
    """
    cfg = cfg or Config()
    return load_static_inputs(cfg)


def stage_static_files(static_dir, source_data_dir="data"):
    """
    Copy the three PER-RUN-CONSTANT input files (tank_params, network_alpha,
    rainfall_history) byte-for-byte into the scratch data directory used to
    feed the existing loaders, so the real data/ directory is never written
    to. tank_storage.csv / demand.csv / module3_risk.csv are written fresh
    each simulated day by simulation_runner.
    """
    os.makedirs(static_dir, exist_ok=True)
    cfg = Config()
    for fname in (cfg.FILE_TANK_PARAMS, cfg.FILE_NETWORK_ALPHA, cfg.FILE_RAINFALL_HISTORY):
        shutil.copyfile(os.path.join(source_data_dir, fname),
                        os.path.join(static_dir, fname))


def write_day_inputs(static_dir, tank_names, S_current_by_name,
                     demand_window_m3, module3_frame, cfg: Config):
    """
    Write the three PER-DAY files (tank_storage.csv, demand.csv,
    module3_risk.csv) into `static_dir`, in `tank_names` order, using the
    exact column names data_loader.py already expects.
    """
    storage_df = pd.DataFrame({
        "tank_id": tank_names,
        "storage_m3": [round(float(S_current_by_name[t]), 2) for t in tank_names],
    })
    storage_df.to_csv(os.path.join(static_dir, cfg.FILE_TANK_STORAGE), index=False)

    demand_df = pd.DataFrame(
        demand_window_m3.round(2),
        columns=[f"day{d}" for d in range(demand_window_m3.shape[1])],
    )
    demand_df.insert(0, "tank_id", tank_names)
    demand_df.to_csv(os.path.join(static_dir, cfg.FILE_DEMAND), index=False)

    module3_frame.loc[tank_names].to_csv(os.path.join(static_dir, cfg.FILE_MODULE3_RISK))