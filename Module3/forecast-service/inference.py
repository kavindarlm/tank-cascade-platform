"""
Module 3 real-time inference. Called via:
    python main.py forecast --tank-id <id> --date <YYYY-MM-DD>

Storage observations now come directly from Module 1's live-update CSV
(tank_storage.csv) instead of a SQLite database — no db.py dependency.

observation_date logic mirrors storage_generate.py EXACTLY:
    date     = Sentinel-1 grid/record date (kept for reference only)
    s2_date  = Sentinel-2 acquisition date when optical substitution
               occurred (populated only for substituted rows)
    observation_date = s2_date.fillna(date)  <- the true measurement date,
               used for every time-based operation (window filtering,
               PCHIP interpolation, exact-date override lookups).
Keeping this identical between training and inference is what stops
timestamp-anchoring skew between the two pipelines.
"""
import json
import os
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import requests
import torch
from scipy.interpolate import PchipInterpolator

from config import FORECASTING_MODEL_DIR, OPEN_METEO_API_BASE, TANKS_CSV_PATH, TANK_STORAGE_CSV_PATH
from models.tft_model import TFTForecastModel


INPUT_DAYS = 14
GAP_REJECT_THRESHOLD_DAYS = 14

FEATURE_COLS_REG = ["precipitation_sum", "temperature_2m_mean", "et0", "upstream_inflow",
                     "storage", "catchment_area_km2", "downstream_demand", "total_outflow_alpha"]
FEATURE_COLS_CLS = [c for c in FEATURE_COLS_REG if c != "storage"]


def load_artifacts():
    model = TFTForecastModel(input_size_full=8, input_size_cls=7)
    model.load_state_dict(torch.load(f"{FORECASTING_MODEL_DIR}/tft_model.pth", map_location="cpu"))
    model.eval()

    scaler_reg = joblib.load(f"{FORECASTING_MODEL_DIR}/feature_scaler_tft.pkl")
    scaler_cls = joblib.load(f"{FORECASTING_MODEL_DIR}/feature_scaler_cls_tft.pkl")

    with open(f"{FORECASTING_MODEL_DIR}/risk_thresholds.json") as f:
        thresholds = json.load(f)

    connectivity_df = pd.read_csv("data/connectivity.csv")
    connectivity_map = {}
    for _, row in connectivity_df.iterrows():
        connectivity_map.setdefault(row["to_tank"], []).append(
            (row["from_tank"], row["alpha"])
        )

    tank_peak_volume = thresholds.get("tank_peak_volume_mcm", {})

    return model, scaler_reg, scaler_cls, thresholds, connectivity_map, tank_peak_volume


def volume_to_pct(volume_mcm, tank_id, tank_peak_volume):
    """Same normalization as storage_generate.py: % of this tank's
    own observed peak volume. Clipped in case a live reading exceeds
    the training-period peak."""
    peak = tank_peak_volume.get(str(tank_id))
    if not peak or peak <= 0:
        return None
    return min((volume_mcm / peak) * 100, 100.0)


def pct_to_volume(pct, tank_id, tank_peak_volume):
    """Inverse of volume_to_pct(). Converts a model-predicted storage
    percentage back into absolute volume_mcm units, so predicted values
    live in the same units as real Module 1 readings and can sit side
    by side (or be swapped for one another) in the output CSV."""
    peak = tank_peak_volume.get(str(tank_id))
    if not peak or peak <= 0:
        return None
    return (pct / 100.0) * peak


# ------------------------------------------------------------------
# CSV storage layer (replaces db.py / get_tank_volume_range for this
# service). Loaded once per file-mtime, not once per tank per call —
# tank_storage.csv is append-only and can grow large, so re-parsing
# it from scratch for every upstream tank in every forecast call
# would be wasteful and get worse over time.
# ------------------------------------------------------------------
_storage_df_cache = {}


def _resolve_observation_date(df):
    """Identical logic to storage_generate.py's observation_date fix.
    'date' stays as the grid/record key; 'observation_date' is what
    every time-based operation downstream should use."""
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce").dt.normalize()

    if "s2_date" in df.columns:
        df["s2_date"] = pd.to_datetime(df["s2_date"], format="mixed", errors="coerce").dt.normalize()
        df["observation_date"] = df["s2_date"].fillna(df["date"])
    else:
        df["observation_date"] = df["date"]

    return df


def _load_storage_df(csv_path=TANK_STORAGE_CSV_PATH):
    """Loads tank_storage.csv, resolves observation_date, and flags
    valid observations. Cached by file mtime — re-read automatically
    whenever Module 1's live-update service appends new rows, but not
    re-parsed on every single tank/date lookup within one run."""
    mtime = os.path.getmtime(csv_path)
    cached = _storage_df_cache.get(csv_path)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    df = pd.read_csv(csv_path)
    df = _resolve_observation_date(df)

    # volume_mcm == 0 means "no usable satellite pixel for this tank on
    # this date", not "tank measured at 0% storage" — same distinction
    # db.py's backfill_from_csv used to document but that was never
    # actually implemented in this file. Implementing it here since
    # removing the DB layer means this filtering has to live somewhere.
    df["is_valid_observation"] = df["volume_mcm"].notna() & (df["volume_mcm"] != 0)

    _storage_df_cache[csv_path] = (mtime, df)
    return df


def get_observed_storage(tank_id, date_str, tank_peak_volume, storage_df):
    """
    EXACT observation_date lookup only. This is intentionally NOT the
    same thing as get_daily_pct_for_tank() / gap_fill_storage() below,
    which fill in every day of a window (real OR PCHIP-interpolated) to
    build model INPUT features.

    This function answers a narrower question: "does Module 1 have a
    real satellite-derived measurement on this exact calendar date?"
    (matched against observation_date, i.e. s2_date when substitution
    occurred, otherwise the grid date). It never interpolates and never
    falls back to a neighbouring date.

    Returns (volume_mcm, pct) if a real observation exists for
    `date_str`, otherwise None.
    """
    target = pd.Timestamp(date_str)
    rows = storage_df[
        (storage_df["tank_id"] == tank_id)
        & (storage_df["observation_date"] == target)
        & storage_df["is_valid_observation"]
    ]
    if rows.empty:
        return None

    # If two grid dates both resolve to the same observation_date
    # (rare, but possible), average them the same way
    # interpolate_storage_pchip() does in storage_generate.py.
    volume_mcm = rows["volume_mcm"].mean()
    pct = volume_to_pct(volume_mcm, tank_id, tank_peak_volume)
    if pct is None:
        return None
    return volume_mcm, pct


def gap_fill_storage(known_daily_pct, window_start, window_end):
    """
    Fill missing dates using PCHIP interpolation.

    IMPORTANT:
    Real Module 1 observations are never modified.
    If a date has an observed value, that value is returned exactly.
    """
    all_dates = pd.date_range(window_start, window_end, freq="D")

    if not known_daily_pct:
        return {}

    obs_dates_sorted = sorted(known_daily_pct.keys())

    if len(obs_dates_sorted) < 2:
        only_val = known_daily_pct[obs_dates_sorted[0]]
        result = {}
        for d in all_dates:
            date_str = d.strftime("%Y-%m-%d")
            result[date_str] = known_daily_pct.get(date_str, only_val)
        return result

    ref_date = pd.Timestamp(obs_dates_sorted[0])

    obs_days = np.array([
        (pd.Timestamp(d) - ref_date).days
        for d in obs_dates_sorted
    ])
    obs_vals = np.array([
        known_daily_pct[d]
        for d in obs_dates_sorted
    ])
    all_days = np.array([
        (d - ref_date).days
        for d in all_dates
    ])

    pchip = PchipInterpolator(obs_days, obs_vals, extrapolate=True)
    interpolated = np.clip(pchip(all_days), 0, 100)

    result = {}
    for d, value in zip(all_dates, interpolated):
        date_str = d.strftime("%Y-%m-%d")
        if date_str in known_daily_pct:
            result[date_str] = known_daily_pct[date_str]
        else:
            result[date_str] = float(value)

    return result


def get_daily_pct_for_tank(tank_id, window_start, window_end, tank_peak_volume, storage_df):
    """Filter + convert + PCHIP-interpolate a SINGLE tank's daily storage
    percentage for the given window, anchored on observation_date (not
    grid date). Returns (daily_pct, num_real_obs) — the observation
    count is used for honest confidence scoring, since PCHIP fills
    every day regardless of how sparse the real data was."""
    mask = (
        (storage_df["tank_id"] == tank_id)
        & (storage_df["observation_date"] >= window_start)
        & (storage_df["observation_date"] <= window_end)
        & storage_df["is_valid_observation"]
    )
    rows = storage_df.loc[mask]

    known_daily_pct = {}
    if not rows.empty:
        grouped = rows.groupby("observation_date")["volume_mcm"].mean()
        for obs_date, volume_mcm in grouped.items():
            pct = volume_to_pct(volume_mcm, tank_id, tank_peak_volume)
            if pct is not None:
                known_daily_pct[obs_date.strftime("%Y-%m-%d")] = pct

    daily_pct = gap_fill_storage(known_daily_pct, window_start, window_end)
    return daily_pct, len(known_daily_pct)


def compute_upstream_inflow(tank_id, window_start, window_end, connectivity_map, tank_peak_volume, storage_df):
    """For each day in the window, sum (upstream_tank_storage x alpha x
    UPSTREAM_FACTOR) across every real upstream neighbor of tank_id —
    each upstream tank's storage comes from ITS OWN fetched data, not
    a stand-in from the requested tank.

    UPSTREAM_FACTOR must match storage_generate.py EXACTLY — the saved
    feature_scaler.pkl was fit on training data generated with that
    specific constant."""
    UPSTREAM_FACTOR = 2.5  # must mirror storage_generate.py exactly

    upstream_tanks = connectivity_map.get(tank_id, [])
    all_dates = pd.date_range(window_start, window_end, freq="D")

    if not upstream_tanks:
        return {d.strftime("%Y-%m-%d"): 0.0 for d in all_dates}

    upstream_daily_pct = {
        upstream_id: get_daily_pct_for_tank(
            upstream_id, window_start, window_end, tank_peak_volume, storage_df
        )[0]
        for upstream_id, _alpha in upstream_tanks
    }

    inflow_by_date = {}
    for d in all_dates:
        date_str = d.strftime("%Y-%m-%d")
        total_inflow = 0.0
        for upstream_id, alpha in upstream_tanks:
            upstream_pct = upstream_daily_pct[upstream_id].get(date_str)
            if upstream_pct is not None:
                total_inflow += upstream_pct * alpha * UPSTREAM_FACTOR
        inflow_by_date[date_str] = total_inflow

    return inflow_by_date


def fetch_weather(tank_id, start_date, end_date):
    tanks_df = pd.read_csv(TANKS_CSV_PATH)
    tank_row = tanks_df[tanks_df["tank_id"] == tank_id].iloc[0]

    resp = requests.get(f"{OPEN_METEO_API_BASE}/archive", params={
        "latitude": tank_row["latitude"], "longitude": tank_row["longitude"],
        "start_date": start_date, "end_date": end_date,
        "daily": ["precipitation_sum", "temperature_2m_mean", "et0_fao_evapotranspiration"],
        "timezone": "auto"
    })
    resp.raise_for_status()
    daily = resp.json()["daily"]
    return pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "precipitation_sum": daily["precipitation_sum"],
        "temperature_2m_mean": daily["temperature_2m_mean"],
        "et0": daily["et0_fao_evapotranspiration"],
    })


def run_forecast(tank_id, target_date):
    model, scaler_reg, scaler_cls, thresholds, connectivity_map, tank_peak_volume = load_artifacts()

    # Loaded once per call (cached by mtime across calls in the same
    # process — see _load_storage_df), then threaded down to every
    # function that needs it, instead of each one re-reading the CSV.
    storage_df = _load_storage_df()

    target = pd.Timestamp(target_date)
    window_start = target - timedelta(days=INPUT_DAYS)
    window_end = target - timedelta(days=1)

    daily_pct, num_real_obs = get_daily_pct_for_tank(
        tank_id, window_start, window_end, tank_peak_volume, storage_df
    )

    if not daily_pct:
        return {"status": "rejected", "reason": "no storage data available for this tank",
                "tank_id": tank_id, "days_gap": INPUT_DAYS}

    days_gap = max(0, INPUT_DAYS - (num_real_obs * 4))
    days_gap = min(days_gap, INPUT_DAYS)

    if num_real_obs < 1:
        return {"status": "rejected", "reason": "insufficient recent storage data",
                "tank_id": tank_id, "days_gap": INPUT_DAYS}

    upstream_inflow_by_date = compute_upstream_inflow(
        tank_id, window_start, window_end, connectivity_map, tank_peak_volume, storage_df
    )

    weather_df = fetch_weather(tank_id, str(window_start.date()), str(window_end.date()))

    tank_metadata_df = pd.read_csv("data/tank_metadata.csv")
    tank_network_df = pd.read_csv("data/tank_network_features.csv")

    tank_meta_row = tank_metadata_df[tank_metadata_df["tank_id"] == tank_id].iloc[0]
    tank_network_row = tank_network_df[tank_network_df["tank_id"] == tank_id].iloc[0]

    rows_reg = []
    for _, w in weather_df.iterrows():
        date_str = w["date"].strftime("%Y-%m-%d")
        rows_reg.append({
            "precipitation_sum": w["precipitation_sum"],
            "temperature_2m_mean": w["temperature_2m_mean"],
            "et0": w["et0"],
            "upstream_inflow": upstream_inflow_by_date.get(date_str, 0.0),
            "storage": daily_pct.get(date_str, 0.0),
            "catchment_area_km2": tank_meta_row["catchment_area_km2"],
            "downstream_demand": tank_network_row["downstream_demand"],
            "total_outflow_alpha": tank_network_row["total_outflow_alpha"],
        })
    window_df = pd.DataFrame(rows_reg)

    X_reg = window_df[FEATURE_COLS_REG].values.reshape(1, INPUT_DAYS, -1)
    X_cls = window_df[FEATURE_COLS_CLS].values.reshape(1, INPUT_DAYS, -1)

    X_reg_scaled = scaler_reg.transform(X_reg.reshape(-1, X_reg.shape[-1])).reshape(X_reg.shape)
    X_cls_scaled = scaler_cls.transform(X_cls.reshape(-1, X_cls.shape[-1])).reshape(X_cls.shape)

    with torch.no_grad():
        storage_forecast, risk_logits = model(
            torch.tensor(X_reg_scaled, dtype=torch.float32),
            torch.tensor(X_cls_scaled, dtype=torch.float32)
        )
    storage_forecast = storage_forecast.numpy()[0]
    risk_probs = torch.softmax(risk_logits, dim=-1).numpy()[0]

    forecast_dates = [
        (target + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(len(storage_forecast))
    ]

    storage_forecast_volume = {}
    storage_forecast_source = {}

    for i, date_str in enumerate(forecast_dates):
        observed = get_observed_storage(tank_id, date_str, tank_peak_volume, storage_df)

        if observed is not None:
            obs_volume, obs_pct = observed
            storage_forecast[i] = obs_pct
            storage_forecast_volume[f"t+{i+1}"] = obs_volume
            storage_forecast_source[f"t+{i+1}"] = "observed"
        else:
            storage_forecast_volume[f"t+{i+1}"] = pct_to_volume(
                float(storage_forecast[i]), tank_id, tank_peak_volume
            )
            storage_forecast_source[f"t+{i+1}"] = "predicted"

    drought_thresh = thresholds["drought_threshold_per_tank"][str(tank_id)]
    overflow_thresh = thresholds["overflow_threshold_per_tank"][str(tank_id)]

    def classify(pct):
        if pct < drought_thresh:
            return "drought"
        elif pct > overflow_thresh:
            return "overflow"
        return "normal"

    primary_risk = classify(storage_forecast[0])
    classifier_risk = ["drought", "normal", "overflow"][int(np.argmax(risk_probs))]

    return {
        "tank_id": tank_id,
        "forecast_date": target_date,
        "storage_forecast": {f"t+{i+1}": round(float(v), 2) for i, v in enumerate(storage_forecast)},
        "storage_forecast_volume": {
            k: (round(float(v), 4) if v is not None else None)
            for k, v in storage_forecast_volume.items()
        },
        "storage_forecast_source": storage_forecast_source,
        "primary_risk": primary_risk,
        "classifier_risk": classifier_risk,
        "agreement": primary_risk == classifier_risk,
        "risk_probabilities": {
            "drought": round(float(risk_probs[0]), 3),
            "normal": round(float(risk_probs[1]), 3),
            "overflow": round(float(risk_probs[2]), 3),
        },
        "drought_duration_days": int((storage_forecast < drought_thresh).sum()),
        "overflow_duration_days": int((storage_forecast > overflow_thresh).sum()),
        "confidence": round(max(0.0, 1.0 - days_gap / INPUT_DAYS), 2),
        "days_gap": days_gap,
    }