"""
Module 4 validation dataset generator.

Storage caching is no longer patched in here — inference.py's
_load_storage_df() already caches the entire tank_storage.csv in
memory (keyed by file mtime), so every tank/date call in this loop
reuses that same cached DataFrame automatically. The old manual
per-tank storage cache existed to avoid repeated SQLite/CSV round
trips; that job is now handled inside inference.py itself.

Weather and artifact caching are unchanged — Open-Meteo still hits a
live API and is worth caching per-tank, and the model/scalers are
still worth loading once via lru_cache.
"""
import time

import pandas as pd
import requests
from functools import lru_cache

import inference
from inference import run_forecast, INPUT_DAYS


START_DATE = "2026-03-01"
END_DATE = "2026-07-26"

TANK_IDS = list(range(1, 33))


# ==========================================================
# Weather cache (unchanged)
# ==========================================================

FETCH_START = (
    pd.Timestamp(START_DATE)
    - pd.Timedelta(days=INPUT_DAYS + 2)
).strftime("%Y-%m-%d")

FETCH_END = END_DATE

_original_fetch_weather = inference.fetch_weather
_weather_cache = {}


def _load_tank_weather(tank_id, retries=5):
    for attempt in range(retries):
        try:
            print(f"Fetching weather tank {tank_id}", flush=True)
            df = _original_fetch_weather(tank_id, FETCH_START, FETCH_END)
            return df.set_index("date")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 429 and attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"Rate limited. Waiting {wait}s", flush=True)
                time.sleep(wait)
            else:
                raise


def cached_fetch_weather(tank_id, start_date, end_date):
    if tank_id not in _weather_cache:
        _weather_cache[tank_id] = _load_tank_weather(tank_id)
        time.sleep(1)

    full_df = _weather_cache[tank_id]
    mask = (
        (full_df.index >= pd.Timestamp(start_date))
        & (full_df.index <= pd.Timestamp(end_date))
    )
    return full_df.loc[mask].reset_index()


inference.fetch_weather = cached_fetch_weather


# ==========================================================
# Artifact cache (unchanged)
# ==========================================================

_original_load_artifacts = inference.load_artifacts


@lru_cache(maxsize=1)
def cached_load_artifacts():
    print("Loading TFT model artifacts...", flush=True)
    return _original_load_artifacts()


inference.load_artifacts = cached_load_artifacts


# ==========================================================
# Main generator
# ==========================================================

def main():
    rows = []
    dates = pd.date_range(START_DATE, END_DATE)
    total = len(TANK_IDS) * len(dates)
    count = 0

    for tank_id in TANK_IDS:
        print("\n==========================")
        print(f"Tank {tank_id}", flush=True)

        for d in dates:
            count += 1
            print(f"Tank {tank_id} | {d.date()} | {count}/{total}", flush=True)

            result = run_forecast(tank_id=tank_id, target_date=d.strftime("%Y-%m-%d"))

            if result.get("status") == "rejected":
                continue

            storage_pct = result["storage_forecast"]["t+1"]
            storage_mcm = result["storage_forecast_volume"]["t+1"]
            storage_source = result["storage_forecast_source"]["t+1"]

            rows.append({
                "tank_id": result["tank_id"],
                "date": result["forecast_date"],
                "storage": storage_mcm,
                "storage_pct": storage_pct,
                "storage_source": storage_source,
                "t+1": result["storage_forecast"]["t+1"],
                "t+2": result["storage_forecast"]["t+2"],
                "t+3": result["storage_forecast"]["t+3"],
                "t+4": result["storage_forecast"]["t+4"],
                "t+5": result["storage_forecast"]["t+5"],
                "t+6": result["storage_forecast"]["t+6"],
                "t+7": result["storage_forecast"]["t+7"],
                "primary_risk": result["primary_risk"],
                "classifier_risk": result["classifier_risk"],
                "agreement": result["agreement"],
                "prob_drought": result["risk_probabilities"]["drought"],
                "prob_normal": result["risk_probabilities"]["normal"],
                "prob_overflow": result["risk_probabilities"]["overflow"],
                "drought_duration_days": result["drought_duration_days"],
                "overflow_duration_days": result["overflow_duration_days"],
                "confidence": result["confidence"],
                "days_gap": result["days_gap"],
            })

    df = pd.DataFrame(rows)
    output = "module4_forecasts_mar_jul_2026_tft_new.csv"
    df.to_csv(output, index=False)

    print("\n==========================")
    print("Finished")
    print("==========================")
    print(df.head())
    print(f"\nGenerated {len(df)} forecasts")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()