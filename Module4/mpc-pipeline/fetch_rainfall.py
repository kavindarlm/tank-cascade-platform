"""
fetch_rainfall.py
================
Build rainfall_history.csv for Module 4 by pulling daily precipitation from the
Open-Meteo Historical Weather API at each tank's coordinates.

This is the input to the scenario generator: for each tank a Gamma distribution
is fitted to this rainfall history, converted to catchment inflow, and sampled
K times per MPC step.

Output format (exactly what data_loader expects):
    tank_id, d0, d1, ..., d{H-1}
one row per tank, each column one day of daily precipitation total (mm).

Open-Meteo is free and needs no API key. Historical/archive endpoint:
    https://archive-api.open-meteo.com/v1/archive
    params: latitude, longitude, start_date, end_date, daily=precipitation_sum

Two modes.

1. SINGLE WINDOW (original)
       python fetch_rainfall.py --end 2024-08-31 --days 184

2. MULTI-YEAR SEASONAL CLIMATOLOGY (recommended)
       python fetch_rainfall.py --years 2022 2023 2024 --season 03-01 08-31

   Fetches the same calendar window from several past years and CONCATENATES
   them into one history. Three Yala seasons -> 3 x 184 = 552 columns.

   Why concatenate rather than average day-by-day? The history is only ever
   used to FIT a distribution (wet-day probability, Gamma shape and scale) -
   it is never replayed as a time series. Averaging three years element-wise
   would smear a storm that fell on day 40 of 2022 into a third of a storm in
   every year, destroying exactly the episodic wet/dry structure the
   Bernoulli-Gamma exists to capture. Concatenating preserves every real day
   and triples the sample size for the fit.

   It also removes the hindsight objection: a distribution built from Yala
   2022-2024 uses only information available before the 2025 season starts, so
   the controller is never fitted on rainfall that had not yet happened.

Usage
-----
    python fetch_rainfall.py --years 2022 2023 2024      # Yala climatology
    python fetch_rainfall.py --years 2020 2021 2022 2023 2024
    python fetch_rainfall.py --end 2024-08-31 --days 184 # single window
    python fetch_rainfall.py --coords /path/to/tank_coordinates.csv
"""

import argparse
import time
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import urllib.request
import urllib.parse
import json

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
COORDS_DEFAULT = "data_/tank_coordinates.csv"


def fetch_one(lat, lon, start_date, end_date, retries=3):
    """
    Fetch a daily precipitation series for one coordinate.

    Returns
    -------
    (dates, precip_mm) : two lists of equal length, or (None, None) on failure.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "auto",
    }
    url = ARCHIVE_URL + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            precip = daily.get("precipitation_sum", [])
            # Open-Meteo returns null for missing days; treat as 0 mm
            precip = [0.0 if v is None else float(v) for v in precip]
            return dates, precip
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))   # back off and retry
            else:
                print(f"    ! failed after {retries} attempts: {e}")
                return None, None


def _in_season(iso_date, m0, d0, m1, d1):
    """True if an ISO date falls inside the (MM-DD .. MM-DD) seasonal window."""
    d = date.fromisoformat(iso_date)
    return (m0, d0) <= (d.month, d.day) <= (m1, d1)


def fetch_multi_year(args):
    """
    Fetch one calendar window from several years and concatenate.

    One request per tank spanning the whole period, then the seasonal days are
    filtered out locally. That is 32 requests rather than 32 x n_years, which
    matters on a free API with no key.
    """
    coords = pd.read_csv(args.coords)
    for col in ("tank_id", "latitude", "longitude"):
        if col not in coords.columns:
            raise ValueError(f"coordinates file must have a '{col}' column")

    m0, d0 = (int(x) for x in args.season[0].split("-"))
    m1, d1 = (int(x) for x in args.season[1].split("-"))
    years = sorted(args.years)
    span_start = date(years[0], m0, d0)
    span_end = date(years[-1], m1, d1)

    print(f"Fetching seasonal climatology")
    print(f"  years  : {', '.join(str(y) for y in years)}")
    print(f"  season : {args.season[0]} -> {args.season[1]}")
    print(f"  span   : {span_start.isoformat()} -> {span_end.isoformat()}")
    print(f"  tanks  : {len(coords)}")
    print(f"  source : Open-Meteo archive API\n")

    series = {}
    kept_dates = None
    for _, row in coords.iterrows():
        tid = row["tank_id"]
        dates, precip = fetch_one(row["latitude"], row["longitude"],
                                  span_start.isoformat(), span_end.isoformat())
        if precip is None:
            print(f"  {tid:32s} FAILED")
            continue
        pairs = [(dt, p) for dt, p in zip(dates, precip)
                 if date.fromisoformat(dt).year in years
                 and _in_season(dt, m0, d0, m1, d1)]
        if not pairs:
            print(f"  {tid:32s} no days in season window")
            continue
        sel_dates = [p[0] for p in pairs]
        vals = [p[1] for p in pairs]
        series[tid] = vals
        if kept_dates is None:
            kept_dates = sel_dates

        per_year = []
        for y in years:
            yv = [p for dt, p in pairs if date.fromisoformat(dt).year == y]
            per_year.append(f"{y}:{sum(yv):.0f}mm")
        print(f"  {tid:32s} {len(vals):3d} days  mean {np.mean(vals):.2f} mm/day"
              f"   [{'  '.join(per_year)}]")
        time.sleep(0.4)

    if not series:
        raise RuntimeError("No data fetched - check connectivity and coordinates.")

    min_len = min(len(v) for v in series.values())
    rows = [[tid] + [round(x, 2) for x in vals[:min_len]]
            for tid, vals in series.items()]
    cols = ["tank_id"] + [f"d{i}" for i in range(min_len)]
    df = pd.DataFrame(rows, columns=cols)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    R = df[cols[1:]].to_numpy(dtype=float)
    print(f"\nrainfall_history.csv written -> {args.out}")
    print(f"  shape        : {df.shape[0]} tanks x {min_len} days "
          f"({len(years)} seasons concatenated)")
    print(f"  mean daily   : {R.mean():.2f} mm/day")
    print(f"  days >= 1 mm : {np.mean(R >= 1.0) * 100:.1f}%")

    # Per-year spread: a single anomalous year should not be allowed to set the
    # distribution unnoticed.
    n_per = min_len // len(years)
    print("\n  per-season totals (mean across tanks):")
    for k, y in enumerate(years):
        blk = R[:, k * n_per:(k + 1) * n_per]
        print(f"    {y}  {blk.sum(axis=1).mean():7.1f} mm   "
              f"{blk.mean():.2f} mm/day")
    print(f"\n  Set cfg.RAINFALL_HISTORY_DAYS = {min_len}")
    print("  (documentation only - the loader reads whatever columns exist)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", default=COORDS_DEFAULT,
                    help="CSV with tank_id, latitude, longitude")
    ap.add_argument("--days", type=int, default=90,
                    help="length of history window (must match "
                         "cfg.RAINFALL_HISTORY_DAYS)")
    ap.add_argument("--end", default=None,
                    help="end date YYYY-MM-DD (default: today). Open-Meteo "
                         "archive lags ~5 days, so 'today' may return a short "
                         "series; use an explicit past date for a full window.")
    ap.add_argument("--years", type=int, nargs="+", default=None,
                    help="fetch this calendar window from each of these years "
                         "and concatenate (e.g. --years 2022 2023 2024)")
    ap.add_argument("--season", nargs=2, default=["03-01", "08-31"],
                    metavar=("MM-DD", "MM-DD"),
                    help="seasonal window used with --years "
                         "(default 03-01 08-31 = Yala)")
    ap.add_argument("--out", default="data/rainfall_history.csv")
    args = ap.parse_args()

    if args.years:
        return fetch_multi_year(args)

    coords = pd.read_csv(args.coords)
    for col in ("tank_id", "latitude", "longitude"):
        if col not in coords.columns:
            raise ValueError(f"coordinates file must have a '{col}' column")

    end = date.fromisoformat(args.end) if args.end else date.today()
    start = end - timedelta(days=args.days - 1)
    print(f"Fetching {args.days} days of daily precipitation")
    print(f"  window: {start.isoformat()} -> {end.isoformat()}")
    print(f"  tanks : {len(coords)}")
    print(f"  source: Open-Meteo archive API\n")

    series = {}
    ref_dates = None
    for _, row in coords.iterrows():
        tid = row["tank_id"]
        dates, precip = fetch_one(row["latitude"], row["longitude"],
                                  start.isoformat(), end.isoformat())
        if precip is None:
            print(f"  {tid:32s} FAILED")
            continue
        series[tid] = precip
        if ref_dates is None:
            ref_dates = dates
        print(f"  {tid:32s} {len(precip):3d} days  "
              f"(total {sum(precip):6.1f} mm, mean {np.mean(precip):.2f} mm/day)")
        time.sleep(0.4)   # be polite to the free API

    if not series:
        raise RuntimeError("No data fetched - check connectivity and coordinates.")

    # Align all tanks to the shortest series length (in case a few days differ)
    min_len = min(len(v) for v in series.values())
    rows = []
    for tid, precip in series.items():
        rows.append([tid] + [round(x, 2) for x in precip[:min_len]])
    cols = ["tank_id"] + [f"d{i}" for i in range(min_len)]
    df = pd.DataFrame(rows, columns=cols)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nrainfall_history.csv written -> {args.out}")
    print(f"  shape: {df.shape[0]} tanks x {min_len} days")
    if min_len < args.days:
        print(f"  NOTE: got {min_len} days, fewer than the {args.days} requested "
              f"(archive lag). Set cfg.RAINFALL_HISTORY_DAYS = {min_len} or pick an "
              f"earlier --end date.")


if __name__ == "__main__":
    main()