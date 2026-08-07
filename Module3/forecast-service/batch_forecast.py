"""
Batch forecast — runs run_forecast() for every tank on one date and
writes a single flat CSV.

Default output:
    outputs/forecast_<date>.csv

Usage:
    python batch_forecast.py --date 2026-07-29

Optional custom output:
    python batch_forecast.py --date 2026-07-29 \
        --output custom_folder/my_forecast.csv
"""

import argparse
import csv
import os

import pandas as pd

from config import TANKS_CSV_PATH
from inference import run_forecast


# ============================================================
# CSV column order (matches Module 4 expectation)
# ============================================================

FIELDNAMES = [
    "tank_id",
    "date",
    "storage",
    "storage_pct",
    "storage_source",
    "t+1",
    "t+2",
    "t+3",
    "t+4",
    "t+5",
    "t+6",
    "t+7",
    "primary_risk",
    "classifier_risk",
    "agreement",
    "prob_drought",
    "prob_normal",
    "prob_overflow",
    "drought_duration_days",
    "overflow_duration_days",
    "confidence",
    "days_gap",
]


# ============================================================
# Convert forecast dictionary into Module 4 row
# ============================================================

def _as_non_negative_percent(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, numeric)


def flatten_result(result):

    # --------------------------------------------------------
    # Handle rejected forecasts
    # --------------------------------------------------------

    if result.get("status") == "rejected":

        return {
            "tank_id": result["tank_id"],
            "date": "",
            "storage": "",
            "storage_pct": "",
            "storage_source": "",
            "t+1": "",
            "t+2": "",
            "t+3": "",
            "t+4": "",
            "t+5": "",
            "t+6": "",
            "t+7": "",
            "primary_risk": "rejected",
            "classifier_risk": "",
            "agreement": "",
            "prob_drought": "",
            "prob_normal": "",
            "prob_overflow": "",
            "drought_duration_days": "",
            "overflow_duration_days": "",
            "confidence": "",
            "days_gap": result.get("days_gap", ""),
        }

    return {

        "tank_id":
            result["tank_id"],

        "date":
            result["forecast_date"],

        # Current day's forecast = t+1
        "storage":
            result["storage_forecast_volume"]["t+1"],

        "storage_pct":
            _as_non_negative_percent(result["storage_forecast"]["t+1"]),

        "storage_source":
            result["storage_forecast_source"]["t+1"],

        # 7-day forecast (%)
        "t+1":
            _as_non_negative_percent(result["storage_forecast"]["t+1"]),

        "t+2":
            _as_non_negative_percent(result["storage_forecast"]["t+2"]),

        "t+3":
            _as_non_negative_percent(result["storage_forecast"]["t+3"]),

        "t+4":
            _as_non_negative_percent(result["storage_forecast"]["t+4"]),

        "t+5":
            _as_non_negative_percent(result["storage_forecast"]["t+5"]),

        "t+6":
            _as_non_negative_percent(result["storage_forecast"]["t+6"]),

        "t+7":
            _as_non_negative_percent(result["storage_forecast"]["t+7"]),

        # Risk
        "primary_risk":
            result["primary_risk"],

        "classifier_risk":
            result["classifier_risk"],

        "agreement":
            result["agreement"],

        # Probabilities
        "prob_drought":
            result["risk_probabilities"]["drought"],

        "prob_normal":
            result["risk_probabilities"]["normal"],

        "prob_overflow":
            result["risk_probabilities"]["overflow"],

        # Durations
        "drought_duration_days":
            result["drought_duration_days"],

        "overflow_duration_days":
            result["overflow_duration_days"],

        # Confidence
        "confidence":
            result["confidence"],

        # Data quality
        "days_gap":
            result["days_gap"],
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Batch forecast all tanks"
    )

    parser.add_argument(
        "--date",
        required=True,
        help="Forecast date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV (default outputs/forecast_<date>.csv)"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    if args.output is None:

        os.makedirs("outputs", exist_ok=True)

        args.output = f"outputs/forecast_{args.date}.csv"

    else:

        output_dir = os.path.dirname(args.output)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    print(f"Output file: {args.output}\n")

    # --------------------------------------------------------
    # Load tank list
    # --------------------------------------------------------

    tanks_df = pd.read_csv(TANKS_CSV_PATH)

    tank_ids = sorted(tanks_df["tank_id"].unique())

    print(f"Forecasting {len(tank_ids)} tanks...\n")

    rows = []

    # --------------------------------------------------------
    # Forecast every tank
    # --------------------------------------------------------

    for tank_id in tank_ids:

        print(f"Forecasting tank_id={tank_id}...")

        try:

            result = run_forecast(
                tank_id=int(tank_id),
                target_date=args.date,
            )

            rows.append(flatten_result(result))

        except Exception as e:

            print(f"  FAILED: {e}")

            rows.append({

                "tank_id": tank_id,
                "date": "",
                "storage": "",
                "storage_pct": "",
                "storage_source": "",
                "t+1": "",
                "t+2": "",
                "t+3": "",
                "t+4": "",
                "t+5": "",
                "t+6": "",
                "t+7": "",
                "primary_risk": "error",
                "classifier_risk": "",
                "agreement": "",
                "prob_drought": "",
                "prob_normal": "",
                "prob_overflow": "",
                "drought_duration_days": "",
                "overflow_duration_days": "",
                "confidence": "",
                "days_gap": "",
            })

    # --------------------------------------------------------
    # Write CSV
    # --------------------------------------------------------

    with open(args.output, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(
        1
        for row in rows
        if row["primary_risk"] not in ("error", "rejected")
    )

    print("\n" + "=" * 50)
    print(f"Done: {ok_count}/{len(tank_ids)} tanks forecasted successfully.")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()