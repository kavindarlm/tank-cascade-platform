"""
CLI entry point for the forecast-service.

Ingestion commands (init-db, backfill, sanity-check, run, query) have
moved out of this service entirely — those belonged to the SQLite/GEE
pipeline, which Module 1's live-update-service now owns independently.
This service only consumes Module 1's tank_storage.csv output.
"""
import argparse

from inference import run_forecast


def cmd_forecast(args):
    """7-day storage forecast + drought/normal/overflow risk for one
    tank, using the last 14 real days from Module 1's live storage CSV
    (gap-filled if Module 1 hasn't updated recently) plus Open-Meteo
    weather."""
    result = run_forecast(tank_id=args.tank_id, target_date=args.date)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tank storage forecasting service')
    sub = parser.add_subparsers(dest='command', required=True)

    p_forecast = sub.add_parser('forecast', help='7-day storage forecast + risk classification for one tank')
    p_forecast.add_argument('--tank-id', type=int, required=True)
    p_forecast.add_argument('--date', required=True, help='YYYY-MM-DD, forecast is generated for this date forward')
    p_forecast.set_defaults(func=cmd_forecast)

    args = parser.parse_args()
    args.func(args)