"""
Central config for the forecast-service. Reads from .env so every team
member can use their own paths without editing code or committing
secrets.

Note: GEE/ingestion-specific settings (GEE_PROJECT_ID, VOLUME_K, etc.)
live in Module 1's live-update-service config now, not here — this
service only consumes Module 1's output CSV, it doesn't run GEE itself.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Module 1's live-updated storage observations (tank_storage.csv).
# This file is append-only — Module 1 adds new rows daily, never
# overwrites, so a full historical lookback is always available here.
TANK_STORAGE_CSV_PATH = os.getenv(
    "TANK_STORAGE_CSV_PATH",
    "../../Module1/live-update-service/data/tank_storage.csv"
)

TANKS_CSV_PATH = os.getenv('TANKS_CSV_PATH', 'data/tanks.csv')

# Module 3 forecasting settings
FORECASTING_MODEL_DIR = os.getenv('FORECASTING_MODEL_DIR', 'artifacts')
OPEN_METEO_API_BASE = os.getenv('OPEN_METEO_API_BASE', 'https://archive-api.open-meteo.com/v1')