"""
tank_weather.py

Downloads historical weather data for all tanks using Open-Meteo API.

Features:
- Downloads weather data for all tanks
- Saves individual tank weather files
- Handles failed downloads
- Automatically retries failed tanks
- Creates final all_tanks_combined.csv

Weather features:
    precipitation_sum
    temperature_2m_mean
    et0

Date range:
    2020-01-01 to 2026-01-01
"""

import os
import time
import openmeteo_requests
import pandas as pd
import requests_cache

from retry_requests import retry


# ============================================================
# Setup Open-Meteo API client
# ============================================================

cache_session = requests_cache.CachedSession(
    ".cache",
    expire_after=-1
)

retry_session = retry(
    cache_session,
    retries=5,
    backoff_factor=0.2
)

openmeteo = openmeteo_requests.Client(
    session=retry_session
)


# ============================================================
# Configuration
# ============================================================

START_DATE = "2020-01-01"
END_DATE = "2026-01-01"

TANK_FILE = "tanks.csv"

OUTPUT_FOLDER = "tank_outputs"

COMBINED_FILE = "all_tanks_combined.csv"

WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)


os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)



# ============================================================
# Download weather for one tank
# ============================================================

def download_tank_weather(row):

    tank_id = row["tank_id"]
    tank_name = row["tank_name"]

    latitude = row["latitude"]
    longitude = row["longitude"]


    params = {

        "latitude": latitude,

        "longitude": longitude,

        "start_date": START_DATE,

        "end_date": END_DATE,

        "daily": [
            "precipitation_sum",
            "temperature_2m_mean",
            "et0_fao_evapotranspiration"
        ],

        "timezone": "auto"
    }


    responses = openmeteo.weather_api(
        WEATHER_URL,
        params=params
    )


    response = responses[0]

    daily = response.Daily()


    dates = pd.date_range(
        start=pd.to_datetime(
            daily.Time()
            + response.UtcOffsetSeconds(),
            unit="s"
        ),

        end=pd.to_datetime(
            daily.TimeEnd()
            + response.UtcOffsetSeconds(),
            unit="s"
        ),

        freq=pd.Timedelta(
            seconds=daily.Interval()
        ),

        inclusive="left"
    )


    tank_data = pd.DataFrame({

        "tank_id": tank_id,

        "tank_name": tank_name,

        "date": dates,

        "precipitation_sum":
            daily.Variables(0).ValuesAsNumpy(),

        "temperature_2m_mean":
            daily.Variables(1).ValuesAsNumpy(),

        "et0":
            daily.Variables(2).ValuesAsNumpy(),


        # Future Module 3 features
        "storage": None,

        "upstream_inflow": None,

        "target": None
    })


    return tank_data




# ============================================================
# Save individual tank file
# ============================================================

def save_tank_file(tank_data):

    tank_name = tank_data["tank_name"].iloc[0]


    safe_name = (
        tank_name
        .replace(" ", "_")
        .replace("/", "_")
    )


    path = os.path.join(
        OUTPUT_FOLDER,
        f"{safe_name}.csv"
    )


    tank_data.to_csv(
        path,
        index=False
    )





# ============================================================
# Download tanks
# ============================================================

def download_tanks(tanks_df):

    downloaded = []

    failed = []


    for index, row in tanks_df.iterrows():

        tank_name = row["tank_name"]


        print(
            f"[{index+1}/{len(tanks_df)}] "
            f"Downloading {tank_name}..."
        )


        try:

            tank_data = download_tank_weather(row)


            save_tank_file(tank_data)


            downloaded.append(tank_data)


            print(
                f"  ✓ Done — "
                f"{len(tank_data)} days downloaded"
            )


        except Exception as e:


            print(
                f"  ✗ FAILED for {tank_name}: {e}"
            )


            failed.append(tank_name)



    return downloaded, failed





# ============================================================
# Main pipeline
# ============================================================

def main():


    # -----------------------------
    # Load tanks
    # -----------------------------

    tanks = pd.read_csv(
        TANK_FILE
    )


    print(
        f"Total tanks to download: {len(tanks)}"
    )


    print("\nStarting first download...\n")



    # -----------------------------
    # First attempt
    # -----------------------------

    downloaded, failed = download_tanks(
        tanks
    )



    # -----------------------------
    # Save initial combined data
    # -----------------------------

    if downloaded:

        combined = pd.concat(
            downloaded,
            ignore_index=True
        )


        combined.to_csv(
            COMBINED_FILE,
            index=False
        )

    else:

        combined = pd.DataFrame()



    # ========================================================
    # Retry failed tanks
    # ========================================================

    if failed:


        print("\n" + "="*50)

        print(
            f"{len(failed)} tanks failed."
        )


        print(
            "Waiting 60 seconds before retry..."
        )


        time.sleep(60)



        failed_df = tanks[
            tanks["tank_name"].isin(failed)
        ].reset_index(drop=True)



        print(
            f"\nRetrying {len(failed_df)} failed tanks...\n"
        )



        retry_downloaded, still_failed = download_tanks(
            failed_df
        )



        # -----------------------------
        # Merge retry results
        # -----------------------------

        if retry_downloaded:


            retry_data = pd.concat(
                retry_downloaded,
                ignore_index=True
            )


            combined = pd.concat(
                [
                    combined,
                    retry_data
                ],

                ignore_index=True
            )


            combined = combined.sort_values(
                [
                    "tank_id",
                    "date"
                ]
            ).reset_index(
                drop=True
            )



            combined.to_csv(
                COMBINED_FILE,
                index=False
            )


        else:

            still_failed = failed



    else:

        still_failed = []



    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "="*50)

    print(
        f"✓ Successful tanks: "
        f"{combined['tank_id'].nunique()}"
    )


    print(
        f"✗ Failed tanks: "
        f"{len(still_failed)}"
    )


    if still_failed:

        print(
            "Failed list:"
        )

        for tank in still_failed:

            print(
                f" - {tank}"
            )


    print(
        f"\nTotal rows: {len(combined)}"
    )


    print(
        f"Saved file: {COMBINED_FILE}"
    )




# ============================================================
# Run pipeline
# ============================================================

if __name__ == "__main__":

    main()