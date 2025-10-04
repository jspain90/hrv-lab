"""
Weather data fetcher using Open-Meteo API for Lawrence, KS.

Fetches:
- Temperature (high) - temperature_2m_max
- Apparent temperature (high) - apparent_temperature_max
- Surface pressure (mean) - surface_pressure_mean

Usage:
    python weather_fetch.py --start 2025-04-01 --end 2025-09-30
    python weather_fetch.py --days 7  # fetch last 7 days
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional
import time

try:
    import requests
except ImportError:
    print("ERROR: requests library required. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)

# Lawrence, KS coordinates
LAWRENCE_LAT = 38.9717
LAWRENCE_LON = -95.2353

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "app").exists():
            return p
    return Path.cwd()

DB_PATH = Path(
    _repo_root() / "hrv_lab.sqlite3"
)

def ensure_weather_table(cx: sqlite3.Connection):
    """Create weather_daily table if it doesn't exist."""
    cx.execute("""
        CREATE TABLE IF NOT EXISTS weather_daily (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL UNIQUE,
            location TEXT NOT NULL,
            temperature_max_c REAL,
            apparent_temperature_max_c REAL,
            surface_pressure_hpa REAL,
            fetched_at TEXT NOT NULL
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_weather_date ON weather_daily(date)")
    cx.commit()

def fetch_weather_data(start_date: str, end_date: str, latitude: float = LAWRENCE_LAT, longitude: float = LAWRENCE_LON) -> dict:
    """
    Fetch weather data from Open-Meteo API.

    Args:
        start_date: ISO format date string (YYYY-MM-DD)
        end_date: ISO format date string (YYYY-MM-DD)
        latitude: Location latitude
        longitude: Location longitude

    Returns:
        JSON response from API
    """
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,apparent_temperature_max,surface_pressure_mean",
        "temperature_unit": "celsius",
        "timezone": "America/Chicago"  # Lawrence, KS timezone
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()

def insert_weather_records(cx: sqlite3.Connection, data: dict, location: str = "Lawrence, KS"):
    """Insert weather records into database."""
    if "daily" not in data:
        raise ValueError("No daily data in API response")

    daily = data["daily"]
    dates = daily.get("time", [])
    temp_max = daily.get("temperature_2m_max", [])
    apparent_temp_max = daily.get("apparent_temperature_max", [])
    pressure_mean = daily.get("surface_pressure_mean", [])

    fetched_at = datetime.utcnow().isoformat()

    inserted = 0
    updated = 0

    for i, d in enumerate(dates):
        temp = temp_max[i] if i < len(temp_max) else None
        apparent = apparent_temp_max[i] if i < len(apparent_temp_max) else None
        pressure = pressure_mean[i] if i < len(pressure_mean) else None

        # Insert or replace
        cur = cx.execute(
            "SELECT id FROM weather_daily WHERE date = ?", (d,)
        )
        exists = cur.fetchone()

        if exists:
            cx.execute(
                """UPDATE weather_daily
                   SET temperature_max_c = ?, apparent_temperature_max_c = ?,
                       surface_pressure_hpa = ?, fetched_at = ?
                   WHERE date = ?""",
                (temp, apparent, pressure, fetched_at, d)
            )
            updated += 1
        else:
            cx.execute(
                """INSERT INTO weather_daily
                   (date, location, temperature_max_c, apparent_temperature_max_c,
                    surface_pressure_hpa, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (d, location, temp, apparent, pressure, fetched_at)
            )
            inserted += 1

    cx.commit()
    return inserted, updated

def fetch_and_store(start_date: str, end_date: str, db_path: Path = DB_PATH, dry_run: bool = False):
    """Main function to fetch and store weather data."""
    print(f"Fetching weather data from {start_date} to {end_date}")

    try:
        data = fetch_weather_data(start_date, end_date)
    except requests.RequestException as e:
        print(f"ERROR: Failed to fetch weather data: {e}", file=sys.stderr)
        return False

    if dry_run:
        print("DRY RUN: Would insert/update the following dates:")
        if "daily" in data and "time" in data["daily"]:
            for d in data["daily"]["time"]:
                print(f"  {d}")
        return True

    with sqlite3.connect(db_path) as cx:
        ensure_weather_table(cx)
        inserted, updated = insert_weather_records(cx, data)
        print(f"OK Inserted: {inserted}, Updated: {updated}")

    return True

def get_missing_dates(db_path: Path = DB_PATH) -> list[str]:
    """Find dates in HRV data that don't have weather data."""
    with sqlite3.connect(db_path) as cx:
        ensure_weather_table(cx)

        # Get all unique dates from hrv_results
        hrv_dates = cx.execute(
            "SELECT DISTINCT DATE(timestamp) FROM hrv_results ORDER BY DATE(timestamp)"
        ).fetchall()

        # Get all dates from weather_daily
        weather_dates = cx.execute(
            "SELECT date FROM weather_daily"
        ).fetchall()

        weather_set = {row[0] for row in weather_dates}
        missing = [row[0] for row in hrv_dates if row[0] not in weather_set]

        return missing


def backfill_missing_weather(db_path: Path = DB_PATH) -> int:
    """
    Backfill weather data for dates that have HRV readings but no weather data.

    Designed for automatic refresh workflow - finds missing dates and fetches them.

    Args:
        db_path: Path to SQLite database

    Returns:
        Number of weather dates fetched (0 if no missing dates or error)
    """
    try:
        missing = get_missing_dates(db_path)

        if not missing:
            print("No missing weather dates")
            return 0

        # Get date range (Open-Meteo allows range queries)
        start_date = missing[0]
        end_date = missing[-1]

        print(f"Backfilling {len(missing)} weather dates from {start_date} to {end_date}")

        # Fetch weather data
        try:
            data = fetch_weather_data(start_date, end_date)
        except requests.RequestException as e:
            print(f"WARNING: Weather API request failed: {e}")
            return 0

        # Store in database
        with sqlite3.connect(db_path) as cx:
            ensure_weather_table(cx)
            inserted, updated = insert_weather_records(cx, data)
            print(f"OK Weather: Inserted {inserted}, Updated {updated}")

        return inserted + updated

    except Exception as e:
        print(f"ERROR backfilling weather: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Fetch weather data from Open-Meteo API")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, help="Fetch last N days (alternative to start/end)")
    parser.add_argument("--backfill", action="store_true", help="Fetch missing dates based on HRV data")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database")
    parser.add_argument("--check-missing", action="store_true", help="Show missing weather dates")

    args = parser.parse_args()

    if args.check_missing:
        missing = get_missing_dates()
        if missing:
            print(f"Missing weather data for {len(missing)} dates:")
            for d in missing[:10]:
                print(f"  {d}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
        else:
            print("No missing dates!")
        return

    if args.backfill:
        missing = get_missing_dates()
        if not missing:
            print("No missing dates to backfill")
            return

        # Group consecutive dates to minimize API calls
        # Open-Meteo allows requests for date ranges
        start_date = missing[0]
        end_date = missing[-1]

        print(f"Backfilling {len(missing)} dates from {start_date} to {end_date}")
        success = fetch_and_store(start_date, end_date, dry_run=args.dry_run)
        return 0 if success else 1

    if args.days:
        end_date = date.today()
        start_date = end_date - timedelta(days=args.days - 1)
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()
    elif args.start and args.end:
        start_str = args.start
        end_str = args.end
    else:
        parser.print_help()
        print("\nERROR: Must specify --start/--end, --days, or --backfill", file=sys.stderr)
        return 1

    success = fetch_and_store(start_str, end_str, dry_run=args.dry_run)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
