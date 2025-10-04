from __future__ import annotations
from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.orm import Session
from ..core.db import get_db
from pydantic import BaseModel
import datetime as dt

router = APIRouter(prefix="/weather", tags=["weather"])

class WeatherDay(BaseModel):
    date: str
    temperature_max_c: float | None
    apparent_temperature_max_c: float | None
    surface_pressure_hpa: float | None

@router.get("/daily", response_model=list[WeatherDay])
def get_weather(
    start: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    end: str | None = Query(None, description="End date (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """Get daily weather data for date range."""
    query = "SELECT date, temperature_max_c, apparent_temperature_max_c, surface_pressure_hpa FROM weather_daily"

    conditions = []
    if start:
        conditions.append(f"date >= '{start}'")
    if end:
        conditions.append(f"date <= '{end}'")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY date ASC"

    result = db.execute(text(query))
    rows = result.fetchall()

    return [
        WeatherDay(
            date=row[0],
            temperature_max_c=row[1],
            apparent_temperature_max_c=row[2],
            surface_pressure_hpa=row[3]
        )
        for row in rows
    ]

@router.get("/summary")
def weather_summary(db: Session = Depends(get_db)):
    """Get summary statistics about weather data."""
    result = db.execute(text("""
        SELECT
            COUNT(*) as total_days,
            MIN(date) as earliest_date,
            MAX(date) as latest_date,
            AVG(temperature_max_c) as avg_temp_max,
            AVG(surface_pressure_hpa) as avg_pressure
        FROM weather_daily
    """))
    row = result.fetchone()

    return {
        "total_days": row[0],
        "earliest_date": row[1],
        "latest_date": row[2],
        "avg_temperature_max_c": round(row[3], 2) if row[3] else None,
        "avg_surface_pressure_hpa": round(row[4], 2) if row[4] else None
    }
