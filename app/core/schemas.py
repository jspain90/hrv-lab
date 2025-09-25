from __future__ import annotations
import datetime as dt
from typing import Optional, List
from pydantic import BaseModel

class InterventionIn(BaseModel):
    user_name: str = "owner"
    name: str
    start_date: dt.date
    duration_weeks: int
    freq_per_week: int
    expected_metric: str
    expected_direction: str
    hypothesis_text: Optional[str] = None
    posture_filter: Optional[str] = None
    time_of_day_filter: Optional[str] = None

class InterventionOut(BaseModel):
    id: int
    name: str
    start_date: dt.date
    duration_weeks: int
    freq_per_week: int
    expected_metric: str
    expected_direction: str
    hypothesis_text: Optional[str]
    class Config:
        from_attributes = True

class ComplianceIn(BaseModel):
    intervention_id: int
    ts: Optional[dt.datetime] = None
    value_bool: bool = True
    notes: Optional[str] = None

class SeriesOut(BaseModel):
    t: List[str]
    v: List[float]
