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
    active: bool = True

class InterventionOut(BaseModel):
    id: int
    name: str
    start_date: dt.date
    duration_weeks: int
    freq_per_week: int
    active: bool
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

class QuickInterventionIn(BaseModel):
    name: str
    start_date: dt.date
    duration_weeks: int
    frequency_per_week: int

class ComplianceEventOut(BaseModel):
    timestamp: str
    intervention_id: int
    intervention_name: str
    class Config:
        from_attributes = True

class DualSeriesOut(BaseModel):
    t: List[str]
    v1: List[float]
    v2: List[float]

class ComplianceStatsOut(BaseModel):
    intervention_id: int
    intervention_name: str
    start_date: str
    end_date: str
    total_expected_compliance: int
    completed_compliance: int
    percent_trial_completed: float
    percent_expected_compliance: float
    percent_completed_compliance: float

class InterventionAnalysisOut(BaseModel):
    intervention_id: int
    intervention_name: str
    start_date: str
    end_date: str
    baseline_start: str
    baseline_end: str
    tp_residual_tau_u: float
    tp_residual_p_value: float
    tp_residual_effect_size: str
    standing_hr_tau_u: float
    standing_hr_p_value: float
    standing_hr_effect_size: str
    percent_compliance: float
    baseline_n: int
    intervention_n: int
