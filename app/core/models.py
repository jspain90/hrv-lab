from __future__ import annotations
import datetime as dt
from typing import Optional, List
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, DateTime, Boolean, Float, ForeignKey, UniqueConstraint, Text
from .db import engine

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    role: Mapped[Optional[str]] = mapped_column(String(50), default="owner")
    interventions: Mapped[List["Intervention"]] = relationship(back_populates="user")

class Intervention(Base):
    __tablename__ = "interventions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String(200))
    start_date: Mapped[dt.date] = mapped_column()
    duration_weeks: Mapped[int] = mapped_column(Integer)
    freq_per_week: Mapped[int] = mapped_column(Integer)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    user: Mapped["User"] = relationship(back_populates="interventions")
    compliance_events: Mapped[List["ComplianceEvent"]] = relationship(back_populates="intervention")
    quick_tokens: Mapped[List["QuickToken"]] = relationship(back_populates="intervention")

class ComplianceEvent(Base):
    __tablename__ = "compliance_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    intervention_id: Mapped[int] = mapped_column(ForeignKey("interventions.id"))
    ts: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    value_bool: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[Optional[str]] = mapped_column(String(500))
    source: Mapped[str] = mapped_column(String(40), default="api")
    intervention: Mapped["Intervention"] = relationship(back_populates="compliance_events")

class DailyAggregate(Base):
    __tablename__ = "daily_aggregates"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[dt.date] = mapped_column(index=True)
    posture: Mapped[Optional[str]] = mapped_column(String(20))
    metric: Mapped[str] = mapped_column(String(60))
    agg: Mapped[str] = mapped_column(String(20))
    value: Mapped[float] = mapped_column(Float)
    window: Mapped[Optional[str]] = mapped_column(String(40))

class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    intervention_id: Mapped[int] = mapped_column(ForeignKey("interventions.id"))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    config_json: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="done")

class QuickToken(Base):
    __tablename__ = "quick_tokens"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    intervention_id: Mapped[int] = mapped_column(ForeignKey("interventions.id"))
    token: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    expires_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    uses: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    purpose: Mapped[str] = mapped_column(String(40), default="compliance_quick")
    intervention: Mapped["Intervention"] = relationship(back_populates="quick_tokens")

Base.metadata.create_all(engine)
