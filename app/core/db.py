from __future__ import annotations
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .settings import settings

engine = create_engine(
    f"sqlite:///{settings.db_path}",
    future=True,
    echo=False,
    connect_args={"timeout": 30},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, future=True)

# dependency
from typing import Generator

def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
