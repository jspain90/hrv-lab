from __future__ import annotations
import os
from fastapi import FastAPI, UploadFile, File, Depends
from .core.db import get_db
from .core.settings import settings
from .core import models  # ensures tables are created
from .services.ingest import ingest_content
from sqlalchemy.orm import Session

# Routers
from .routers import interventions, compliance, metrics, dashboard, dev

def create_app() -> FastAPI:
    app = FastAPI(title="HRV-Lab LAN API", version="0.2.0")

    # file upload endpoint
    @app.post("/ingest/file")
    def ingest_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
        content = file.file.read()
        inserted = ingest_content(content, file.filename, db)
        return {"ok": True, "readings": inserted}

    # include routers
    app.include_router(interventions.router)
    app.include_router(compliance.router)
    app.include_router(metrics.router)
    app.include_router(dashboard.router)
    app.include_router(dev.router)

    # startup: create required dirs
    @app.on_event("startup")
    def _startup_dirs():
        os.makedirs(settings.inbox_dir, exist_ok=True)
        os.makedirs(settings.archive_dir, exist_ok=True)

    return app

app = create_app()

# Run with: uvicorn app.main:app --reload --port 8000
