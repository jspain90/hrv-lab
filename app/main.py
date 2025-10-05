from __future__ import annotations
import os
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.responses import FileResponse, RedirectResponse
from .core.db import get_db
from .core.settings import settings
from .core import models  # ensures tables are created
from sqlalchemy.orm import Session
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Routers
from .routers import interventions, compliance, metrics, dashboard, dev, refresh, weather, analysis

def create_app() -> FastAPI:
    app = FastAPI(title="HRV-Lab LAN API", version="0.2.0")

    # templates + static
    templates = Jinja2Templates(directory="templates")
    app.state.templates = templates                      # make accessible to routers
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # include routers
    app.include_router(interventions.router)
    app.include_router(compliance.router)
    app.include_router(metrics.router)
    app.include_router(dashboard.router)
    app.include_router(dev.router)
    app.include_router(refresh.router)
    app.include_router(weather.router)
    app.include_router(analysis.router)

    # Root redirect to dashboard
    @app.get("/")
    async def root():
        return RedirectResponse(url="/now")

    # Favicon route
    @app.get("/favicon.ico")
    async def favicon():
        return FileResponse("favicon.ico")

    return app

app = create_app()

# Run with: uvicorn app.main:app --reload --port 8000
