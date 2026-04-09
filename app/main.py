"""FastAPI application entry point.

Configures the application lifespan, mounts static file directories,
registers API and UI routers, and defines global exception handlers.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import api_router, ui_router
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Configure logging, ensure directories exist, and log startup/shutdown events."""
    settings = get_settings()
    configure_logging(settings)
    settings.ensure_directories()
    logging.getLogger(__name__).info(
        "Starting %s (embedding_mode=%s)", settings.app_name, settings.embedding_mode
    )
    yield
    logging.getLogger(__name__).info("Shutting down")


app = FastAPI(title="YouTube Daily Comments Intelligence", version="1.0.0", lifespan=lifespan)
app.mount("/static/app", StaticFiles(directory="frontend/dist", check_dir=False), name="app_static")
app.include_router(api_router)
app.include_router(ui_router)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    """Handle known application errors and return a 400 JSON response."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler that logs unexpected errors and returns a 500 JSON response."""
    logging.getLogger(__name__).exception("Unhandled server error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
