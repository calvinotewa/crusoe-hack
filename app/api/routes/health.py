from fastapi import APIRouter
from sqlalchemy import text

from app.db import engine
from app.services import storage

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/health/deps")
async def health_deps() -> dict:
    """Check Neon Postgres and Neon bucket connectivity."""
    checks: dict[str, str] = {}
    try:
        async with engine.connect() as conn:
            await conn.execute(text("select 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"
    try:
        await storage.ping()
        checks["bucket"] = "ok"
    except Exception as e:
        checks["bucket"] = f"error: {e}"
    return checks
