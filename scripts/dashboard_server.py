#!/usr/bin/env python
"""Shared Brain dashboard: HTML UI + JSON API over the instrumented harness runner.

    uv run python scripts/dashboard_server.py [--port 8877]

Then open http://127.0.0.1:8877. The page shows orchestration status by phase,
streams progress events, shows per-generation results as they land, and has a
Start button that triggers the full 5-step loop automatically. Every run is
persisted to data/harness_runs/<id>.json (Crusoe itself stores nothing).

Only needs CRUSOE_API_KEY in .env; no Postgres/bucket/Modal.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.services.harness_runner import HarnessRunner, load_dataset  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "harness_runs"
HTML_PATH = Path(__file__).resolve().parent / "dashboard.html"

app = FastAPI(title="Shared Brain dashboard")

# One active run at a time (MVP: in-process asyncio task, JSON persistence).
active: HarnessRunner | None = None


class StartBody(BaseModel):
    """Start a run. `hypotheses` is a fixed count: every generation of the run
    generates exactly that many (default 20)."""

    dataset: str = Field(default="churn", pattern="^(churn|ab)$")
    agents: int = Field(default=6, ge=1, le=100)
    hypotheses: int = Field(default=20, ge=1, le=100)
    generations: int = Field(default=2, ge=1, le=10)
    top_k: int = Field(default=3, ge=1, le=10)
    sources: int = Field(default=5, ge=0, le=20)
    concurrency: int = Field(default=4, ge=1, le=10)
    goal: str | None = None


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(HTML_PATH, media_type="text/html")


@app.get("/api/state")
async def state() -> dict:
    if active is None:
        return {"active": None}
    return {"active": active.state}


@app.post("/api/start", status_code=202)
async def start(body: StartBody) -> dict:
    global active
    if active is not None and active.state["status"] == "running":
        raise HTTPException(409, "a run is already active; stop it first")
    params = body.model_dump()
    _, rows = load_dataset(params["dataset"])
    if not rows or "y" not in rows[0]:
        raise HTTPException(500, f"dataset {params['dataset']} has no rows/y column")
    active = HarnessRunner(params, rows, DATA_DIR)
    asyncio.get_running_loop().create_task(active.run())
    return {"id": active.state["id"], "params": params}


@app.post("/api/stop")
async def stop() -> dict:
    if active is None or active.state["status"] != "running":
        raise HTTPException(409, "no active run")
    # The runner records the cancellation in its finally-block and persists.
    for task in asyncio.all_tasks():
        if task.get_coro().__qualname__.startswith("HarnessRunner.run"):
            task.cancel()
    return {"status": "cancelling"}


@app.get("/api/runs")
async def runs() -> list[dict]:
    out = []
    for f in sorted(DATA_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
        try:
            d = json.loads(f.read_text())
            out.append(
                {
                    "id": d.get("id"),
                    "dataset": d.get("params", {}).get("dataset"),
                    "status": d.get("status"),
                    "best_score": d.get("best_score"),
                    "generations": len(d.get("generations", [])),
                    "started_at": d.get("started_at"),
                }
            )
        except (ValueError, OSError):
            continue
    return out


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    path = (DATA_DIR / f"{run_id}.json").resolve()
    if DATA_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "run not found")
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8877)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
