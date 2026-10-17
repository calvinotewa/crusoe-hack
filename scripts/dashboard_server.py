#!/usr/bin/env python
"""Shared Brain dashboard: projects + orchestration runs, HTML UI + JSON API.

    uv run python scripts/dashboard_server.py [--port 8877]     # http://127.0.0.1:8877

Projects (name, goal, uploaded validation CSV) are created in the UI; runs execute
the 5-step loop against a project's dataset. Everything persists in SQLite
(data/dashboard.db) with logical separation: projects -> runs -> generations ->
experiments. Crusoe and you.com store nothing themselves.
"""

import argparse
import asyncio
import csv
import io
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.services.dashboard_store import Store  # noqa: E402
from app.services.harness_runner import HarnessRunner, coerce_row  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "dashboard.db"
HTML_PATH = Path(__file__).resolve().parent / "dashboard.html"
LEGACY_DIR = DATA_DIR / "harness_runs"

store = Store(DB_PATH)
runners: dict[str, HarnessRunner] = {}


@asynccontextmanager
async def lifespan(_app):
    n = await store.mark_interrupted()
    if n:
        print(f"[dashboard] marked {n} interrupted run(s) as stopped")
    yield


app = FastAPI(title="Shared Brain dashboard", lifespan=lifespan)


class RunStart(BaseModel):
    """Start a run. `hypotheses` is a fixed count: every generation generates exactly n."""

    agents: int = Field(default=6, ge=1, le=100)
    hypotheses: int = Field(default=20, ge=1, le=100)
    generations: int = Field(default=2, ge=1, le=10)
    top_k: int = Field(default=3, ge=1, le=10)
    sources: int = Field(default=10, ge=0, le=20)
    concurrency: int = Field(default=4, ge=1, le=10)


class RunContinue(BaseModel):
    """Continue a run's project with MORE generations, inheriting its knowledge:
    the benchmark to beat, the last hypotheses, and the last summary + learnings."""

    generations: int = Field(default=2, ge=1, le=10)
    agents: int | None = Field(default=None, ge=1, le=100)
    hypotheses: int | None = Field(default=None, ge=1, le=100)
    top_k: int | None = Field(default=None, ge=1, le=10)
    sources: int | None = Field(default=None, ge=0, le=20)
    concurrency: int | None = Field(default=None, ge=1, le=10)


# --- html ---------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(HTML_PATH, media_type="text/html")


# --- projects -----------------------------------------------------------


@app.get("/api/projects")
async def list_projects() -> list[dict]:
    return await store.list_projects()


@app.post("/api/projects", status_code=201)
async def create_project(
    name: str = Form(...),
    goal: str = Form(...),
    dataset: UploadFile | None = File(None),  # noqa: B008 (FastAPI idiom)
    csv_text: str | None = Form(None),
) -> dict:
    """Create a project. Upload a CSV file, or paste CSV text; needs a `y` column."""
    content = ""
    if dataset is not None:
        raw = await dataset.read()
        content = raw.decode("utf-8", errors="replace")
    elif csv_text:
        content = csv_text
    if not content.strip():
        raise HTTPException(422, "upload a CSV file or paste CSV text")
    rows = list(csv.DictReader(io.StringIO(content)))
    if not rows or "y" not in rows[0]:
        raise HTTPException(422, "CSV needs a header row and a `y` column")
    columns = [c for c in rows[0] if c and c.strip()]
    project = await store.create_project(name.strip(), goal.strip(), content, columns)
    return {k: v for k, v in project.items() if k != "dataset_csv"}


@app.get("/api/projects/{pid}")
async def get_project(pid: str) -> dict:
    p = await store.get_project(pid)
    if not p:
        raise HTTPException(404, "project not found")
    return p


@app.delete("/api/projects/{pid}")
async def delete_project(pid: str) -> dict:
    if not await store.delete_project(pid):
        raise HTTPException(404, "project not found")
    for rid in [r for r, rn in runners.items() if rn.params.get("project_id") == pid]:
        runners.pop(rid, None)
    return {"deleted": pid}


# --- runs ---------------------------------------------------------------


@app.post("/api/projects/{pid}/runs", status_code=202)
async def start_run(pid: str, body: RunStart) -> dict:
    project = await store.get_project(pid)
    if not project:
        raise HTTPException(404, "project not found")
    params = body.model_dump() | {"project_id": pid}
    run = await store.create_run(pid, params)
    rows = [coerce_row(r) for r in csv.DictReader(io.StringIO(project["dataset_csv"]))]
    runner = HarnessRunner(params, project["goal"], rows, store, run["id"])
    runners[run["id"]] = runner
    asyncio.get_running_loop().create_task(runner.run())
    return run


@app.get("/api/runs/{rid}")
async def get_run(rid: str) -> dict:
    run = await store.get_run(rid)
    if not run:
        raise HTTPException(404, "run not found")
    return run


@app.post("/api/runs/{rid}/stop")
async def stop_run(rid: str) -> dict:
    runner = runners.get(rid)
    if not runner or runner.state["status"] != "running":
        raise HTTPException(409, "no active run")
    for task in asyncio.all_tasks():
        if task.get_coro().__qualname__.startswith("HarnessRunner.run"):
            task.cancel()
    return {"status": "cancelling"}


@app.post("/api/runs/{rid}/continue", status_code=202)
async def continue_run(rid: str, body: RunContinue) -> dict:
    """Invest more in a project: a NEW run in the same project that inherits the
    previous run's knowledge (best score to beat, last hypotheses, last summary
    + web learnings) and runs `body.generations` more generations."""
    prev = await store.get_run(rid)
    if not prev:
        raise HTTPException(404, "run not found")
    project = await store.get_project(prev["project_id"])
    if not project:
        raise HTTPException(404, "project not found")

    params = dict(prev.get("params") or {})
    for k, v in body.model_dump(exclude={"generations"}).items():
        if v is not None:
            params[k] = v
    params["generations"] = body.generations

    # inherit knowledge: the best score across the project is the benchmark to beat
    best = max(
        (
            x
            for x in [
                prev.get("best_score"),
                *(r.get("best_score") for r in project.get("runs", [])),
            ]
            if x is not None
        ),
        default=None,
    )

    # last generation's hypotheses (kept at the fixed count) + memory from its summary
    gens = prev.get("generations") or []
    last = gens[-1] if gens else None
    hyps = [h for h in (last or {}).get("hypotheses", []) if h]
    n = int(params.get("hypotheses", 20))
    while len(hyps) < n:  # keep the fixed count exact
        hyps.append(f"Open question: what else could beat {best}? (slot {len(hyps) + 1})")
    memory = ""
    if last:
        learnings = (
            "\n".join(f"- {s['title']} ({s['url']})" for s in last.get("sources", []))
            or "(no results)"
        )
        memory = (
            f"## Last generation's summary\n{last.get('summary') or ''}\n\n"
            f"## Web learnings\n{learnings}"
        )

    run = await store.create_run(prev["project_id"], params, continued_from=rid)
    rows = [coerce_row(r) for r in csv.DictReader(io.StringIO(project["dataset_csv"]))]
    runner = HarnessRunner(
        params,
        project["goal"],
        rows,
        store,
        run["id"],
        initial_hypotheses=hyps,
        initial_memory=memory,
        initial_best_score=best,
    )
    runners[run["id"]] = runner
    asyncio.get_running_loop().create_task(runner.run())
    return run


def import_legacy_runs() -> int:
    """One-time import of pre-projects JSON runs (data/harness_runs) into the DB.
    Runs are grouped into one project per dataset name."""
    if not LEGACY_DIR.exists():
        return 0
    if asyncio.run(store.list_projects()):
        return 0  # already migrated
    count = 0
    project_by_ds: dict[str, str] = {}
    for f in sorted(LEGACY_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except ValueError:
            continue
        ds = d.get("params", {}).get("dataset", "imported")
        csv_path = ROOT / "examples" / ("churn_synthetic.csv" if ds == "churn" else "")
        if not csv_path.is_file():
            continue
        if ds not in project_by_ds:
            csv_text = csv_path.read_text()
            rows = list(csv.DictReader(io.StringIO(csv_text)))
            proj = asyncio.run(
                store.create_project(
                    f"{ds} (imported)", d.get("goal") or "(imported run)", csv_text, list(rows[0])
                )
            )
            project_by_ds[ds] = proj["id"]
        run = asyncio.run(store.create_run(project_by_ds[ds], d.get("params", {})))
        asyncio.run(
            store.update_run(
                run["id"],
                status=d.get("status", "completed"),
                current_phase="done",
                generation=d.get("generation", 0),
                best_score=d.get("best_score"),
                events=d.get("events", []),
                finished_at=d.get("finished_at"),
            )
        )
        for g in d.get("generations", []):
            asyncio.run(store.add_generation(run["id"], g, g.get("experiments", [])))
        count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8877)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    imported = import_legacy_runs()
    if imported:
        print(f"[dashboard] imported {imported} legacy run(s) into the project store")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
