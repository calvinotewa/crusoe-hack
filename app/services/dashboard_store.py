"""SQLite store for the dashboard: projects -> runs -> generations -> experiments.

Logical separation is the schema: every run hangs off its project row via foreign
keys, so one project's data never mixes with another's. SQLite via sqlite3 run in
a worker thread (same pattern as storage.py's boto3). File: data/dashboard.db.
"""

import asyncio
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    goal       TEXT NOT NULL,
    dataset_csv TEXT NOT NULL,
    num_rows   INTEGER NOT NULL,
    columns    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    params       TEXT NOT NULL,
    status       TEXT NOT NULL,
    current_phase TEXT,
    generation   INTEGER NOT NULL DEFAULT 0,
    best_score   REAL,
    events       TEXT NOT NULL DEFAULT '[]',
    error        TEXT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    continued_from TEXT
);
CREATE TABLE IF NOT EXISTS run_generations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    gen         INTEGER NOT NULL,
    best_score  REAL,
    num_experiments INTEGER NOT NULL DEFAULT 0,
    summary     TEXT,
    query       TEXT,
    sources     TEXT NOT NULL DEFAULT '[]',
    hypotheses  TEXT NOT NULL DEFAULT '[]',
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_experiments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL REFERENCES run_generations(id) ON DELETE CASCADE,
    name         TEXT,
    score        REAL,
    rmse         REAL,
    expression   TEXT,
    hypothesis   TEXT,
    rationale    TEXT
);
"""


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


class Store:
    """All dashboard persistence. Methods are async; blocking sqlite runs in threads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        try:  # migrate pre-`continued_from` databases (CREATE TABLE is IF NOT EXISTS)
            conn.execute("ALTER TABLE runs ADD COLUMN continued_from TEXT")
        except sqlite3.OperationalError:
            pass
        return conn

    # --- projects ---------------------------------------------------------

    async def create_project(
        self, name: str, goal: str, dataset_csv: str, columns: list[str]
    ) -> dict:
        import uuid as uuid_mod

        def op() -> dict:
            with self._lock:
                conn = self._db()
                try:
                    pid = uuid_mod.uuid4().hex[:12]
                    rows = dataset_csv.strip().splitlines()
                    num_rows = max(0, len(rows) - 1)
                    conn.execute(
                        "INSERT INTO projects"
                        " (id, name, goal, dataset_csv, num_rows, columns, created_at)"
                        " VALUES (?,?,?,?,?,?,?)",
                        (pid, name, goal, dataset_csv, num_rows, json.dumps(columns), now_iso()),
                    )
                    conn.commit()
                    row = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
                    return _dict(row)
                finally:
                    conn.close()

        return await asyncio.to_thread(op)

    async def list_projects(self) -> list[dict]:
        def op() -> list[dict]:
            with self._lock:
                conn = self._db()
                try:
                    rows = conn.execute(
                        "SELECT * FROM projects ORDER BY created_at DESC"
                    ).fetchall()
                    out = []
                    for r in rows:
                        d = _dict(r)
                        d.pop("dataset_csv")  # bulky; fetched per project
                        agg = conn.execute(
                            "SELECT COUNT(*) c, MAX(best_score) b FROM runs WHERE project_id=?",
                            (d["id"],),
                        ).fetchone()
                        d["num_runs"] = agg["c"]
                        d["best_score"] = agg["b"]
                        out.append(d)
                    return out
                finally:
                    conn.close()

        return await asyncio.to_thread(op)

    async def get_project(self, pid: str) -> dict | None:
        def op() -> dict | None:
            with self._lock:
                conn = self._db()
                try:
                    p = _dict(conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone())
                    if not p:
                        return None
                    p["columns"] = json.loads(p["columns"])
                    runs = conn.execute(
                        "SELECT * FROM runs WHERE project_id=? ORDER BY started_at DESC", (pid,)
                    ).fetchall()
                    p["runs"] = []
                    for r in runs:
                        d = _dict(r)
                        d["params"] = json.loads(d["params"])
                        d["events"] = json.loads(d["events"])
                        gens = conn.execute(
                            "SELECT gen, best_score, num_experiments FROM run_generations"
                            " WHERE run_id=? ORDER BY gen",
                            (d["id"],),
                        ).fetchall()
                        d["generations_summary"] = [_dict(g) for g in gens]
                        p["runs"].append(d)
                    return p
                finally:
                    conn.close()

        return await asyncio.to_thread(op)

    async def delete_project(self, pid: str) -> bool:
        def op() -> bool:
            with self._lock:
                conn = self._db()
                try:
                    cur = conn.execute("DELETE FROM projects WHERE id=?", (pid,))
                    conn.commit()
                    return cur.rowcount > 0
                finally:
                    conn.close()

        return await asyncio.to_thread(op)

    # --- runs --------------------------------------------------------------

    async def create_run(self, pid: str, params: dict, continued_from: str | None = None) -> dict:
        import uuid as uuid_mod

        def op() -> dict:
            with self._lock:
                conn = self._db()
                try:
                    rid = uuid_mod.uuid4().hex[:12]
                    conn.execute(
                        "INSERT INTO runs (id, project_id, params, status, current_phase,"
                        " generation, events, started_at, continued_from)"
                        " VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            rid,
                            pid,
                            json.dumps(params),
                            "running",
                            "starting",
                            0,
                            "[]",
                            now_iso(),
                            continued_from,
                        ),
                    )
                    conn.commit()
                    row = conn.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
                    return _dict(row)
                finally:
                    conn.close()

        return await asyncio.to_thread(op)

    async def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        current_phase: str | None = None,
        generation: int | None = None,
        best_score: float | None = None,
        events: list[dict] | None = None,
        error: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        def op() -> None:
            with self._lock:
                conn = self._db()
                try:
                    sets, vals = [], []
                    for k, v in {
                        "status": status,
                        "current_phase": current_phase,
                        "generation": generation,
                        "best_score": best_score,
                        "events": json.dumps(events) if events is not None else None,
                        "error": error,
                        "finished_at": finished_at,
                    }.items():
                        if v is not None:
                            sets.append(f"{k}=?")
                            vals.append(v)
                    if not sets:
                        return
                    vals.append(run_id)
                    conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id=?", vals)
                    conn.commit()
                finally:
                    conn.close()

        await asyncio.to_thread(op)

    async def add_generation(self, run_id: str, gen: dict, experiments: list[dict]) -> int:
        def op() -> int:
            with self._lock:
                conn = self._db()
                try:
                    cur = conn.execute(
                        "INSERT INTO run_generations (run_id, gen, best_score, num_experiments,"
                        " summary, query, sources, hypotheses, created_at)"
                        " VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            run_id,
                            gen.get("gen"),
                            gen.get("best_score"),
                            gen.get("num_experiments", len(experiments)),
                            gen.get("summary"),
                            gen.get("query"),
                            json.dumps(gen.get("sources", [])),
                            json.dumps(gen.get("hypotheses", [])),
                            now_iso(),
                        ),
                    )
                    gid = cur.lastrowid
                    conn.executemany(
                        "INSERT INTO run_experiments (generation_id, name, score, rmse, expression,"
                        " hypothesis, rationale) VALUES (?,?,?,?,?,?,?)",
                        [
                            (
                                gid,
                                e.get("name"),
                                e.get("score"),
                                e.get("rmse"),
                                e.get("expression"),
                                e.get("hypothesis"),
                                e.get("rationale"),
                            )
                            for e in experiments
                        ],
                    )
                    conn.commit()
                    return gid
                finally:
                    conn.close()

        return await asyncio.to_thread(op)

    async def get_run(self, rid: str) -> dict | None:
        def op() -> dict | None:
            with self._lock:
                conn = self._db()
                try:
                    r = conn.execute(
                        "SELECT r.*, p.id AS project_id, p.name AS project_name,"
                        " p.goal AS project_goal, p.columns AS project_columns,"
                        " p.num_rows AS project_num_rows"
                        " FROM runs r JOIN projects p ON p.id = r.project_id WHERE r.id=?",
                        (rid,),
                    ).fetchone()
                    if not r:
                        return None
                    d = _dict(r)
                    d["params"] = json.loads(d["params"])
                    d["events"] = json.loads(d["events"])
                    d["columns"] = json.loads(d.pop("project_columns"))
                    gens = conn.execute(
                        "SELECT * FROM run_generations WHERE run_id=? ORDER BY gen", (rid,)
                    ).fetchall()
                    d["generations"] = []
                    for g in gens:
                        gd = _dict(g)
                        gd["sources"] = json.loads(gd["sources"])
                        gd["hypotheses"] = json.loads(gd["hypotheses"])
                        exps = conn.execute(
                            "SELECT name, score, rmse, expression, hypothesis, rationale"
                            " FROM run_experiments WHERE generation_id=? ORDER BY score DESC",
                            (gd["id"],),
                        ).fetchall()
                        gd["experiments"] = [_dict(e) for e in exps]
                        d["generations"].append(gd)
                    return d
                finally:
                    conn.close()

        return await asyncio.to_thread(op)

    async def mark_interrupted(self) -> int:
        """On startup: runs left 'running' by a restart are marked stopped."""

        def op() -> int:
            with self._lock:
                conn = self._db()
                try:
                    cur = conn.execute(
                        "UPDATE runs SET status='stopped', finished_at=?,"
                        " error='interrupted by server restart' WHERE status='running'",
                        (now_iso(),),
                    )
                    conn.commit()
                    return cur.rowcount
                finally:
                    conn.close()

        return await asyncio.to_thread(op)
