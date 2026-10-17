"""Unit tests for the dashboard store (SQLite, logical separation by project).

Uses a throwaway DB file; no providers involved.
"""

from pathlib import Path

from app.services.dashboard_store import Store

CSV = "a,b,y\n1,2,3\n2,3,5\n"


async def test_project_run_generation_experiments_hierarchy(tmp_path: Path):
    store = Store(tmp_path / "t.db")
    p = await store.create_project("churn", "predict churn", CSV, ["a", "b", "y"])
    assert p["name"] == "churn" and p["num_rows"] == 2

    # a second project: results must not leak across projects
    p2 = await store.create_project("other", "other goal", "x,y\n1,2\n", ["x", "y"])

    run = await store.create_run(p["id"], {"agents": 2, "generations": 1})
    assert run["project_id"] == p["id"]
    gid = await store.add_generation(
        run["id"],
        {"gen": 1, "best_score": 0.5, "num_experiments": 1, "sources": [], "hypotheses": ["h1"]},
        [{"name": "e1", "score": 0.5, "rmse": 1.0}],
    )
    assert gid

    await store.update_run(run["id"], status="completed", best_score=0.5, events=[{"msg": "m"}])

    got = await store.get_run(run["id"])
    assert got["status"] == "completed"
    assert got["best_score"] == 0.5
    assert got["project_id"] == p["id"]
    assert got["columns"] == ["a", "b", "y"]
    assert len(got["generations"]) == 1
    assert got["generations"][0]["hypotheses"] == ["h1"]
    assert got["generations"][0]["experiments"][0]["name"] == "e1"

    # listing is per project with aggregates
    projects = await store.list_projects()
    by_id = {x["id"]: x for x in projects}
    assert by_id[p["id"]]["num_runs"] == 1
    assert by_id[p["id"]]["best_score"] == 0.5
    assert by_id[p2["id"]]["num_runs"] == 0
    assert "dataset_csv" not in by_id[p["id"]]  # bulky field not in list view

    # deleting a project cascades to its runs/generations/experiments only
    assert await store.delete_project(p["id"]) is True
    assert await store.get_run(run["id"]) is None
    assert await store.get_project(p2["id"]) is not None


async def test_mark_interrupted(tmp_path: Path):
    store = Store(tmp_path / "t2.db")
    p = await store.create_project("p", "g", CSV, ["a", "b", "y"])
    r = await store.create_run(p["id"], {})
    n = await store.mark_interrupted()
    assert n == 1
    got = await store.get_run(r["id"])
    assert got["status"] == "stopped" and "interrupted" in (got["error"] or "")
    assert await store.mark_interrupted() == 0  # idempotent
