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


async def test_continue_lineage(tmp_path: Path):
    """A continued run is a new run row in the same project, with lineage."""
    store = Store(tmp_path / "t3.db")
    p = await store.create_project("p", "g", CSV, ["a", "b", "y"])
    r1 = await store.create_run(p["id"], {"agents": 3})
    r2 = await store.create_run(p["id"], {"agents": 5}, continued_from=r1["id"])
    assert r2["continued_from"] == r1["id"]
    got = await store.get_run(r2["id"])
    assert got["continued_from"] == r1["id"] and got["project_id"] == p["id"]


async def test_runner_inherits_state(tmp_path: Path, monkeypatch):
    """Continued runs start from the previous run's best score + hypotheses."""
    from app.services import shared_brain
    from app.services.dashboard_store import Store as S
    from app.services.harness_runner import HarnessRunner

    seen = {}

    async def fake_agent(n, goal, cols, hyp, mem, best):
        seen[n] = (hyp, mem, best)
        return {"name": f"a{n}", "expression": "y", "code": "def predict(row):\n    return 0.5"}

    async def fake_run(code, rows, timeout=20.0):
        return {
            "score": 0.4,
            "rmse": 1.5,
            "mae": 1.0,
            "r2": 0.0,
            "exit_code": 0,
            "stderr_tail": None,
        }

    async def fake_sum(goal, top, k):
        return "s"

    async def fake_gap(s):
        return "q"

    async def fake_search(q, n=5):
        return [{"title": "t", "url": "u", "snippet": "s"}]

    async def fake_hyps(goal, s, r, n, batch=10, concurrency=4, rounds_cap=10):
        return [f"h{i}" for i in range(n)]

    monkeypatch.setattr(shared_brain, "agent_experiment", fake_agent)
    monkeypatch.setattr(shared_brain, "run_experiment", fake_run)
    monkeypatch.setattr(shared_brain, "summarize_top", fake_sum)
    monkeypatch.setattr(shared_brain, "search_gap_query", fake_gap)
    monkeypatch.setattr(shared_brain, "web_search", fake_search)
    monkeypatch.setattr(shared_brain, "generate_hypotheses", fake_hyps)

    store = S(tmp_path / "t4.db")
    p = await store.create_project("p", "g", CSV, ["a", "b", "y"])
    params = {"agents": 2, "hypotheses": 3, "generations": 1, "top_k": 2, "sources": 2, "concurrency": 2}
    run = await store.create_run(p["id"], params)
    runner = HarnessRunner(
        params,
        "goal",
        [{"a": 1.0, "y": 2.0}],
        store,
        run["id"],
        initial_hypotheses=["inh1", "inh2", "inh3"],
        initial_memory="inherited memory",
        initial_best_score=0.71,
    )
    await runner.run()
    hyp, mem, best = seen[1]
    assert hyp == "inh1" and "inherited memory" in mem and best == 0.71
    got = await store.get_run(run["id"])
    assert got["best_score"] == 0.71  # inherited best is kept (new gens scored lower)
    assert got["generations"][0]["hypotheses"] == ["h0", "h1", "h2"]  # fixed count kept
