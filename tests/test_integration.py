"""End-to-end run against a real Postgres with Crusoe / Modal / bucket faked.

Runs only when DATABASE_URL points at a disposable database that has been migrated.
"""

import os

import pytest
from httpx import ASGITransport, AsyncClient

from app.schemas import SandboxResult

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION"), reason="set RUN_INTEGRATION=1 with a migrated DATABASE_URL"
)


async def test_dataset_to_run(monkeypatch):
    from app.db import engine
    from app.main import app
    from app.services import inference, sandbox, storage

    bucket: dict[str, list] = {}

    async def put_jsonl(key, rows):
        bucket[key] = rows
        return key

    async def get_jsonl(key):
        return bucket[key]

    async def chat(messages, model=None, **kw):
        return "```python\ndef add(a, b):\n    return a + b\n```", 5

    async def run_python(code, timeout=None):
        ok = "return a + b" in code
        return SandboxResult(
            stdout="", stderr="" if ok else "AssertionError", exit_code=0 if ok else 1
        )

    monkeypatch.setattr(storage, "put_jsonl", put_jsonl)
    monkeypatch.setattr(storage, "get_jsonl", get_jsonl)
    monkeypatch.setattr(inference, "chat", chat)
    monkeypatch.setattr(sandbox, "run_python", run_python)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/datasets",
            json={
                "name": f"it-{os.urandom(4).hex()}",
                "tasks": [
                    {
                        "id": "add",
                        "prompt": "Write add(a, b)",
                        "grader": "code",
                        "test": "assert add(2, 3) == 5",
                    }
                ],
            },
        )
        assert r.status_code == 201, r.text
        dataset_id = r.json()["id"]

        r = await client.post("/runs", json={"dataset_id": dataset_id})
        assert r.status_code == 202, r.text
        run_id = r.json()["id"]

        # BackgroundTasks run after the response inside the ASGI call, so the run is done here.
        run = (await client.get(f"/runs/{run_id}")).json()
        assert run["status"] == "completed", run
        assert run["score"] == 1.0
        assert run["results"][0]["sandbox_exit_code"] == 0
        assert run["trace_s3_key"] in bucket

        assert (await client.get(f"/runs/{run_id}/failures")).json() == []
    await engine.dispose()


async def test_harness_session_loop(monkeypatch):
    """Steps 1-7 end-to-end: CSV in, worker formulae, sandboxed benchmarks, summary,
    research, prompt rewrite, next generation."""
    import json

    from app.db import engine
    from app.main import app
    from app.services import inference, research, sandbox, storage

    bucket: dict[str, bytes] = {}

    async def put_bytes(key, data, content_type="application/octet-stream"):
        bucket[key] = data
        return key

    async def get_bytes(key):
        return bucket[key]

    async def chat(messages, model=None, temperature=0.2, max_tokens=4096, **kw):
        if any("worker agent" in m.get("content", "") for m in messages if isinstance(m, dict)):
            return json.dumps(
                {
                    "name": "sum",
                    "expression": "y = a + b",
                    "description": "add features",
                    "code": "def predict(row):\n    return row['a'] + row['b']",
                    "rationale": "y looked additive",
                }
            ), 5
        return "SUMMARY: agents assumed additivity; tried interactions; best was linear.", 5

    async def run_python(code, timeout=None):
        # run the actual benchmark wrapper inline (it's pure python)
        import io
        import math
        import statistics

        ns: dict = {"json": json, "math": math, "statistics": statistics}
        try:
            from contextlib import redirect_stdout

            out = io.StringIO()
            with redirect_stdout(out):
                exec(code, ns)  # noqa: S102 - executing our own benchmarker wrapper
            return SandboxResult(stdout=out.getvalue(), stderr="", exit_code=0)
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)

    async def fetch_sources(query, n):
        return [
            {"title": f"src: {query}", "url": "https://example.com/x", "snippet": "use ridge"}
        ] * n

    monkeypatch.setattr(storage, "put_bytes", put_bytes)
    monkeypatch.setattr(storage, "get_bytes", get_bytes)
    monkeypatch.setattr(inference, "chat", chat)
    monkeypatch.setattr(sandbox, "run_python", run_python)
    monkeypatch.setattr(research, "fetch_sources", fetch_sources)

    csv_text = "a,b,y\n1,2,3\n2,3,5\n3,4,7\n"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/sessions",
            json={
                "name": f"harness-{os.urandom(4).hex()}",
                "goal": "predict y from a and b",
                "validation_csv": csv_text,
                "seed_formulae": [
                    {
                        "name": "mean",
                        "expression": "y = 5",
                        "code": "def predict(row):\n    return 5.0",
                    }
                ],
            },
        )
        assert r.status_code == 202, r.text
        session_id = r.json()["id"]

        detail = (await client.get(f"/sessions/{session_id}")).json()
        assert detail["status"] == "completed", detail
        assert detail["num_validation_rows"] == 3
        assert len(detail["sources"]) > 0  # step 5 stored sources
        assert detail["system_prompt"], "step 6 rewrote the prompt"
        assert detail["best_score"] and detail["best_score"] >= 0.9  # y = a + b is exact
        assert any(f["source"] == "seed" for f in detail["formulae"])  # step 1 seeds
        assert any(f["source"] == "agent" for f in detail["formulae"])  # step 2 workers

        # step 6 tool: RAG read over findings
        findings = (
            await client.get(f"/sessions/{session_id}/findings", params={"q": "ridge"})
        ).json()
        assert isinstance(findings, list)

        # step 7: iterate another generation with the rewritten prompt
        r = await client.post(f"/sessions/{session_id}/iterate", json={"generations": 1})
        assert r.status_code == 202, r.text
        detail = (await client.get(f"/sessions/{session_id}")).json()
        assert detail["generation"] >= 1
    await engine.dispose()
