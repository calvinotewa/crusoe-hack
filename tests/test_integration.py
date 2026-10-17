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
