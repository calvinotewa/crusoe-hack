"""Unit tests for the orchestration harness (steps 1-7).

Tests never call Crusoe, Modal or the bucket: inference, sandboxes and search are
monkeypatched at the `app.services` module boundary, per AGENTS.md. Postgres-backed
behaviour (sessions, benchmarks) is covered by tests/test_integration.py.
"""

import json

from app.services import benchmarker, research, subagent
from app.services.formulae import FORMULA_SCHEMA


def fake_formula_json(name="f"):
    return {
        "name": name,
        "expression": "y ~ a",
        "description": "test",
        "code": "def predict(row):\n    return 42.0",
        "rationale": "guessed a constant",
    }


# --- step 2: runSubAgent ---


async def test_run_sub_agent_passes_system_prompt_and_tools(monkeypatch):
    calls = []

    async def fake_chat(messages, model=None, temperature=0.2, max_tokens=4096, **kw):
        calls.append(messages)
        return json.dumps(fake_formula_json()), 5

    monkeypatch.setattr(subagent.inference, "chat", fake_chat)
    out = await subagent.run_json(
        "make a formula", shape="json", system_prompt="sys", tools=["storeFormulae"]
    )
    assert out["name"] == "f"
    assert calls[0][0]["role"] == "system" and calls[0][0]["content"] == "sys"
    assert "storeFormulae" in calls[0][1]["content"]


async def test_run_json_repairs_once(monkeypatch):
    responses = ["definitely not json", json.dumps({"name": "fixed"})]

    async def fake_chat(messages, **kw):
        return responses.pop(0), 1

    monkeypatch.setattr(subagent.inference, "chat", fake_chat)
    assert (await subagent.run_json("x", shape="json"))["name"] == "fixed"


async def test_run_json_gives_up_after_repair(monkeypatch):
    async def fake_chat(messages, **kw):
        return "still not json", 1

    monkeypatch.setattr(subagent.inference, "chat", fake_chat)
    assert await subagent.run_json("x", shape="json") == {}


def test_worker_prompt_contract():
    p = subagent.worker_prompt("predict y", ["a", "b"], 3)
    assert FORMULA_SCHEMA.splitlines()[0] in p
    assert "predict y" in p and "'a'" in p


def test_worker_prompt_carries_hypothesis_and_memory():
    p = subagent.worker_prompt(
        "goal", ["a"], 1, hypothesis="try interactions", memory="linear failed"
    )
    assert "try interactions" in p and "linear failed" in p


# --- Shared Brain loop helpers (services/shared_brain.py) ---


def test_parse_numbered_list():
    from app.services import shared_brain

    text = "1. Use log transforms\n2) Try interaction terms\n\nnot a list line\n"
    assert shared_brain.parse_numbered_list(text) == ["Use log transforms", "Try interaction terms"]


async def test_generate_hypotheses_batches(monkeypatch):
    from app.services import shared_brain

    calls = []

    async def fake_curate(prompt):
        calls.append(prompt)
        return "1. alpha idea\n2. beta idea"

    monkeypatch.setattr(shared_brain, "_curate", fake_curate)
    out = await shared_brain.generate_hypotheses("goal", "summary", "research", 4, batch=2)
    assert out[0] == "alpha idea" and out[1] == "beta idea"
    assert len(out) == 4  # exactly n, padded if the curator repeats itself
    assert len(set(out)) == 4  # unique


async def test_generate_hypotheses_exact_count(monkeypatch):
    """The count is FIXED: exactly n, every call, de-duplicated."""
    from app.services import shared_brain

    counter = {"i": 0}

    async def fake_curate(prompt):
        counter["i"] += 1
        return f"1. hypothesis {counter['i']}a\n2. hypothesis {counter['i']}b"

    monkeypatch.setattr(shared_brain, "_curate", fake_curate)
    for n in (1, 5, 20):
        out = await shared_brain.generate_hypotheses("goal", "sum", "res", n, batch=2)
        assert len(out) == n, f"n={n}: {len(out)}"
        assert len(set(out)) == n


async def test_web_search_fallback_is_safe(monkeypatch):
    """Failing you.com AND failing DDG -> empty list, not an exception."""
    from app.services import shared_brain

    async def failing_youcom(api_key, base, query, n):
        raise RuntimeError("you.com down")

    async def failing_ddg(query, n):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(shared_brain, "_youcom_search", failing_youcom)
    monkeypatch.setattr(shared_brain, "_ddg_search", failing_ddg)
    monkeypatch.delenv("YOU_API_KEY", raising=False)
    assert await shared_brain.web_search("anything") == []


async def test_web_search_uses_youcom_when_configured(monkeypatch):
    from app.config import Settings
    from app.services import shared_brain

    calls = {}

    async def fake_youcom(api_key, base, query, n):
        calls.update(api_key=api_key, base=base, query=query, n=n)
        return [{"title": f"t {query}", "url": "u", "snippet": "s"}] * n

    monkeypatch.setattr(
        shared_brain,
        "get_settings",
        lambda: Settings(you_api_key="k", you_api_base="https://x.io"),
    )
    monkeypatch.setattr(shared_brain, "_youcom_search", fake_youcom)
    out = await shared_brain.web_search("q1", 7)
    assert len(out) == 7 and out[0]["title"] == "t q1"
    assert calls["n"] == 7 and calls["base"] == "https://x.io"


async def test_agent_experiment_parses_json(monkeypatch):
    from app.services import shared_brain

    async def fake_chat(messages, model=None, temperature=0.7, max_tokens=4096, **kw):
        return '{"name": "h", "expression": "y~a", "code": "def predict(row):\\n    return 1"}', 3

    import app.services.inference as inference_mod

    monkeypatch.setattr(inference_mod, "chat", fake_chat)
    out = await shared_brain.agent_experiment(1, "goal", ["a"], "hyp", "mem", 0.5)
    assert out["name"] == "h"


# --- step 3: the benchmark program ---


def _run_program(code: str, rows: list[dict]) -> dict:
    """Exec a generated benchmark program and parse its JSON output, like the sandbox would."""
    import contextlib
    import io

    program = benchmarker.build_program(code, rows)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(program, {})  # noqa: S102 - executing the harness's own benchmark wrapper
    return json.loads(out.getvalue().strip().splitlines()[-1])


def test_build_program_exact_formula_scores_perfectly():
    rows = [
        {"a": 1.0, "b": 2.0, "y": 3.0},
        {"a": 2.0, "b": 3.0, "y": 5.0},
    ]
    metrics = _run_program("def predict(row):\n    return row['a'] + row['b']", rows)
    assert metrics["n"] == 2 and metrics["skipped"] == 0
    assert abs(metrics["rmse"]) < 1e-9 and metrics["score"] > 0.99


def test_build_program_skips_crashing_rows():
    rows = [{"a": 1.0, "y": 2.0}, {"a": 1.0, "y": 2.0}, {"a": 1.0, "y": 2.0}]
    metrics = _run_program("def predict(row):\n    return 1 / 0", rows)
    assert metrics["n"] == 0 and metrics["skipped"] == 3 and metrics["score"] == 0.0


def test_coerce_row():
    row = benchmarker._coerce({"a": " 1.5 ", "b": "", "c": "x", "y": "3"})
    assert row == {"a": 1.5, "b": None, "c": "x", "y": 3}


# --- step 5: research source fetching ---


async def test_fetch_sources_stub_when_search_empty(monkeypatch):
    """Keyless search that yields nothing -> stub sources, so the loop stays runnable."""
    from app.config import Settings
    from app.services import shared_brain

    async def empty_search(query, n):
        return []

    monkeypatch.setattr(research, "get_settings", lambda: Settings(you_api_key=""))
    monkeypatch.setattr(shared_brain, "web_search", empty_search)
    sources = await research.fetch_sources("kernel tricks", 3)
    assert len(sources) == 3 and "kernel tricks" in sources[0]["title"]


# --- step 1: input contract ---


def test_parse_validation_csv():
    from fastapi import HTTPException
    from fastapi.testclient import TestClient

    from app.api.routes import sessions as routes_sessions
    from app.main import app

    rows = routes_sessions._parse_validation_csv("a,b,y\n1,2,3\n2,3,5\n3,4,7\n")
    assert len(rows) == 3 and "y" in rows[0]

    try:
        routes_sessions._parse_validation_csv("a,b\n1,2\n")
        raise AssertionError("missing y column should 422")
    except HTTPException as e:
        assert e.status_code == 422

    # app still boots with the new router
    assert TestClient(app).get("/health").json() == {"status": "ok"}


# --- repair_code: agents sometimes return double-escaped JSON string code ---


def test_repair_code_unescapes_double_escaped_newlines():
    from app.services import shared_brain

    broken = "def predict(row):\\n    return 0.5"  # literal backslash-n
    fixed = shared_brain.repair_code(broken)
    ns: dict = {}
    exec(fixed, ns)  # noqa: S102 - the point of the test
    assert ns["predict"]({"a": 1}) == 0.5


def test_repair_code_keeps_valid_code_untouched():
    from app.services import shared_brain

    good = "def predict(row):\n    return row['a'] + 1\n"
    assert shared_brain.repair_code(good) == good
