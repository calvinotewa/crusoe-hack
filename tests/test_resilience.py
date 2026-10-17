"""Resilience tests: LLM errors must degrade, retry and fall back — never
kill a harness run. All offline: the inference seam (`_create_once`) and the
service functions are monkeypatched; nothing touches Crusoe or the network.
"""

from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, NotFoundError

from app.config import Settings
from app.services import inference, shared_brain
from app.services.harness_runner import HarnessRunner
from app.services.inference import RateLimiter

FAST = Settings(crusoe_chat_backoff_seconds=0.01, crusoe_rate_limit_rpm=0)


def _resp(status=404):
    return httpx.Response(status, request=httpx.Request("POST", "https://api.test/v1"))


def _msg(content, finish="stop"):
    return SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)


# --- inference.chat: transient retry, model fallback, budget retry -----------


async def test_chat_retries_transient_errors(monkeypatch):
    monkeypatch.setattr(inference, "get_settings", lambda: FAST)
    monkeypatch.setattr(inference, "_limiter", lambda: RateLimiter(0))
    calls = []

    async def flaky_create(model, messages, temperature, max_tokens, **kw):
        calls.append(model)
        if len(calls) < 3:
            raise APIConnectionError(request=httpx.Request("POST", "https://api.test"))
        return SimpleNamespace(choices=[_msg("ok")])

    monkeypatch.setattr(inference, "_create_once", flaky_create)
    content, _ = await inference.chat([{"role": "user", "content": "hi"}])
    assert content == "ok" and len(calls) == 3


async def test_chat_falls_back_to_next_model(monkeypatch):
    monkeypatch.setattr(inference, "get_settings", lambda: FAST)
    monkeypatch.setattr(inference, "_limiter", lambda: RateLimiter(0))
    seen = []

    async def picky_create(model, messages, temperature, max_tokens, **kw):
        seen.append(model)
        if "primary" in model:
            raise NotFoundError("no such model", response=_resp(), body={})
        return SimpleNamespace(choices=[_msg(f"served by {model}")])

    monkeypatch.setattr(
        inference,
        "_fallback_chain",
        lambda model: [model or "primary/1", "fallback/2"],
    )
    monkeypatch.setattr(inference, "_create_once", picky_create)
    content, _ = await inference.chat([{"role": "user", "content": "hi"}], model="primary/1")
    assert "served by fallback/2" in content
    assert "primary/1" in seen and "fallback/2" in seen


async def test_chat_doubles_budget_on_empty_length_finish(monkeypatch):
    monkeypatch.setattr(inference, "get_settings", lambda: FAST)
    monkeypatch.setattr(inference, "_limiter", lambda: RateLimiter(0))
    budgets = []

    async def stingy_create(model, messages, temperature, max_tokens, **kw):
        budgets.append(max_tokens)
        if len(budgets) == 1:
            return SimpleNamespace(choices=[_msg("", finish="length")])
        return SimpleNamespace(choices=[_msg("finally answered")])

    monkeypatch.setattr(inference, "_create_once", stingy_create)
    content, _ = await inference.chat([{"role": "user", "content": "hi"}], max_tokens=1024)
    assert content == "finally answered"
    assert budgets[1] == 2048  # doubled once


async def test_chat_retries_truncated_nonempty_length(monkeypatch):
    """Truncated-but-nonempty output (reasoning model runs out mid-JSON) retries
    with a doubled budget; the longer answer wins."""
    monkeypatch.setattr(inference, "get_settings", lambda: FAST)
    monkeypatch.setattr(inference, "_limiter", lambda: RateLimiter(0))
    budgets = []

    async def truncating(model, messages, temperature, max_tokens, **kw):
        budgets.append(max_tokens)
        if len(budgets) == 1:
            return SimpleNamespace(choices=[_msg('{"name": "half', finish="length")])
        return SimpleNamespace(choices=[_msg('{"name": "complete model"}')])

    monkeypatch.setattr(inference, "_create_once", truncating)
    content, _ = await inference.chat([{"role": "user", "content": "hi"}], max_tokens=1024)
    assert "complete model" in content
    assert budgets[1] == 2048


async def test_chat_returns_longest_partial_if_always_truncated(monkeypatch):
    monkeypatch.setattr(inference, "get_settings", lambda: FAST)
    monkeypatch.setattr(inference, "_limiter", lambda: RateLimiter(0))

    async def always_truncated(model, messages, temperature, max_tokens, **kw):
        text = '{"name": "' + "x" * 100 + " truncated"
        return SimpleNamespace(choices=[_msg(text, finish="length")])

    monkeypatch.setattr(inference, "_create_once", always_truncated)
    content, _ = await inference.chat([{"role": "user", "content": "hi"}])
    assert content.startswith('{"name"')  # usable partial, not an exception


async def test_chat_raises_only_after_every_fallback(monkeypatch):
    monkeypatch.setattr(inference, "get_settings", lambda: FAST)
    monkeypatch.setattr(inference, "_limiter", lambda: RateLimiter(0))
    seen = []

    async def always_dead(model, messages, temperature, max_tokens, **kw):
        seen.append(model)
        raise APIConnectionError(request=httpx.Request("POST", "https://api.test"))

    monkeypatch.setattr(inference, "_fallback_chain", lambda model: ["m/one", "m/two", "m/three"])
    monkeypatch.setattr(inference, "_create_once", always_dead)
    with pytest.raises(APIConnectionError):
        await inference.chat([{"role": "user", "content": "hi"}])
    # every model got its full retry budget, in fallback order
    assert seen[0] == "m/one" and seen[-1] == "m/three"
    assert seen.count("m/one") == 3 and seen.count("m/three") == 3


# --- agent failures become stubs, not exceptions -----------------------------


def test_parse_agent_json_recovers_truncated():
    """A completion that runs out of tokens mid-string still parses via
    brace-closing — the common reasoning-model truncation failure."""
    truncated = (
        '{"name": "CubicVelocity", "expression": "y = b0 + b1*log(f)", '
        '"description": "cubic in alpha", "code": "def fit(rows):\\n    return [1.0]'
    )  # no closing quotes/braces
    parsed = shared_brain._parse_agent_json("\nthinking...\n" + truncated)
    assert parsed is not None and parsed["name"] == "CubicVelocity"
    assert parsed["code"].startswith("def fit")


def test_parse_agent_json_returns_none_on_garbage():
    assert shared_brain._parse_agent_json("no json at all here") is None
    assert shared_brain._parse_agent_json("") is None


async def test_agent_experiment_repairs_unparseable_json(monkeypatch):
    """Unparseable/incomplete JSON gets one repair round before the stub."""
    calls = []

    async def flaky_chat(messages, model=None, temperature=0.2, max_tokens=4096, **kw):
        calls.append(len(messages))
        if len(calls) == 1:
            return "Sure! Here's the experiment you asked for — great question.", 1  # no JSON
        # the repair call: assistant turn carries the broken output
        assert len(messages) == 3 and messages[1]["role"] == "assistant"
        return (
            '{"name": "Fixed", "code": "def fit(rows):\\n    return [1]", '
            '"expression": "y", "description": "d", "rationale": "r"}',
            1,
        )

    monkeypatch.setattr(shared_brain.inference, "chat", flaky_chat)
    exp = await shared_brain.agent_experiment(1, "goal", ["a"], "hyp", "", None)
    assert exp["name"] == "Fixed" and exp["code"]
    assert len(calls) == 2  # original + repair


async def test_agent_experiment_repairs_codeless_recovery(monkeypatch):
    """Truncation that brace-closes into a dict WITHOUT `code` still repairs."""
    calls = []

    async def once_then_dead(messages, model=None, temperature=0.2, max_tokens=4096, **kw):
        calls.append(1)
        if len(calls) == 1:
            return '{"name": "half', 1  # brace-closes fine, but no code
        raise AssertionError("repair round missing")

    monkeypatch.setattr(shared_brain.inference, "chat", once_then_dead)
    exp = await shared_brain.agent_experiment(1, "goal", ["a"], "hyp", "", None)
    assert len(calls) == 2  # original + repair (the repair raising -> stub)
    assert "unparsed" in exp["name"]


async def test_agent_experiment_returns_error_stub_on_llm_failure(monkeypatch):
    async def dead_chat(*a, **kw):
        raise APIConnectionError(request=httpx.Request("POST", "https://api.test"))

    monkeypatch.setattr(shared_brain.inference, "chat", dead_chat)
    exp = await shared_brain.agent_experiment(1, "goal", ["a"], "hyp", "", None)
    assert exp["name"] == "agent-1-error" and exp["code"] is None


async def test_generate_hypotheses_survives_batch_failures(monkeypatch):
    calls = {"n": 0}

    async def flaky_curate(prompt):
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            raise APIConnectionError(request=httpx.Request("POST", "https://api.test"))
        return "1. reliable idea\n2. another idea"

    monkeypatch.setattr(shared_brain, "_curate", flaky_curate)
    out = await shared_brain.generate_hypotheses("goal", "sum", "res", 4, batch=2, concurrency=1)
    assert len(out) == 4  # exact fixed count despite the dead batches
    assert calls["n"] >= 4  # failing batches were retried, not fatal


async def test_search_gap_query_degrades_on_empty_curator(monkeypatch):
    async def empty_curate(prompt):
        return "   \n"

    monkeypatch.setattr(shared_brain, "_curate", empty_curate)
    q = await shared_brain.search_gap_query("alpha intake scaled pressure summary")
    assert q and "alpha" in q  # deterministic fallback, no IndexError


# --- the run itself: phases degrade, the run survives -------------------------


async def test_run_degrades_gracefully_when_curator_dies(tmp_path, monkeypatch):
    from app.services import curation
    from app.services.dashboard_store import Store

    async def fake_agent(n, goal, columns, hyp, mem, best, stats=""):
        return {
            "name": f"a{n}",
            "code": "def fit(rows):\n    return [1.0]\n\ndef predict(row, params):\n    return 1.0",
            "expression": "y = 1",
            "rationale": "r",
        }

    async def fake_run(code, rows, timeout=60.0, train=()):
        return {
            "score": 0.5,
            "rmse": 1.0,
            "mae": 0.8,
            "r2": 0.0,
            "exit_code": 0,
            "stderr_tail": None,
            "n": len(rows),
            "skipped": 0,
        }

    async def fake_verify(code, rows):
        return [1.0, 1.0]

    async def _fake_curate(memory, scored, gen, incumbent_loss=None, verify=None):
        curation.add(memory, "works", "experiment one worked", gen)
        return None  # no promotion

    async def dead_gap_query(summary):
        raise RuntimeError("curator down")

    async def dead_hypotheses(*a, **kw):
        raise RuntimeError("curator down")

    monkeypatch.setattr(shared_brain, "agent_experiment", fake_agent)
    monkeypatch.setattr(shared_brain, "run_experiment", fake_run)
    monkeypatch.setattr(shared_brain, "verify_experiment", fake_verify)
    monkeypatch.setattr(shared_brain, "search_gap_query", dead_gap_query)
    monkeypatch.setattr(shared_brain, "generate_hypotheses", dead_hypotheses)
    monkeypatch.setattr(curation, "curate", _fake_curate)

    store = Store(tmp_path / "res.db")
    p = await store.create_project("p", "goal", "a,y\n1,2\n", ["a", "y"])
    params = {
        "agents": 2,
        "hypotheses": 3,
        "generations": 2,
        "top_k": 2,
        "sources": 2,
        "concurrency": 2,
    }
    run = await store.create_run(p["id"], params)
    runner = HarnessRunner(
        params,
        "goal",
        [{"a": 1.0, "y": 2.0}, {"a": 2.0, "y": 3.0}],
        store,
        run["id"],
        initial_hypotheses=["h1", "h2", "h3"],
    )
    state = await runner.run()

    assert state["status"] == "completed", (state.get("error"), state)
    # the curator gate never promoted (fake), so best stays None — that's correct;
    # the point here is the run *completed* with every generation fully recorded.
    assert len(state["generations"]) == 2
    assert all(g["num_experiments"] == 2 for g in state["generations"])
    msgs = " ".join(e["msg"] for e in state["events"])
    assert "DEGRADED" in msgs  # degraded phases are visible, not silent
    for g in state["generations"]:
        assert len(g["hypotheses"]) == 3  # exact fixed count even while degraded
