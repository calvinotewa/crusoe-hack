"""Curator gate and curated memory, plus the HarnessRunner curate phase.

Pure logic plus monkeypatched `app.services.shared_brain` calls: no Crusoe, Modal or search.
"""

import math

import pytest

from app.services import curation, harness_runner, shared_brain
from app.services.curation import TOKEN_CAP, Memory, MemoryFull


def exp(name, rmse, hypothesis="h"):
    code = f"rmse={rmse}"
    return {"name": name, "rmse": rmse, "code": code, "hypothesis": hypothesis, "expression": name}


async def const_verify(losses):
    async def verify(code):
        return losses

    return verify


def test_memory_renders_in_fixed_order_and_compact_drops_try_next_first():
    m = Memory()
    m.set_best("best claim", gen=0)
    for i in range(10_000):
        try:
            m.upsert("failed", f"f{i} " + "x" * 150, gen=0)
        except MemoryFull:
            break
    for i in range(10_000):
        try:
            m.upsert("try_next", f"t{i} " + "y" * 150, gen=1)
        except MemoryFull:
            break
    kinds = [line.split("]")[0].split("[")[1].split(" ")[0] for line in m.render().splitlines()]
    assert kinds[0] == "best"
    failed_before = sum(e.kind == "failed" for e in m.entries.values())
    m.compact()
    assert m.tokens() <= TOKEN_CAP
    assert sum(e.kind == "failed" for e in m.entries.values()) == failed_before
    assert any(e.kind == "best" for e in m.entries.values())


def test_stale_entries_after_two_best_versions():
    m = Memory()
    m.upsert("works", "old fact", gen=0)
    m.set_best("b1", gen=1)
    assert m.stale() == []
    m.set_best("b2", gen=2)
    assert [e.claim for e in m.stale()] == ["old fact"]


@pytest.mark.asyncio
async def test_curate_promotes_only_when_resampled_loss_beats_incumbent():
    m = Memory()
    scored = [exp("good", 0.5), exp("bad", 0.9)]
    promoted = await curation.curate(m, scored, 1, 1.0, await const_verify([0.5, 0.5, 0.5]))
    assert promoted is not None and promoted.experiment["name"] == "good"
    assert any(e.kind == "best" for e in m.entries.values())

    m2 = Memory()
    promoted2 = await curation.curate(m2, scored, 1, 1.0, await const_verify([0.995] * 3))
    assert promoted2 is None
    assert any(e.kind == "failed" and "did not replicate" in e.claim for e in m2.entries.values())


@pytest.mark.asyncio
async def test_crashed_top_is_never_promoted_and_no_incumbent_takes_first_verified():
    m = Memory()
    scored = [exp("crash", None), exp("ok", 0.7)]
    promoted = await curation.curate(m, scored, 1, None, await const_verify([0.7, 0.7, 0.7]))
    assert promoted is not None and promoted.experiment["name"] == "ok"

    m2 = Memory()
    assert (
        await curation.curate(m2, [exp("crash", None)], 1, 1.0, await const_verify([math.inf] * 3))
    ) is None


@pytest.mark.asyncio
async def test_finite_benchmark_whose_resamples_all_crash_is_not_promoted():
    m = Memory()
    promoted = await curation.curate(
        m, [exp("flaky", 0.4)], 1, None, await const_verify([math.inf] * 3)
    )
    assert promoted is None
    assert not any(e.kind == "best" for e in m.entries.values())


@pytest.mark.asyncio
async def test_bottom_is_recorded_as_failed_and_middle_as_try_next():
    m = Memory()
    scored = [exp("a", 0.3), exp("b", 0.5, "mid idea"), exp("c", 0.9, "bottom idea")]
    await curation.curate(m, scored, 1, 1.0, await const_verify([0.3] * 3))
    claims = {e.kind: [] for e in m.entries.values()}
    for e in m.entries.values():
        claims.setdefault(e.kind, []).append(e.claim)
    assert any("mid idea" in c for c in claims.get("try_next", []))
    assert any("bottom idea" in c for c in claims.get("failed", []))


class FakeStore:
    def __init__(self):
        self.runs = []
        self.generations = []

    async def update_run(self, run_id, **kw):
        self.runs.append(kw)

    async def add_generation(self, run_id, record, experiments):
        self.generations.append(record)


@pytest.mark.asyncio
async def test_harness_runner_moves_best_only_after_replicated_claim(monkeypatch):
    """Gen 1 promotes its verified top. Gen 2 improves within the noise floor, so no promotion."""
    rounds = iter([0.5, 0.4995])

    async def fake_agent(n, goal, columns, hyp, mem, best):
        return {
            "name": f"a{n}",
            "code": f"rmse={next(rounds)}",
            "expression": "x",
            "hypothesis": hyp,
        }

    async def fake_run(code, rows, timeout=20.0):
        return {"rmse": float(code.split("=")[1]), "score": 1 / (1 + float(code.split("=")[1]))}

    async def fake_verify(code, rows, resamples=3, seed=0):
        return [float(code.split("=")[1])] * 3

    async def fake_gap(summary):
        return "q"

    async def fake_search(query, n=5):
        return []

    async def fake_hyps(goal, summary, learnings, n, concurrency=4):
        return [f"hyp {i}" for i in range(n)]

    monkeypatch.setattr(shared_brain, "agent_experiment", fake_agent)
    monkeypatch.setattr(shared_brain, "run_experiment", fake_run)
    monkeypatch.setattr(shared_brain, "verify_experiment", fake_verify)
    monkeypatch.setattr(shared_brain, "search_gap_query", fake_gap)
    monkeypatch.setattr(shared_brain, "web_search", fake_search)
    monkeypatch.setattr(shared_brain, "generate_hypotheses", fake_hyps)

    rows = [{"x": 1.0, "y": 1.0}]
    params = {
        "agents": 1,
        "generations": 2,
        "hypotheses": 1,
        "top_k": 1,
        "concurrency": 1,
        "sources": 0,
    }
    store = FakeStore()
    runner = harness_runner.HarnessRunner(params, "goal", rows, store, "run1")
    state = await runner.run()

    assert state["status"] == "completed"
    assert state["generations"][0]["best_score"] == pytest.approx(1 / 1.5)
    assert state["best_score"] == pytest.approx(1 / 1.5)
    assert any(e.kind == "best" for e in runner.memory.entries.values())
    assert "best" in state["generations"][1]["summary"]
