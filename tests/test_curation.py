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


def install_fakes(monkeypatch, proposals, baseline_rmse=1.0, linear_rmse=2.0):
    """Patch shared_brain so the runner sees a naive baseline and scripted candidate losses."""
    rounds = iter(proposals)

    async def fake_agent(n, goal, columns, hyp, mem, best, stats=""):
        return {
            "name": f"a{n}",
            "code": f"rmse={next(rounds)}",
            "expression": "x",
            "hypothesis": hyp,
        }

    async def fake_run(code, rows, timeout=20.0, train=()):
        if code == shared_brain.NAIVE_CODE:
            return {"rmse": baseline_rmse, "score": 1 / (1 + baseline_rmse)}
        if "COLUMNS" in code:
            return {"rmse": linear_rmse, "score": 1 / (1 + linear_rmse)}
        rmse = float(code.split("=")[1])
        return {"rmse": rmse, "score": 1 / (1 + rmse)}

    async def fake_verify(code, rows, train=(), resamples=3, seed=0):
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


ROWS = [{"x": float(i), "y": float(i) * 2} for i in range(60)]
PARAMS = {
    "agents": 1,
    "generations": 2,
    "hypotheses": 1,
    "top_k": 1,
    "concurrency": 1,
    "sources": 0,
}


@pytest.mark.asyncio
async def test_harness_runner_moves_best_only_after_replicated_claim(monkeypatch):
    """Gen 1 beats the naive baseline and is promoted. Gen 2 improves within the noise floor."""
    install_fakes(monkeypatch, [0.5, 0.4995])
    runner = harness_runner.HarnessRunner(PARAMS, "goal", ROWS, FakeStore(), "run1")
    state = await runner.run()

    assert state["status"] == "completed"
    assert state["generations"][0]["best_score"] == pytest.approx(1 / 1.5)
    assert state["best_score"] == pytest.approx(1 / 1.5)
    assert any(e.kind == "best" for e in runner.memory.entries.values())
    assert "best" in state["generations"][1]["summary"]
    assert state["test"]["beats_baseline"] is True


@pytest.mark.asyncio
async def test_naive_baseline_is_the_incumbent_so_worse_formulas_never_become_best(monkeypatch):
    install_fakes(monkeypatch, [2.0, 3.0], baseline_rmse=1.0)
    runner = harness_runner.HarnessRunner(PARAMS, "goal", ROWS, FakeStore(), "run2")
    state = await runner.run()

    assert state["baseline_rmse"] == 1.0
    assert state["best_score"] == pytest.approx(1 / 2.0)
    assert not any(e.kind == "best" for e in runner.memory.entries.values())
    assert state["test"]["best_rmse"] is None
    assert state["test"]["beats_baseline"] is None


@pytest.mark.asyncio
async def test_held_out_test_rows_are_never_passed_to_agents_or_selection(monkeypatch):
    install_fakes(monkeypatch, [0.5, 0.5])
    seen = []
    original = shared_brain.run_experiment

    async def spy(code, rows, timeout=20.0, train=()):
        seen.append((code, {r["x"] for r in rows}))
        return await original(code, rows, timeout=timeout, train=train)

    monkeypatch.setattr(shared_brain, "run_experiment", spy)
    runner = harness_runner.HarnessRunner(PARAMS, "goal", ROWS, FakeStore(), "run3")
    await runner.run()

    _, _, test = shared_brain.split_rows(ROWS)
    test_xs = {r["x"] for r in test}
    touching = [i for i, (_, xs) in enumerate(seen) if xs & test_xs]
    assert touching == list(range(len(seen) - 4, len(seen)))
    assert seen[-1][0] == "rmse=0.5"


def test_split_is_deterministic_and_disjoint():
    train, val, test = shared_brain.split_rows(ROWS)
    assert len(train) + len(val) + len(test) == len(ROWS)
    keys = [{r["x"] for r in part} for part in (train, val, test)]
    assert not (keys[0] & keys[1]) and not (keys[0] & keys[2]) and not (keys[1] & keys[2])
    assert shared_brain.split_rows(ROWS) == (train, val, test)


def test_build_program_hides_target_from_predict_and_fits_on_train_only():
    from app.services import benchmarker

    rows = [{"a": 1.0, "y": 999.0}]
    code = (
        "def fit(rows):\n"
        "    return rows[0]['y']\n"
        "def predict(row, params):\n"
        "    assert 'y' not in row\n"
        "    return params\n"
    )
    program = benchmarker.build_program(code, rows, train=[{"a": 0.0, "y": 7.0}])
    import contextlib
    import io
    import json

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(program, {})  # noqa: S102 - the harness's own benchmark wrapper
    metrics = json.loads(out.getvalue().strip().splitlines()[-1])
    assert metrics["n"] == 1 and metrics["rmse"] == pytest.approx(992.0)


@pytest.mark.asyncio
async def test_stronger_linear_baseline_becomes_the_incumbent(monkeypatch):
    install_fakes(monkeypatch, [5.0, 5.0], baseline_rmse=3.0, linear_rmse=1.0)
    runner = harness_runner.HarnessRunner(PARAMS, "goal", ROWS, FakeStore(), "run4")
    state = await runner.run()

    assert state["baseline_name"] == "linear"
    assert state["baseline_rmse"] == 1.0
    assert state["test"]["incumbent"] == "linear"


def test_linear_baseline_recovers_exact_linear_relationship():
    code = shared_brain.linear_baseline_code(["a"])
    ns: dict = {}
    exec(code, ns)  # noqa: S102 - the generated baseline itself
    params = ns["fit"]([{"a": 1.0, "y": 3.0}, {"a": 2.0, "y": 5.0}, {"a": 3.0, "y": 7.0}])
    assert ns["predict"]({"a": 10.0}, params) == pytest.approx(21.0, abs=1e-6)
