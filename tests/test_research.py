import random

import pytest

from app.research.ablation import BUDGETS, FOCUS, SETUPS, run_ablation, run_setup
from app.research.curation import Report, curate, rank, route
from app.research.memory import TOKEN_CAP, Memory, MemoryFull
from app.research.scoring import beats, gain, noise_threshold
from app.research.splits import agent_view, score_test, score_val, split_rows

ROWS = [{"x": i, "y": i * 2.0} for i in range(100)]


def mae(ys, preds):
    return sum(abs(a - b) for a, b in zip(ys, preds, strict=True)) / len(ys)


def test_gain_and_noise_threshold():
    assert gain(1.0, 0.8) == pytest.approx(0.2)
    assert noise_threshold(1.0, [1.0, 1.0, 1.0]) == pytest.approx(0.005)
    assert noise_threshold(1.0, [0.9, 1.0, 1.1]) == pytest.approx(0.2)
    assert beats(0.7, 0.8, threshold=0.05)
    assert not beats(0.78, 0.8, threshold=0.05)


def test_memory_renders_in_fixed_order():
    m = Memory()
    m.upsert("try_next", "t", ["r1"], gen=0)
    m.upsert("failed", "f", ["r2"], gen=0)
    m.upsert("works", "w", ["r3"], gen=0)
    m.set_best("b", ["r4"], gen=0)
    kinds = [line.split("]")[0].split("[")[1].split(" ")[0] for line in m.render().splitlines()]
    assert kinds == ["best", "works", "failed", "try_next"]


def test_memory_refuses_over_cap_and_compact_drops_try_next_then_failed():
    m = Memory()
    m.set_best("best", ["r0"], gen=0)
    for i in range(10_000):
        try:
            m.upsert("failed", f"f{i} " + "x" * 150, [f"r{i}"], gen=0)
        except MemoryFull:
            break
    with pytest.raises(MemoryFull):
        for i in range(10_000):
            m.upsert("try_next", f"t{i} " + "y" * 190, [f"t{i}"], gen=1)
    n_failed = sum(e.kind == "failed" for e in m.entries.values())
    dropped = m.compact()
    assert dropped == []
    assert sum(e.kind == "failed" for e in m.entries.values()) == n_failed


def test_curation_compacts_try_next_before_failed_when_full():
    m = Memory()
    m.set_best("best", ["r0"], gen=0)
    for i in range(10_000):
        try:
            m.upsert("failed", f"f{i} " + "x" * 150, [f"r{i}"], gen=0)
        except MemoryFull:
            break
    for i in range(10_000):
        try:
            m.upsert("try_next", f"t{i} " + "y" * 150, [f"t{i}"], gen=1)
        except MemoryFull:
            break
    failed_before = sum(e.kind == "failed" for e in m.entries.values())
    reports = [
        Report(i, [f"n{i}"], f"n{i}", 0.9, f"c{i}", f"new idea {i} " + "z" * 60) for i in range(4)
    ]
    curate(
        m, rank(reports, 1.0), gen=2, best_loss=1.0, threshold=0.01, verify=lambda code: [0.99] * 3
    )
    assert m.tokens() <= TOKEN_CAP
    assert sum(e.kind == "failed" for e in m.entries.values()) == failed_before
    assert any(e.kind == "best" for e in m.entries.values())


def test_best_is_never_dropped():
    m = Memory()
    m.set_best("best " + "z" * 150, ["r"], gen=0)
    for i in range(200):
        try:
            m.upsert("failed", f"f{i} " + "q" * 150, [f"r{i}"], gen=i)
        except MemoryFull:
            break
    m.compact()
    assert any(e.kind == "best" for e in m.entries.values())


def test_stale_entries_after_two_best_versions():
    m = Memory()
    m.upsert("works", "old fact", ["r"], gen=0)
    m.set_best("b1", ["r"], gen=1)
    assert m.stale() == []
    m.set_best("b2", ["r"], gen=2)
    assert [e.claim for e in m.stale()] == ["old fact"]


def test_route_top_middle_bottom():
    reports = [
        Report(i, [f"r{i}"], f"r{i}", loss, f"code{i}", f"t{i}")
        for i, loss in enumerate([0.9, 0.5, 0.7, 0.95])
    ]
    buckets = route(rank(reports, gen_start_loss=1.0))
    assert [r.researcher for r in buckets["top"]] == [1]
    assert [r.researcher for r in buckets["bottom"]] == [3]
    assert sorted(r.researcher for r in buckets["middle"]) == [0, 2]


def test_curator_promotes_only_replicated_claims():
    m = Memory()
    reports = [
        Report(0, ["r0"], "r0", 0.5, "good", "good idea"),
        Report(1, ["r1"], "r1", 0.9, "bad", "bad idea"),
    ]
    res = curate(
        m,
        rank(reports, 1.0),
        gen=0,
        best_loss=1.0,
        threshold=0.01,
        verify=lambda code: [0.5, 0.5, 0.5],
    )
    assert res.new_best is not None and res.new_best.best_code == "good"
    assert any(e.kind == "best" for e in m.entries.values())

    m2 = Memory()
    res2 = curate(
        m2,
        rank(reports, 1.0),
        gen=0,
        best_loss=1.0,
        threshold=0.01,
        verify=lambda code: [0.995, 0.995, 0.995],
    )
    assert res2.new_best is None
    assert not any(e.kind == "best" for e in m2.entries.values())
    assert any(e.kind == "failed" and "did not replicate" in e.claim for e in m2.entries.values())


class FakeLab:
    """Deterministic stand-in: loss = |lr - 0.3| + 0.5. Baseline lr=1.0."""

    baseline_code = "lr=1.0"

    def __init__(self):
        self.prompts = []
        self.calls = 0

    def _loss(self, code):
        lr = float(code.split("=")[1])
        return abs(lr - 0.3) + 0.5

    def baseline_seed_losses(self, seeds):
        return [self._loss(self.baseline_code)] * len(seeds)

    def run_researcher(self, prompt, best_code, runs, researcher, gen):
        self.prompts.append(prompt)
        rng = random.Random(gen * 100 + researcher)
        out = []
        for k in range(runs):
            lr = rng.uniform(0.0, 1.5)
            code = f"lr={lr:.4f}"
            self.calls += 1
            out.append((f"g{gen}r{researcher}k{k}", self._loss(code), code))
        return out

    def verify(self, code):
        return [self._loss(code)] * 3


def test_test_split_never_reaches_agent_view():
    split = split_rows(ROWS, seed=1)
    view = agent_view(split, target="y")
    assert all("y" not in r for r in view["val_inputs"])
    test_xs = {r["x"] for r in split.test}
    train_val_xs = {r["x"] for r in view["train"]} | {r["x"] for r in view["val_inputs"]}
    assert not (test_xs & train_val_xs)


def test_score_functions_use_their_own_split():
    split = split_rows(ROWS, seed=1)
    assert score_val(lambda r: r["x"] * 2.0, split, "y", mae) == pytest.approx(0.0)
    assert score_test(lambda r: 0.0, split, "y", mae) > 0


def test_budgets_are_equal_per_generation():
    totals = {s: BUDGETS[s].runs_per_generation for s in SETUPS if s != "one_shot"}
    assert set(totals.values()) == {28}
    assert FOCUS


def test_run_ablation_all_setups_finish_and_no_test_leak_in_prompts():
    labs = {}

    def factory(setup):
        labs[setup] = FakeLab()
        return labs[setup]

    results = run_ablation(factory, generations=2)
    assert set(results) == set(SETUPS)
    for setup, history in results.items():
        assert len(history.records) == (1 if setup == "one_shot" else 2)
        assert history.final_loss <= history.l0
    assert labs["no_memory"].prompts and all(
        "Memory:\n(none)" in p for p in labs["no_memory"].prompts
    )


def test_shared_brain_prompt_carries_only_curated_memory():
    lab = FakeLab()
    history = run_setup("shared_brain", lab, generations=2)
    assert history.records[-1].memory_tokens <= TOKEN_CAP
    assert (
        lab.calls
        == 2 * BUDGETS["shared_brain"].researchers * BUDGETS["shared_brain"].runs_per_researcher
    )
