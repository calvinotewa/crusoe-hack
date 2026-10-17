"""Matched-budget memory ablation: one-shot, no memory, raw log, and curated Shared Brain."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Protocol

from app.research.curation import Report, curate, rank
from app.research.memory import Memory, count_tokens
from app.research.scoring import beats, gain, noise_threshold

SETUPS = ("one_shot", "no_memory", "raw_log", "shared_brain")
FOCUS = ("optimizer", "architecture", "regularization", "data")
RAW_LOG_TOKEN_CAP = 100_000
SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class Budget:
    researchers: int
    runs_per_researcher: int
    curator_runs: int

    @property
    def runs_per_generation(self) -> int:
        return self.researchers * self.runs_per_researcher + self.curator_runs


BUDGETS = {
    "one_shot": Budget(researchers=1, runs_per_researcher=1, curator_runs=0),
    "no_memory": Budget(researchers=4, runs_per_researcher=7, curator_runs=0),
    "raw_log": Budget(researchers=4, runs_per_researcher=7, curator_runs=0),
    "shared_brain": Budget(researchers=4, runs_per_researcher=6, curator_runs=4),
}


class Lab(Protocol):
    """Everything the loop needs from the outside world. Real adapters: LLM + Modal sandbox."""

    baseline_code: str

    def baseline_seed_losses(self, seeds: tuple[int, ...]) -> list[float]: ...

    def run_researcher(
        self, prompt: str, best_code: str, runs: int, researcher: int, gen: int
    ) -> list[tuple[str, float, str]]:
        """Returns (run_id, val_loss, code) for each run the researcher used."""
        ...

    def verify(self, code: str) -> list[float]:
        """Mean and spread are computed from these seed losses on val, 3 seeds."""
        ...


@dataclass
class GenRecord:
    gen: int
    best_loss: float
    memory_tokens: int
    runs_used: int


@dataclass
class History:
    setup: str
    l0: float
    threshold: float
    records: list[GenRecord] = field(default_factory=list)
    final_code: str = ""
    final_loss: float = float("nan")


def build_prompt(setup_text: str, best_code: str, focus: str) -> str:
    return (
        f"Focus: {focus}\n\nMemory:\n{setup_text or '(none)'}\n\n"
        f"Best train.py so far:\n{best_code}\n\nPropose and run experiments."
    )


def _trim_raw_log(entries: list[str]) -> str:
    kept: list[str] = []
    total = 0
    for entry in reversed(entries):
        cost = count_tokens(entry)
        if total + cost > RAW_LOG_TOKEN_CAP:
            break
        kept.append(entry)
        total += cost
    return "\n".join(reversed(kept))


def run_setup(setup: str, lab: Lab, generations: int) -> History:
    if setup not in SETUPS:
        raise ValueError(f"unknown setup {setup!r}")
    budget = BUDGETS[setup]

    seed_losses = lab.baseline_seed_losses(SEEDS)
    l0 = statistics.fmean(seed_losses)
    threshold = noise_threshold(l0, seed_losses)
    history = History(setup=setup, l0=l0, threshold=threshold)

    best_code, best_loss = lab.baseline_code, l0
    memory = Memory()
    raw_entries: list[str] = []

    for gen in range(generations):
        if setup == "shared_brain":
            setup_text = memory.render()
        elif setup == "raw_log":
            setup_text = _trim_raw_log(raw_entries)
        else:
            setup_text = ""

        reports: list[Report] = []
        for i in range(budget.researchers):
            focus = FOCUS[i % len(FOCUS)]
            prompt = build_prompt(setup_text, best_code, focus)
            runs = lab.run_researcher(prompt, best_code, budget.runs_per_researcher, i, gen)
            run_id, loss, code = min(runs, key=lambda r: r[1])
            reports.append(
                Report(
                    researcher=i,
                    run_ids=[r[0] for r in runs],
                    best_run_id=run_id,
                    best_loss=loss,
                    best_code=code,
                    text=f"{focus}: best val loss {loss:.4f} ({run_id})",
                )
            )
            if setup == "raw_log":
                raw_entries.append(f"gen {gen} researcher {i}: {reports[-1].text}")

        ranked = rank(reports, best_loss)

        if setup == "shared_brain":
            result = curate(memory, ranked, gen, best_loss, threshold, lab.verify)
            if result.new_best is not None:
                best_code, best_loss = result.new_best.best_code, result.new_best_loss
        else:
            candidate = ranked[0]
            if beats(candidate.best_loss, best_loss, threshold):
                best_code, best_loss = candidate.best_code, candidate.best_loss

        history.records.append(
            GenRecord(
                gen=gen,
                best_loss=best_loss,
                memory_tokens=memory.tokens()
                if setup == "shared_brain"
                else count_tokens(setup_text),
                runs_used=budget.runs_per_generation,
            )
        )

    history.final_code = best_code
    history.final_loss = best_loss
    return history


def run_ablation(lab_for_setup, generations: int, setups=SETUPS) -> dict[str, History]:
    """Run each setup with a fresh lab. Refuses to run unless all setups spend equal runs."""
    totals = {s: BUDGETS[s].runs_per_generation for s in setups if s != "one_shot"}
    if len(set(totals.values())) > 1:
        raise AssertionError(f"unequal budgets: {totals}")
    return {
        s: run_setup(s, lab_for_setup(s), 1 if s == "one_shot" else generations) for s in setups
    }


def final_gain(history: History) -> float:
    return gain(history.l0, history.final_loss)
