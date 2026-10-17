"""Curator gate: rank researchers by gain, verify the top claim on seeds, route the rest."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass

from app.research.memory import Memory, MemoryFull
from app.research.scoring import beats, gain


@dataclass(frozen=True)
class Report:
    researcher: int
    run_ids: list[str]
    best_run_id: str
    best_loss: float
    best_code: str
    text: str


def rank(reports: list[Report], gen_start_loss: float) -> list[Report]:
    return sorted(reports, key=lambda r: gain(gen_start_loss, r.best_loss), reverse=True)


def route(ranked: list[Report]) -> dict[str, list[Report]]:
    """Top 1 is verified, bottom 1 is recorded as failed, the rest are try_next candidates."""
    if len(ranked) == 1:
        return {"top": ranked, "middle": [], "bottom": []}
    return {"top": ranked[:1], "middle": ranked[1:-1], "bottom": ranked[-1:]}


@dataclass(frozen=True)
class CurationResult:
    new_best: Report | None
    new_best_loss: float | None


def _add(memory: Memory, kind: str, claim: str, run_ids: list[str], gen: int) -> None:
    """Upsert, compacting once (oldest try_next, then failed) if the cap refuses the write."""
    try:
        memory.upsert(kind, claim, run_ids, gen)
    except MemoryFull:
        memory.compact()
        try:
            memory.upsert(kind, claim, run_ids, gen)
        except MemoryFull:
            pass


def curate(
    memory: Memory,
    ranked: list[Report],
    gen: int,
    best_loss: float,
    threshold: float,
    verify: Callable[[str], list[float]],
) -> CurationResult:
    buckets = route(ranked)
    new_best = None
    new_best_loss = None

    for r in buckets["top"]:
        seed_losses = verify(r.best_code)
        mean = statistics.fmean(seed_losses)
        if beats(mean, best_loss, threshold):
            _add(
                memory,
                "works",
                f"{r.text} (val {best_loss:.4f} -> {mean:.4f} on 3 seeds)"[:200],
                r.run_ids,
                gen,
            )
            new_best, new_best_loss = r, mean
        else:
            _add(memory, "failed", f"did not replicate: {r.text}"[:200], r.run_ids, gen)

    for r in buckets["middle"]:
        _add(memory, "try_next", r.text[:200], r.run_ids, gen)

    for r in buckets["bottom"]:
        _add(memory, "failed", r.text[:200], r.run_ids, gen)

    if new_best is not None:
        memory.set_best(new_best.text[:200], new_best.run_ids, gen)
    return CurationResult(new_best, new_best_loss)
