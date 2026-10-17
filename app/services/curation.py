"""Curator gate and curated shared memory for the Shared Brain loop.

Each generation the curator ranks experiments by validation RMSE (lower is better, which is
the same order as gain over the generation's starting best). The top one is re-verified on
bootstrap resamples of the validation rows and promoted to best only if it beats the incumbent
by more than the noise threshold. The middle experiments become open questions (try_next),
and the bottom one, including any crashed experiment, is recorded as failed.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

KIND_ORDER = ("best", "works", "failed", "try_next")
TOKEN_CAP = 2000
CLAIM_MAX_CHARS = 200
STALE_AFTER_VERSIONS = 2
NOISE_FLOOR = 0.005


class MemoryFull(Exception):
    pass


def count_tokens(text: str) -> int:
    return len(text) // 4


@dataclass
class Entry:
    id: int
    kind: str
    claim: str
    gen: int
    tested_on: int


@dataclass
class Memory:
    entries: dict[int, Entry] = field(default_factory=dict)
    best_version: int = 0
    _next_id: int = 1

    def render(self) -> str:
        order = {k: i for i, k in enumerate(KIND_ORDER)}
        ordered = sorted(self.entries.values(), key=lambda e: (order[e.kind], e.id))
        return "\n".join(f"[{e.kind} #{e.id}] {e.claim}" for e in ordered)

    def tokens(self) -> int:
        return count_tokens(self.render())

    def upsert(self, kind: str, claim: str, gen: int) -> Entry:
        claim = claim[:CLAIM_MAX_CHARS]
        entry = Entry(self._next_id, kind, claim, gen, self.best_version)
        trial = Memory(entries={**self.entries, entry.id: entry}, best_version=self.best_version)
        if trial.tokens() > TOKEN_CAP:
            raise MemoryFull(f"memory would be {trial.tokens()} tokens, cap is {TOKEN_CAP}")
        self._next_id += 1
        self.entries[entry.id] = entry
        return entry

    def remove(self, id: int) -> None:
        del self.entries[id]

    def set_best(self, claim: str, gen: int) -> Entry:
        for e in [e for e in self.entries.values() if e.kind == "best"]:
            self.remove(e.id)
        self.best_version += 1
        try:
            return self.upsert("best", claim, gen)
        except MemoryFull:
            self.compact()
            return self.upsert("best", claim, gen)

    def compact(self) -> None:
        """Drop oldest try_next, then oldest failed, until under cap. Never drops best or works."""
        for kind in ("try_next", "failed"):
            victims = sorted(
                (e for e in self.entries.values() if e.kind == kind),
                key=lambda e: (e.gen, e.id),
            )
            for e in victims:
                if self.tokens() <= TOKEN_CAP:
                    return
                self.remove(e.id)

    def stale(self) -> list[Entry]:
        return [
            e
            for e in self.entries.values()
            if e.kind != "best" and self.best_version - e.tested_on >= STALE_AFTER_VERSIONS
        ]


def add(memory: Memory, kind: str, claim: str, gen: int) -> None:
    """Upsert; if the cap refuses it, compact once and retry, else skip the write."""
    try:
        memory.upsert(kind, claim, gen)
    except MemoryFull:
        memory.compact()
        try:
            memory.upsert(kind, claim, gen)
        except MemoryFull:
            pass


def loss_of(experiment: dict) -> float:
    rmse = experiment.get("rmse")
    if rmse is None or not math.isfinite(rmse):
        return math.inf
    return float(rmse)


def noise_threshold(incumbent_loss: float, verify_losses: list[float]) -> float:
    spread = statistics.stdev(verify_losses) if len(verify_losses) > 1 else 0.0
    return max(2 * spread, NOISE_FLOOR * incumbent_loss)


@dataclass(frozen=True)
class Promotion:
    experiment: dict
    loss: float


async def curate(
    memory: Memory,
    scored: list[dict],
    gen: int,
    incumbent_loss: float | None,
    verify: Callable[[str], Awaitable[list[float]]],
) -> Promotion | None:
    """Rank `scored` experiments, update `memory`, and return the promoted best, if any."""
    ranked = sorted(scored, key=loss_of)
    if not ranked:
        return None

    promoted: Promotion | None = None
    top = ranked[0]
    if loss_of(top) < math.inf:
        verified = await verify(top["code"])
        mean = statistics.fmean(verified)
        replicated = math.isfinite(mean)
        if replicated and incumbent_loss is not None:
            replicated = mean < incumbent_loss - noise_threshold(incumbent_loss, verified)
        if replicated:
            claim = f"{top.get('name')}: {top.get('hypothesis') or top.get('expression')}"
            add(
                memory,
                "works",
                f"{claim} (val rmse {loss_of(top):.4f} -> {mean:.4f} on resamples)",
                gen,
            )
            memory.set_best(claim, gen)
            promoted = Promotion(top, mean)
        else:
            add(memory, "failed", f"did not replicate: {top.get('name')} ({mean:.4f})", gen)

    for e in ranked[1:-1]:
        if loss_of(e) < math.inf:
            add(memory, "try_next", f"{e.get('hypothesis') or e.get('name')}", gen)

    if len(ranked) > 1:
        bottom = ranked[-1]
        add(
            memory,
            "failed",
            f"{bottom.get('name')}: {bottom.get('hypothesis') or 'no result'}",
            gen,
        )

    memory.compact()
    return promoted
