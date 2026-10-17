"""Curated shared memory: kinds, token cap, drop order, staleness, stable rendering."""

from __future__ import annotations

from dataclasses import dataclass, field

KIND_ORDER = ("best", "works", "failed", "try_next")
TOKEN_CAP = 2000
CLAIM_MAX_CHARS = 200
STALE_AFTER_VERSIONS = 2


class MemoryFull(Exception):
    pass


def count_tokens(text: str) -> int:
    return len(text) // 4


@dataclass
class Entry:
    id: int
    kind: str
    claim: str
    run_ids: list[str]
    tested_on: int
    created_gen: int
    updated_gen: int


@dataclass
class Memory:
    entries: dict[int, Entry] = field(default_factory=dict)
    best_version: int = 0
    events: list[tuple[str, int | None, str]] = field(default_factory=list)
    _next_id: int = 1

    def render(self) -> str:
        order = {k: i for i, k in enumerate(KIND_ORDER)}
        lines = []
        for e in sorted(self.entries.values(), key=lambda e: (order[e.kind], e.id)):
            lines.append(f"[{e.kind} #{e.id}] {e.claim} (runs: {', '.join(e.run_ids)})")
        return "\n".join(lines)

    def tokens(self) -> int:
        return count_tokens(self.render())

    def upsert(
        self,
        kind: str,
        claim: str,
        run_ids: list[str],
        gen: int,
        id: int | None = None,
    ) -> Entry:
        if kind not in KIND_ORDER:
            raise ValueError(f"unknown kind {kind!r}")
        if len(claim) > CLAIM_MAX_CHARS:
            raise ValueError(f"claim longer than {CLAIM_MAX_CHARS} characters")
        if id is None:
            new_id = self._next_id
            created = gen
        else:
            new_id = id
            created = self.entries[id].created_gen
        entry = Entry(new_id, kind, claim, run_ids, self.best_version, created, gen)
        trial = Memory(
            entries={**self.entries, new_id: entry},
            best_version=self.best_version,
        )
        if trial.tokens() > TOKEN_CAP:
            raise MemoryFull(f"memory would be {trial.tokens()} tokens, cap is {TOKEN_CAP}")
        if id is None:
            self._next_id += 1
        self.entries[new_id] = entry
        self.events.append(("upsert", entry.id, claim))
        return entry

    def remove(self, id: int, reason: str) -> None:
        del self.entries[id]
        self.events.append(("remove", id, reason))

    def set_best(self, claim: str, run_ids: list[str], gen: int) -> Entry:
        for e in [e for e in self.entries.values() if e.kind == "best"]:
            self.remove(e.id, "superseded by new best")
        self.best_version += 1
        return self.upsert("best", claim, run_ids, gen)

    def compact(self) -> list[int]:
        """Drop oldest try_next, then oldest failed, until under cap. Never drops best or works."""
        dropped = []
        for kind in ("try_next", "failed"):
            victims = sorted(
                (e for e in self.entries.values() if e.kind == kind),
                key=lambda e: (e.created_gen, e.id),
            )
            for e in victims:
                if self.tokens() <= TOKEN_CAP:
                    return dropped
                self.remove(e.id, f"over cap: dropped oldest {kind}")
                dropped.append(e.id)
        return dropped

    def stale(self) -> list[Entry]:
        return [
            e
            for e in self.entries.values()
            if e.kind != "best" and self.best_version - e.tested_on >= STALE_AFTER_VERSIONS
        ]
