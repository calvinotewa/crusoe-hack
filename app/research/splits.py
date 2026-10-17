"""Three-way split with the test split held out from every agent-facing view."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    train: list[dict]
    val: list[dict]
    test: list[dict]


def split_rows(rows: list[dict], seed: int = 42, fractions=(0.70, 0.15, 0.15)) -> Split:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    return Split(
        train=shuffled[:n_train],
        val=shuffled[n_train : n_train + n_val],
        test=shuffled[n_train + n_val :],
    )


def agent_view(split: Split, target: str) -> dict:
    """What research agents may see: train rows, and val inputs with labels removed."""
    val_inputs = [{k: v for k, v in row.items() if k != target} for row in split.val]
    return {"train": split.train, "val_inputs": val_inputs}


def score_val(predict: Callable[[dict], float], split: Split, target: str, loss) -> float:
    """Orchestrator-side scoring on val labels; never called from inside an agent sandbox."""
    preds = [predict({k: v for k, v in r.items() if k != target}) for r in split.val]
    return loss([r[target] for r in split.val], preds)


def score_test(predict: Callable[[dict], float], split: Split, target: str, loss) -> float:
    """Final scoring only. Research code must not call this before the run ends."""
    preds = [predict({k: v for k, v in r.items() if k != target}) for r in split.test]
    return loss([r[target] for r in split.test], preds)
