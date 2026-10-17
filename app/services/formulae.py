"""The formulae tool layer: store, list and fetch formulae (harness step 2).

`predict` code convention: `code` must define `predict(row: dict) -> float`, where `row`
is one CSV row as a dict (features as floats; `y` is the target). This is what both
the benchmarker (step 3) and the worker agents (step 7) produce and consume.
"""

import re

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Formula, HarnessSession

FORMULA_SCHEMA = '''def predict(row: dict) -> float:
    """row: one validation row as a dict of column name -> value. Return the predicted y."""
    ...
'''


def extract_code(text: str) -> str:
    """Pull the first fenced ```python block out of a completion, or return it unchanged."""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


async def store_formulae(
    session_id,
    formulae: list[dict],
    generation: int = 0,
    source: str = "agent",
    parent_id=None,
) -> list:
    """Store formulae for a harness session. Each dict: name, expression, code, description,
    rationale. Returns the created rows."""
    async with SessionLocal() as session:
        harness = await session.get(HarnessSession, session_id)
        if not harness:
            raise ValueError(f"harness session {session_id} not found")
        rows = [
            Formula(
                session_id=session_id,
                generation=generation,
                source=source,
                parent_id=parent_id,
                **f,
            )
            for f in formulae
        ]
        session.add_all(rows)
        await session.commit()
        for r in rows:
            await session.refresh(r)
        return rows


async def list_formulae(session_id, generation: int | None = None) -> list[Formula]:
    async with SessionLocal() as session:
        stmt = select(Formula).where(Formula.session_id == session_id)
        if generation is not None:
            stmt = stmt.where(Formula.generation == generation)
        return list((await session.scalars(stmt.order_by(Formula.created_at))).all())


async def get_formula(formula_id) -> Formula | None:
    async with SessionLocal() as session:
        return await session.get(Formula, formula_id)
