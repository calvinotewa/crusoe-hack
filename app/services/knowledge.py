"""Step 6: rewrite the system prompt for the next generation and expose the RAG read tool.

The knowledge prompt has three parts (per the brief):
  1. the general human-written task (the session `goal`),
  2. consolidated inherited learnings (step 4 summary + top formulae + their rationales),
  3. research learnings (step 5 sources and snippets).

`read_findings` is the RAG tool worker agents can ask the operator to run (ranked against
a query, drawing from stored sources, formulae and session summaries).
"""

import logging

from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.models import Benchmark, Formula, HarnessSession, ResearchSource
from app.services import subagent

log = logging.getLogger(__name__)

REWRITE_PROMPT = """You maintain the system prompt for a colony of formula-discovery agents.

## Part 1 — the general human-written task
{goal}

## Part 2 — consolidated inherited learnings (from the best agents so far)
{inherited}

## Part 3 — research learnings (web sources found for the knowledge gaps)
{research}

Rewrite this into ONE system prompt (keep the three sections as headings) for the next
generation of worker agents. Make parts 2 and 3 specific and actionable: what worked,
what to avoid, what to try first. Keep it under ~600 words. Reply with ONLY the prompt."""


def _simple_rank(text: str, query: str) -> float:
    """Cheap lexical relevance: fraction of query terms present in the text."""
    terms = [t for t in set(query.lower().split()) if len(t) > 3]
    if not terms:
        return 0.0
    low = text.lower()
    return sum(1 for t in terms if t in low) / len(terms)


async def read_findings(session_id, query: str, limit: int = 5, formula_id=None) -> list[dict]:
    """RAG read tool over everything the harness knows: research sources, formulae and
    session summaries, ranked against `query`. Worker agents call this via the operator
    (it's exposed on the API as GET /sessions/{id}/findings)."""
    async with SessionLocal() as session:
        srcs = list(
            (
                await session.scalars(
                    select(ResearchSource).where(ResearchSource.session_id == session_id)
                )
            ).all()
        )
        stmt = select(Formula).where(Formula.session_id == session_id)
        if formula_id:
            stmt = stmt.where(Formula.id == formula_id)
        forms = list((await session.scalars(stmt)).all())
        s = await session.get(HarnessSession, session_id)

    candidates: list[tuple[float, str, str, str | None]] = []

    def _add(text: str, title: str, url: str | None) -> None:
        score = _simple_rank(text, query)
        if score > 0:
            candidates.append((score, title, text[:2000], url))

    for src in srcs:
        _add(f"{src.title} {src.snippet or ''} {src.url}", src.title, src.url)
    for f in forms:
        _add(
            f"{f.expression}\n{f.description or ''}\n{f.rationale or ''}",
            f"Formula: {f.name}",
            None,
        )
    if s and s.summary:
        _add(s.summary, "Session summary", None)

    candidates.sort(key=lambda c: c[0], reverse=True)
    return [
        {"title": t, "text": text, "score": sc, "url": u} for sc, t, text, u in candidates[:limit]
    ]


async def rewrite_prompt(session: HarnessSession) -> str:
    """Rewrite the knowledge prompt for the next generation (all three parts)."""
    async with SessionLocal() as db:
        formulae = list(
            (
                await db.scalars(
                    select(Formula)
                    .where(Formula.session_id == session.id)
                    .order_by(Formula.created_at.desc())
                )
            ).all()
        )
        top = formulae[:6]
        scores = {}
        for f in top:
            bench = (
                await db.scalars(
                    select(Benchmark)
                    .where(Benchmark.formula_id == f.id)
                    .order_by(Benchmark.score.desc())
                )
            ).first()
            if bench:
                scores[f.id] = bench.score
        sources = list(
            (
                await db.scalars(
                    select(ResearchSource)
                    .where(ResearchSource.session_id == session.id)
                    .order_by(ResearchSource.created_at.desc())
                )
            ).all()
        )[:10]

    s = get_settings()
    inherited = session.summary or "(nothing yet)"
    if top:
        inherited += "\n\nBest formulae so far:\n" + "\n".join(
            f"- {f.name}: {f.expression} (score {scores.get(f.id, '?')}) — {f.rationale or ''}"
            for f in top
        )
    research = (
        "\n".join(f"- {src.title}: {src.snippet or src.url}" for src in sources) or "(none yet)"
    )

    return await subagent.run_sub_agent(
        REWRITE_PROMPT.format(goal=session.goal, inherited=inherited, research=research),
        model=s.crusoe_judge_model,
        temperature=0.2,
        max_tokens=2048,
    )
