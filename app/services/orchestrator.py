"""The orchestration harness: a generational loop that tries to beat a benchmark.

One session, per the hackathon brief, runs these steps per generation:

1. (once, at session creation) Ingest the validation CSV + seed formulae into the
   multi-tenant store (Neon bucket + Postgres rows); see routes/sessions.py.
2. runSubAgent x N: workers propose alternative formulae for the maths goal
   (services/subagent.py + services/formulae.py: store_formulae).
3. Run every formula of the generation against the validation CSV in Modal sandboxes
   and store benchmarks (services/benchmarker.py).
4. Summarise the session thinking of the best 3 and consolidate the learnings
   (services/research.summarise_session).
5. Research: "if the agents knew X" — fetch `research_num_sources` new sources via
   you.com (services/research.research_gaps).
6. Rewrite the system prompt: general human task + consolidated inherited learnings +
   research learnings (services/knowledge.rewrite_prompt); expose the RAG read tool
   (knowledge.read_findings, GET /sessions/{id}/findings).
7. The next generation's workers use the new prompt to propose formulae that try to
   beat the benchmark (worker_prompt in services/subagent.py).

Steps 2..7 repeat for `harness_max_generations` generations (or more via
POST /sessions/{id}/iterate). All provider calls (Crusoe, Modal, you.com, bucket) stay
inside services; the loop is one BackgroundTask per request, like evaluator.execute_run.
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.models import Benchmark, Formula, HarnessSession, HarnessStatus
from app.services import benchmarker, knowledge, research, subagent
from app.services.formulae import store_formulae

log = logging.getLogger(__name__)

SELECTION_FRACTION = 0.1  # top 10% of a generation writes the knowledge prompt (brief)


async def _top_formulae(session_id, generation: int, n: int = 3):
    """The n best (formula, benchmark) pairs of one generation, by score."""
    async with SessionLocal() as session:
        stmt = (
            select(Formula, Benchmark)
            .join(Benchmark, Benchmark.formula_id == Formula.id)
            .where(Formula.session_id == session_id, Benchmark.generation == generation)
            .order_by(Benchmark.score.desc().nulls_last())
        )
        return [(f, b) for f, b in (await session.execute(stmt)).all()][:n]


async def run_generation(session_id, generation: int) -> float:
    """Steps 2-7 for one generation. Returns the generation's best score."""
    s = get_settings()

    async with SessionLocal() as session:
        harness = await session.get(HarnessSession, session_id)
        goal, system_prompt = harness.goal, harness.system_prompt
        # Step 3 input: the validation CSV (bucket) for the worker prompts + benchmarks.
        rows = await benchmarker.get_validation_rows(harness)
    columns = sorted(set(rows[0].keys()) - {"y"}) if rows else []

    # --- Step 2: runSubAgent x N — workers propose alternative formulae ---
    sem = asyncio.Semaphore(4)  # Crusoe free tier: 30 RPM

    async def _worker(n: int) -> dict:
        async with sem:
            return await subagent.run_json(
                subagent.worker_prompt(goal, columns, n),
                shape="name, expression, description, code, rationale",
                system_prompt=system_prompt,
            )

    n_agents = s.harness_num_agents
    answers = await asyncio.gather(*(_worker(i + 1) for i in range(n_agents)))

    # --- store formulae with lineage: parent = the current best formula ---
    parent_id = None
    top_prev = await _top_formulae(session_id, generation - 1, 1) if generation > 0 else []
    if top_prev:
        parent_id = top_prev[0][0].id
    created = await store_formulae(
        session_id,
        [
            {
                "name": a.get("name") or f"gen{generation}-agent{n}",
                "expression": a.get("expression", ""),
                "description": a.get("description"),
                "code": a.get("code"),
                "rationale": a.get("rationale"),
            }
            for n, a in enumerate(answers, start=1)
            if a
        ],
        generation=generation,
        source="agent",
        parent_id=parent_id,
    )

    # --- Step 3: benchmark all of them against the validation CSV in sandboxes ---
    metrics = await benchmarker.benchmark_generation(session_id, generation)
    by_formula = {m["formula_id"]: m for m in metrics}
    for f in created:
        m = by_formula.get(str(f.id), {})
        await benchmarker.store_benchmark(f.id, generation, m)

    # --- Step 4: best 3, summarise the session thinking, consolidate ---
    top = await _top_formulae(session_id, generation, 3)
    best = max((b.score for _, b in top if b.score is not None), default=None)
    async with SessionLocal() as session:
        harness = await session.get(HarnessSession, session_id)
        harness.summary = await research.summarise_session(
            harness, [f for f, _ in top], [b for _, b in top]
        )
        await session.commit()

    # --- Step 5: research the gaps ("if the agents knew X") via you.com ---
    query, sources = await research.research_gaps(
        harness, [f for f, _ in top], [b for _, b in top], generation
    )
    async with SessionLocal() as session:
        h2 = await session.get(HarnessSession, session_id)
        h2.research_query = query
        await session.commit()

    # --- Step 6: rewrite the system prompt (task + inherited + research) for gen n+1 ---
    new_prompt = await knowledge.rewrite_prompt(harness)
    async with SessionLocal() as session:
        h3 = await session.get(HarnessSession, session_id)
        h3.system_prompt = new_prompt
        known_best = [x for x in (h3.best_score, best) if x is not None]
        h3.best_score = max(known_best) if known_best else None
        await session.commit()

    return best or 0.0


async def execute_session(session_id: uuid.UUID, generations: int | None = None) -> None:
    """Run the harness loop for a session: `generations` rounds (default from settings)."""
    s = get_settings()
    async with SessionLocal() as session:
        harness = await session.get(HarnessSession, session_id)
        if not harness:
            raise ValueError(f"harness session {session_id} not found")
        harness.status = HarnessStatus.running
        await session.commit()
        start = harness.generation + 1

    try:
        # gen 0 = seeds (stored at session creation, step 1); loop runs 1..N
        for generation in range(start, start + (generations or s.harness_max_generations)):
            score = await run_generation(session_id, generation)
            async with SessionLocal() as session:
                h = await session.get(HarnessSession, session_id)
                h.generation = generation
                await session.commit()
            log.info("session %s generation %d best score %.4f", session_id, generation, score)
        async with SessionLocal() as session:
            h = await session.get(HarnessSession, session_id)
            h.status = HarnessStatus.completed
            h.finished_at = datetime.now(UTC)
            await session.commit()
    except Exception as e:
        log.exception("harness session %s failed", session_id)
        async with SessionLocal() as session:
            h = await session.get(HarnessSession, session_id)
            h.status = HarnessStatus.failed
            h.error = str(e)
            h.finished_at = datetime.now(UTC)
            await session.commit()
        raise
