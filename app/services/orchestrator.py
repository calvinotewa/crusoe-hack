"""The orchestration harness (API path): a generational Shared Brain loop.

Per the spec, each generation runs:
1. Run all experiments (N researcher agents propose formulae; every formula is
   benchmarked against the validation CSV in Modal sandboxes).
2. Summarize the thinking of the top 10 experiments (curator model).
3. Supplement the summary's learnings with a web search on what could improve it.
4. Generate hypotheses on improvement ideas (curator model).
5. The next generation's agents produce experiments based on those hypotheses.

`app/services/shared_brain.py` holds the same flow as in-memory pure functions;
`scripts/shared_brain_mvp.py` runs it standalone with only CRUSOE_API_KEY.
All provider calls (Crusoe, Modal, you.com, bucket) stay inside services; the loop
is one BackgroundTask per request, like evaluator.execute_run.
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.models import Benchmark, Formula, HarnessSession, HarnessStatus
from app.services import benchmarker, knowledge, research, shared_brain, subagent
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


async def run_generation(session_id, generation: int, hypotheses: list[str], memory: str):
    """One generation of the Shared Brain loop. Returns (best, next_hypotheses, memory).

    1. Run all experiments (N runSubAgent workers -> Modal-sandboxed benchmarks).
    2. Summarize the thinking of the top 10 experiments (curator).
    3. Supplement the summary with a web search on what could improve it.
    4. Generate `harness_num_hypotheses` improvement hypotheses (curator).
    5. Next generation's agents get the hypotheses round-robin (caller passes them back).
    """
    s = get_settings()

    async with SessionLocal() as session:
        harness = await session.get(HarnessSession, session_id)
        goal, system_prompt = harness.goal, harness.system_prompt
        # Step 1 input: the validation CSV (bucket) for the worker prompts + benchmarks.
        rows = await benchmarker.get_validation_rows(harness)
    columns = sorted(set(rows[0].keys()) - {"y"}) if rows else []

    # --- Step 5/1: N agents produce experiments testing their assigned hypotheses ---
    sem = asyncio.Semaphore(4)  # Crusoe free tier: 30 RPM

    async def _worker(n: int) -> dict:
        async with sem:
            return await subagent.run_json(
                subagent.worker_prompt(
                    goal,
                    columns,
                    n,
                    hypothesis=hypotheses[(n - 1) % len(hypotheses)] if hypotheses else "",
                    memory=memory,
                ),
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

    # --- Step 1 cont.: benchmark all of them against the validation CSV in sandboxes ---
    metrics = await benchmarker.benchmark_generation(session_id, generation)
    by_formula = {m["formula_id"]: m for m in metrics}
    for f in created:
        m = by_formula.get(str(f.id), {})
        await benchmarker.store_benchmark(f.id, generation, m)

    # --- Step 2: summarize the thinking of the top 10 experiments ---
    top = await _top_formulae(session_id, generation, s.harness_top_k)
    best = max((b.score for _, b in top if b.score is not None), default=None)
    async with SessionLocal() as session:
        harness = await session.get(HarnessSession, session_id)
        harness.summary = await research.summarise_session(
            harness, [f for f, _ in top], [b for _, b in top]
        )
        await session.commit()

    # --- Step 3: supplement the summary with a web search on what could improve it ---
    query, sources = await research.research_gaps(
        harness, [f for f, _ in top], [b for _, b in top], generation
    )
    async with SessionLocal() as session:
        h2 = await session.get(HarnessSession, session_id)
        h2.research_query = query
        await session.commit()
    learnings = (
        "\n".join(f"- {src.title}: {src.snippet or src.url}" for src in sources) or "(no results)"
    )

    # --- Step 4: generate N hypotheses on improvement ideas (curator) ---
    next_hypotheses = await shared_brain.generate_hypotheses(
        goal, harness.summary or "", learnings, s.harness_num_hypotheses, concurrency=4
    )

    # --- Step 6 (unchanged): rewrite the system prompt (task + inherited + research) ---
    new_prompt = await knowledge.rewrite_prompt(harness)
    async with SessionLocal() as session:
        h3 = await session.get(HarnessSession, session_id)
        h3.system_prompt = new_prompt
        known_best = [x for x in (h3.best_score, best) if x is not None]
        h3.best_score = max(known_best) if known_best else None
        await session.commit()

    return best, next_hypotheses, new_prompt


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
        # gen 0 = seeds (stored at session creation, step 1); loop runs 1..N.
        # Hypotheses and the rewritten prompt flow from each generation to the next.
        hypotheses: list[str] = []
        memory = ""
        for generation in range(start, start + (generations or s.harness_max_generations)):
            score, hypotheses, memory = await run_generation(
                session_id, generation, hypotheses, memory
            )
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
