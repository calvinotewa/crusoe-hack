"""Harness session endpoints (orchestration harness, steps 1-7).

POST /sessions          step 1: ingest validation CSV + seeds, kick off the loop
GET  /sessions         list sessions
GET  /sessions/{id}    full detail: formulae, benchmarks, research sources
POST /sessions/{id}/iterate    run more generations with the current prompt
GET  /sessions/{id}/findings   the RAG read tool for worker agents
"""

import csv
import io
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlalchemy import select

from app.db import SessionDep
from app.models import Benchmark, Formula, HarnessSession, ResearchSource
from app.schemas import (
    HarnessIterate,
    HarnessSessionCreate,
    HarnessSessionDetail,
    HarnessSessionOut,
)
from app.services import knowledge, orchestrator, storage

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _parse_validation_csv(csv_text: str) -> list[dict]:
    """Step 1 input contract: header row + a `y` column (the prediction target)."""
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    if not rows or "y" not in rows[0]:
        raise HTTPException(422, "validation_csv must have a header row and a `y` column")
    return rows


@router.post("", response_model=HarnessSessionOut, status_code=202)
async def create_session(
    body: HarnessSessionCreate, background: BackgroundTasks, session: SessionDep
):
    # Step 1: store the validation CSV in the bucket (multi-tenant: one prefix per session)
    rows = _parse_validation_csv(body.validation_csv)
    session_id = uuid.uuid4()
    key = await storage.put_bytes(
        f"sessions/{session_id}/validation.csv", body.validation_csv.encode(), "text/csv"
    )
    harness = HarnessSession(
        id=session_id,
        name=body.name,
        goal=body.goal,
        validation_s3_key=key,
        num_validation_rows=len(rows),
    )
    session.add(harness)
    await session.commit()

    # Seed formulae become generation 0 (they set the baseline benchmark to beat).
    if body.seed_formulae:
        from app.services.formulae import store_formulae

        await store_formulae(
            session_id,
            [f.model_dump() for f in body.seed_formulae],
            generation=0,
            source="seed",
        )
    await session.refresh(harness)
    # Steps 2..7 loop as a BackgroundTask (same scaffold tradeoff as POST /runs).
    background.add_task(orchestrator.execute_session, harness.id)
    return harness


@router.get("", response_model=list[HarnessSessionOut])
async def list_sessions(session: SessionDep):
    stmt = select(HarnessSession).order_by(HarnessSession.created_at.desc())
    return (await session.scalars(stmt)).all()


@router.get("/{session_id}", response_model=HarnessSessionDetail)
async def get_session(session_id: uuid.UUID, session: SessionDep):
    h = await session.get(HarnessSession, session_id)
    if not h:
        raise HTTPException(404, "session not found")
    formulae = list(
        (await session.scalars(select(Formula).where(Formula.session_id == session_id))).all()
    )
    benchmarks = list(
        (
            await session.scalars(
                select(Benchmark).join(Formula).where(Formula.session_id == session_id)
            )
        ).all()
    )
    sources = list(
        (
            await session.scalars(
                select(ResearchSource).where(ResearchSource.session_id == session_id)
            )
        ).all()
    )
    return {
        **{c: getattr(h, c) for c in HarnessSessionOut.model_fields},
        "formulae": formulae,
        "benchmarks": benchmarks,
        "sources": sources,
    }


@router.post("/{session_id}/iterate", response_model=HarnessSessionOut, status_code=202)
async def iterate_session(
    session_id: uuid.UUID,
    background: BackgroundTasks,
    session: SessionDep,
    body: HarnessIterate | None = None,
):
    """Step 7 (again): more generations with the current rewritten system prompt."""
    h = await session.get(HarnessSession, session_id)
    if not h:
        raise HTTPException(404, "session not found")
    generations = body.generations if body else 1
    background.add_task(orchestrator.execute_session, h.id, generations)
    await session.refresh(h)
    return h


@router.get("/{session_id}/findings")
async def get_findings(session_id: uuid.UUID, q: str, session: SessionDep, limit: int = 5):
    """The RAG read tool (step 6): findings ranked against `q`, for worker agents."""
    if not await session.get(HarnessSession, session_id):
        raise HTTPException(404, "session not found")
    return await knowledge.read_findings(session_id, q, limit)
