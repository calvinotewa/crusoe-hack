import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.db import SessionDep
from app.models import Dataset, Run
from app.schemas import RunCreate, RunDetail, RunOut, RunResultOut
from app.services import evaluator, storage

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", response_model=RunOut, status_code=202)
async def create_run(body: RunCreate, background: BackgroundTasks, session: SessionDep):
    if not await session.get(Dataset, body.dataset_id):
        raise HTTPException(404, "dataset not found")
    run = Run(
        dataset_id=body.dataset_id,
        model=body.model or get_settings().crusoe_default_model,
        system_prompt=body.system_prompt,
        params=body.params,
    )
    session.add(run)
    await session.commit()
    # Fine for a scaffold; move to a real queue (or a Modal Function) for long runs.
    background.add_task(evaluator.execute_run, run.id)
    return run


@router.get("", response_model=list[RunOut])
async def list_runs(session: SessionDep, dataset_id: uuid.UUID | None = None):
    stmt = select(Run).order_by(Run.created_at.desc())
    if dataset_id:
        stmt = stmt.where(Run.dataset_id == dataset_id)
    return (await session.scalars(stmt)).all()


@router.get("/{run_id}", response_model=RunDetail)
async def get_run(run_id: uuid.UUID, session: SessionDep):
    run = await session.get(Run, run_id, options=[selectinload(Run.results)])
    if not run:
        raise HTTPException(404, "run not found")
    return run


@router.get("/{run_id}/failures", response_model=list[RunResultOut])
async def get_failures(run_id: uuid.UUID, threshold: float = 1.0):
    return await evaluator.failures(run_id, threshold)


@router.get("/{run_id}/trace")
async def get_trace_url(run_id: uuid.UUID, session: SessionDep):
    run = await session.get(Run, run_id)
    if not run or not run.trace_s3_key:
        raise HTTPException(404, "trace not available")
    return {"url": await storage.presigned_url(run.trace_s3_key)}
