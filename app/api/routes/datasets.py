import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.db import SessionDep
from app.models import Dataset
from app.schemas import DatasetCreate, DatasetOut, Task
from app.services import storage

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("", response_model=DatasetOut, status_code=201)
async def create_dataset(body: DatasetCreate, session: SessionDep):
    dataset_id = uuid.uuid4()
    key = await storage.put_jsonl(
        f"datasets/{dataset_id}/tasks.jsonl", [t.model_dump() for t in body.tasks]
    )
    dataset = Dataset(
        id=dataset_id,
        name=body.name,
        description=body.description,
        s3_key=key,
        num_tasks=len(body.tasks),
    )
    session.add(dataset)
    await session.commit()
    return dataset


@router.get("", response_model=list[DatasetOut])
async def list_datasets(session: SessionDep):
    return (await session.scalars(select(Dataset).order_by(Dataset.created_at.desc()))).all()


@router.get("/{dataset_id}/tasks", response_model=list[Task])
async def get_tasks(dataset_id: uuid.UUID, session: SessionDep):
    dataset = await session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(404, "dataset not found")
    return await storage.get_jsonl(dataset.s3_key)
