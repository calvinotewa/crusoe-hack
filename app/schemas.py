import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models import HarnessStatus, RunStatus


class Task(BaseModel):
    """A single eval item. One line of a dataset's JSONL file.

    grader:
      - "exact":   output.strip() == expected.strip()
      - "code":    output is Python code; `test` is appended and run in a Modal sandbox
      - "judge":   an LLM on Crusoe scores the output against `expected` / `rubric`
    """

    id: str
    prompt: str
    grader: Literal["exact", "code", "judge"] = "judge"
    expected: str | None = None
    test: str | None = None
    rubric: str | None = None


class DatasetCreate(BaseModel):
    name: str
    description: str | None = None
    tasks: list[Task]


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    s3_key: str
    num_tasks: int
    created_at: datetime


class RunCreate(BaseModel):
    dataset_id: uuid.UUID
    model: str | None = None
    system_prompt: str | None = None
    params: dict[str, Any] = Field(default_factory=lambda: {"temperature": 0.2, "max_tokens": 4096})


class RunResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    output: str | None
    score: float | None
    grader: str
    feedback: str | None
    latency_ms: int | None
    sandbox_exit_code: int | None


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    dataset_id: uuid.UUID
    model: str
    system_prompt: str | None
    params: dict[str, Any]
    status: RunStatus
    score: float | None
    error: str | None
    trace_s3_key: str | None
    created_at: datetime
    finished_at: datetime | None


class RunDetail(RunOut):
    results: list[RunResultOut]


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096


class ChatResponse(BaseModel):
    model: str
    content: str
    latency_ms: int


class SandboxRequest(BaseModel):
    code: str
    timeout: int | None = None


class SandboxResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int


# --- Orchestration harness ---


class FormulaIn(BaseModel):
    """A seed formula (step 1). `code` must define `predict(row: dict) -> float`."""

    name: str
    expression: str
    description: str | None = None
    code: str | None = None


class HarnessSessionCreate(BaseModel):
    """Step 1: ingest a validation CSV (must have a `y` column) and optional seed formulae."""

    name: str
    goal: str
    validation_csv: str
    seed_formulae: list[FormulaIn] = Field(default_factory=list)


class FormulaOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    generation: int
    name: str
    description: str | None
    expression: str
    code: str | None
    source: str
    parent_id: uuid.UUID | None
    rationale: str | None
    created_at: datetime


class BenchmarkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    formula_id: uuid.UUID
    generation: int
    rmse: float | None
    mae: float | None
    r2: float | None
    score: float | None
    num_rows: int
    sandbox_exit_code: int | None
    error_tail: str | None
    created_at: datetime


class ResearchSourceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    generation: int
    query: str
    title: str
    url: str
    snippet: str | None
    created_at: datetime


class HarnessSessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    goal: str
    system_prompt: str | None
    status: HarnessStatus
    generation: int
    num_validation_rows: int
    best_score: float | None
    summary: str | None
    research_query: str | None
    error: str | None
    created_at: datetime
    finished_at: datetime | None


class HarnessSessionDetail(HarnessSessionOut):
    formulae: list[FormulaOut]
    benchmarks: list[BenchmarkOut]
    sources: list[ResearchSourceOut]


class HarnessIterate(BaseModel):
    """Step 7 again: run more generations with the current (rewritten) system prompt."""

    generations: int = 1


class Finding(BaseModel):
    """One row of the RAG read tool (knowledge.read_findings)."""

    kind: Literal["source", "formula", "session"]
    title: str
    text: str
    score: float
    url: str | None = None
