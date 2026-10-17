import enum
import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class RunStatus(enum.StrEnum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class HarnessStatus(enum.StrEnum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class Dataset(Base):
    """A set of eval tasks. The tasks themselves live as JSONL in Neon object storage."""

    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    s3_key: Mapped[str] = mapped_column(String(500))
    num_tasks: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    runs: Mapped[list["Run"]] = relationship(back_populates="dataset")


class Run(Base):
    """One evaluation of a model + prompt configuration against a dataset."""

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id"))
    model: Mapped[str] = mapped_column(String(200))
    system_prompt: Mapped[str | None] = mapped_column(Text)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus), default=RunStatus.pending)
    score: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    trace_s3_key: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    dataset: Mapped[Dataset] = relationship(back_populates="runs")
    results: Mapped[list["RunResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class RunResult(Base):
    """Per-task outcome of a run."""

    __tablename__ = "run_results"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    task_id: Mapped[str] = mapped_column(String(200))
    prompt: Mapped[str] = mapped_column(Text)
    output: Mapped[str | None] = mapped_column(Text)
    score: Mapped[float | None] = mapped_column(Float)
    grader: Mapped[str] = mapped_column(String(50))
    feedback: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    sandbox_exit_code: Mapped[int | None] = mapped_column(Integer)

    run: Mapped[Run] = relationship(back_populates="results")


class HarnessSession(Base):
    """One orchestration-harness loop: formulae -> benchmarks -> summary -> research -> new prompt.

    The validation CSV lives as CSV in Neon object storage; formulae, benchmarks and research
    sources are queryable rows here.
    """

    __tablename__ = "harness_sessions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    goal: Mapped[str] = mapped_column(Text)
    system_prompt: Mapped[str | None] = mapped_column(Text)
    status: Mapped[HarnessStatus] = mapped_column(
        Enum(HarnessStatus), default=HarnessStatus.pending
    )
    generation: Mapped[int] = mapped_column(Integer, default=0)
    validation_s3_key: Mapped[str] = mapped_column(String(500))
    num_validation_rows: Mapped[int] = mapped_column(Integer, default=0)
    best_score: Mapped[float | None] = mapped_column(Float)
    summary: Mapped[str | None] = mapped_column(Text)
    research_query: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    formulae: Mapped[list["Formula"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    sources: Mapped[list["ResearchSource"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Formula(Base):
    """A candidate formula produced by a worker agent (or seeded). Code must define predict(row)."""

    __tablename__ = "formulae"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("harness_sessions.id", ondelete="CASCADE")
    )
    generation: Mapped[int] = mapped_column(Integer, default=0)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    expression: Mapped[str] = mapped_column(Text)
    code: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(50))  # "seed" or "agent"
    parent_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("formulae.id"))
    rationale: Mapped[str | None] = mapped_column(Text)  # the agent's own thinking, kept for step 4
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped[HarnessSession] = relationship(back_populates="formulae")
    benchmarks: Mapped[list["Benchmark"]] = relationship(
        back_populates="formula", cascade="all, delete-orphan"
    )


class Benchmark(Base):
    """One run of a formula against the validation data (in a Modal sandbox)."""

    __tablename__ = "benchmarks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    formula_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("formulae.id", ondelete="CASCADE"))
    generation: Mapped[int] = mapped_column(Integer, default=0)
    rmse: Mapped[float | None] = mapped_column(Float)
    mae: Mapped[float | None] = mapped_column(Float)
    r2: Mapped[float | None] = mapped_column(Float)
    score: Mapped[float | None] = mapped_column(Float)  # 1/(1+rmse), higher is better
    num_rows: Mapped[int] = mapped_column(Integer, default=0)
    sandbox_exit_code: Mapped[int | None] = mapped_column(Integer)
    error_tail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    formula: Mapped[Formula] = relationship(back_populates="benchmarks")


class ResearchSource(Base):
    """A web source fetched between generations ("if the agent knew X ...")."""

    __tablename__ = "research_sources"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("harness_sessions.id", ondelete="CASCADE")
    )
    generation: Mapped[int] = mapped_column(Integer, default=0)
    query: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(String(500))
    url: Mapped[str] = mapped_column(String(500))
    snippet: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped[HarnessSession] = relationship(back_populates="sources")
