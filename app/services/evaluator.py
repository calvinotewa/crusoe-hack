"""The improvement loop: generate (Crusoe) -> grade (exact / Modal sandbox / LLM judge) -> persist.

Results go to Postgres for querying, and a full JSONL trace goes to the Neon bucket so it can be
reused later (e.g. as fine-tuning or prompt-optimisation data).
"""

import asyncio
import json
import logging
import re
import uuid
from datetime import UTC, datetime

from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.models import Dataset, Run, RunResult, RunStatus
from app.schemas import Task
from app.services import inference, sandbox, storage

log = logging.getLogger(__name__)

MAX_CONCURRENCY = 8

JUDGE_PROMPT = """You are grading a model's answer.

Task:
{prompt}

Reference answer (may be empty):
{expected}

Rubric (may be empty):
{rubric}

Model answer:
{output}

Respond with JSON only: {{"score": <float 0..1>, "feedback": "<one or two sentences>"}}"""


def extract_code(text: str) -> str:
    """Pull the first fenced code block out of a completion, or return it unchanged."""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


async def grade(task: Task, output: str) -> dict:
    if task.grader == "exact":
        ok = output.strip() == (task.expected or "").strip()
        return {"score": float(ok), "feedback": None, "sandbox_exit_code": None}

    if task.grader == "code":
        program = f"{extract_code(output)}\n\n{task.test or ''}"
        res = await sandbox.run_python(program)
        # Modal: exec timeout -> -1 (no exception); killed by signal -> 128+N (137 = SIGKILL/OOM)
        status = {-1: "timed out", 137: "killed (OOM or terminated)"}.get(res.exit_code)
        output_tail = (res.stderr or res.stdout)[-2000:]
        return {
            "score": 1.0 if res.exit_code == 0 else 0.0,
            "feedback": "\n".join(filter(None, [status, output_tail])) or None,
            "sandbox_exit_code": res.exit_code,
        }

    content, _ = await inference.chat(
        [
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    prompt=task.prompt,
                    expected=task.expected or "",
                    rubric=task.rubric or "",
                    output=output,
                ),
            }
        ],
        model=get_settings().crusoe_judge_model,
        temperature=0.0,
        max_tokens=2048,  # gpt-oss reasons before answering
    )
    try:
        parsed = json.loads(content[content.index("{") : content.rindex("}") + 1])
        return {
            "score": float(parsed["score"]),
            "feedback": parsed.get("feedback"),
            "sandbox_exit_code": None,
        }
    except (ValueError, KeyError):
        return {
            "score": None,
            "feedback": f"unparseable judge output: {content[:500]}",
            "sandbox_exit_code": None,
        }


async def _evaluate_task(run: Run, task: Task, sem: asyncio.Semaphore) -> dict:
    async with sem:
        messages = []
        if run.system_prompt:
            messages.append({"role": "system", "content": run.system_prompt})
        messages.append({"role": "user", "content": task.prompt})
        try:
            output, latency_ms = await inference.chat(messages, model=run.model, **run.params)
            graded = await grade(task, output)
        except Exception as e:  # keep the run going; record the failure on this task
            log.exception("task %s failed", task.id)
            output, latency_ms = None, None
            graded = {"score": 0.0, "feedback": f"error: {e}", "sandbox_exit_code": None}
        return {
            "task_id": task.id,
            "prompt": task.prompt,
            "output": output,
            "grader": task.grader,
            "latency_ms": latency_ms,
            **graded,
        }


async def execute_run(run_id: uuid.UUID) -> None:
    async with SessionLocal() as session:
        run = await session.get(Run, run_id)
        dataset = await session.get(Dataset, run.dataset_id)
        run.status = RunStatus.running
        await session.commit()

        try:
            tasks = [Task(**row) for row in await storage.get_jsonl(dataset.s3_key)]
            sem = asyncio.Semaphore(MAX_CONCURRENCY)
            rows = await asyncio.gather(*(_evaluate_task(run, t, sem) for t in tasks))

            session.add_all(RunResult(run_id=run.id, **r) for r in rows)
            scored = [r["score"] for r in rows if r["score"] is not None]
            run.score = sum(scored) / len(scored) if scored else None
            run.trace_s3_key = await storage.put_jsonl(
                f"runs/{run.id}/trace.jsonl",
                [
                    {
                        "run_id": str(run.id),
                        "model": run.model,
                        "system_prompt": run.system_prompt,
                        **r,
                    }
                    for r in rows
                ],
            )
            run.status = RunStatus.completed
        except Exception as e:
            log.exception("run %s failed", run_id)
            run.status = RunStatus.failed
            run.error = str(e)
        run.finished_at = datetime.now(UTC)
        await session.commit()


async def failures(run_id: uuid.UUID, threshold: float = 1.0) -> list[RunResult]:
    """Results below `threshold` — the raw material for the next improvement iteration."""
    async with SessionLocal() as session:
        stmt = select(RunResult).where(RunResult.run_id == run_id, RunResult.score < threshold)
        return list((await session.scalars(stmt)).all())
