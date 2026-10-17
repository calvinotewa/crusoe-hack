"""Step 3: run stored formulae against the validation CSV inside Modal sandboxes and
store per-formula benchmarks (RMSE / MAE / R2 / score) in Postgres.

The formula's `predict` code and the validation rows are shipped into the sandbox as a
single `python -c` program (see sandbox.run_python: block_network, hard timeout).
"""

import asyncio
import csv
import io
import json
import logging

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Benchmark, Formula, HarnessSession
from app.services import sandbox, storage

log = logging.getLogger(__name__)

# Formulae within a generation are benchmarked with bounded concurrency (independent,
# untrusted code => sandboxed; Modal scales sandboxes horizontally).
MAX_CONCURRENCY = 8


def _coerce(row: dict) -> dict:
    """Best-effort numeric coercion: '3.14' -> 3.14, '' -> None; leaves other strings."""
    out = {}
    for k, v in row.items():
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                out[k] = None
            else:
                try:
                    out[k] = float(s)
                except ValueError:
                    try:
                        out[k] = int(s)
                    except ValueError:
                        out[k] = v
        else:
            out[k] = v
    return out


def build_program(code: str, rows: list[dict], train: list[dict] | tuple = ()) -> str:
    """Wrap an experiment plus the rows to score in a benchmark program.

    Contract: `code` defines `predict(row)` — or, for calibratable models,
    `fit(rows) -> params` + `predict(row, params)`. `fit` receives only the training split
    (`train`), so the scored rows never influence the fit. `predict` receives features only:
    the target `y` is stripped before the call. A single-arg `predict` falls back automatically.
    """
    return f"""
import json, math, statistics

ns = {{}}
exec({code!r}, ns)

rows = {rows!r}
pred = ns.get("predict")
fit = ns.get("fit")
params = fit({list(train)!r}) if callable(fit) else None

errs, abs_errs, ys, skipped = [], [], [], 0
for row in rows:
    try:
        features = {{k: v for k, v in row.items() if k != "y"}}
        if params is not None:
            try:
                p = pred(features, params)
            except TypeError:  # single-arg predict
                p = pred(features)
        else:
            p = pred(features)
        y = row["y"]
        if p is None or not isinstance(p, (int, float)) or math.isnan(float(p)):
            skipped += 1
            continue
        p = float(p)
        y = float(y)
    except Exception:
        skipped += 1
        continue
    ys.append(y)
    errs.append((p - y) ** 2)
    abs_errs.append(abs(p - y))

n = len(ys)
rmse = math.sqrt(sum(errs) / n) if n else float("nan")
mae = sum(abs_errs) / n if n else float("nan")
if n >= 2 and len(set(ys)) > 1:
    var = sum((y - statistics.mean(ys)) ** 2 for y in ys) / n
    ss_res = sum(errs)
    r2 = 1 - ss_res / (var * n) if var else float("nan")
else:
    r2 = float("nan")

print(json.dumps({{
    "n": n, "skipped": skipped, "rmse": rmse, "mae": mae, "r2": r2,
    "score": 1.0 / (1.0 + rmse) if n else 0.0,
}}))
"""


async def get_validation_rows(session: HarnessSession) -> list[dict]:
    """CSV from the bucket -> list of coerced row dicts (one benchmark input)."""
    raw = (await storage.get_bytes(session.validation_s3_key)).decode()
    return [_coerce(row) for row in csv.DictReader(io.StringIO(raw))]


async def benchmark_formula(formula: Formula, rows: list[dict], timeout: int | None = None) -> dict:
    """Run one formula over the validation rows in a sandbox; return metrics + diagnostics."""
    program = build_program(formula.code or formula.expression, rows)
    res = await sandbox.run_python(program, timeout=timeout)
    metrics: dict = {}
    if res.exit_code == 0:
        try:
            metrics = json.loads(res.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            metrics = {}
    return {
        **{k: metrics.get(k) for k in ("rmse", "mae", "r2", "score")},
        "num_rows": len(rows),
        "sandbox_exit_code": res.exit_code,
        "error_tail": (res.stderr or res.stdout)[-2000:] or None if not metrics else None,
    }


async def store_benchmark(formula_id, generation: int, metrics: dict) -> Benchmark:
    async with SessionLocal() as session:
        row = Benchmark(formula_id=formula_id, generation=generation, **metrics)
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return row


async def benchmark_generation(session_id, generation: int, timeout: int | None = None) -> list:
    """Benchmark every formula of one generation against the session's validation CSV."""
    async with SessionLocal() as session:
        harness = await session.get(HarnessSession, session_id)
        rows = await get_validation_rows(harness)
        stmt = select(Formula).where(
            Formula.session_id == session_id, Formula.generation == generation
        )
        formulae = list((await session.scalars(stmt)).all())
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _one(f: Formula) -> dict:
        async with sem:
            try:
                return {"formula_id": str(f.id), **(await benchmark_formula(f, rows, timeout))}
            except Exception as e:  # a broken formula shouldn't sink the generation
                log.exception("benchmark failed for formula %s", f.id)
                return {
                    "formula_id": str(f.id),
                    "sandbox_exit_code": None,
                    "error_tail": str(e)[:2000],
                }

    return await asyncio.gather(*(_one(f) for f in formulae))
