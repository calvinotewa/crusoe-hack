---
name: crusoe-hack-dev
description: Conventions, layout and commands for this repo (FastAPI + Crusoe + Modal + Neon eval service). Load before changing code - endpoints, models/migrations, services, settings, tests - or debugging how integrations are wired here.
---

# crusoe-hack development

## Commands
```bash
uv sync && cp .env.example .env
uv run alembic upgrade head                     # direct (non-pooler) Neon URL, on a branch
uv run fastapi dev app/main.py                  # /docs; smoke: /health/deps, /playground/*
uv run ruff check --fix . && uv run ruff format . && uv run pytest     # required before finishing
uv run alembic revision --autogenerate -m "…"   # after editing models; review the file
RUN_INTEGRATION=1 DATABASE_URL=… uv run pytest  # real (branch) Postgres, faked providers
```

## Layout
- `app/config.py`: `Settings` (pydantic-settings, reads `.env` then `.env.local`). Every new setting also goes in `.env.example`, with a comment saying where the value comes from.
- `app/db.py`: async engine, `SessionLocal`, `Base`, `SessionDep` (use in routes; in background work, `async with SessionLocal()`).
- `app/models.py`: `Dataset`, `Run`, `RunResult`. `app/schemas.py`: Pydantic I/O models plus `Task` (one eval item).
- `app/api/routes/`: thin routers (`health`, `datasets`, `runs`, `playground`), registered in `main.py`.
- `app/services/`: all logic and provider clients.
  - `inference.py`: Crusoe via `AsyncOpenAI`, `max_retries=6`.
  - `storage.py`: Neon bucket via boto3 in `asyncio.to_thread`, path-style + SigV4.
  - `sandbox.py`: Modal.
  - `evaluator.py`: generate → grade → persist, plus the graders.
- Flow:
  1. `POST /datasets` writes `datasets/<id>/tasks.jsonl` to the bucket and adds a row.
  2. `POST /runs` adds a row and starts a BackgroundTask (`execute_run`).
  3. The run writes `run_results` rows and `runs/<id>/trace.jsonl`.
- Postgres holds what you query; the bucket holds blobs. Rows store keys; presign the URL on read.

## Rules
- Async everywhere; logic in services, not routes. Match the surrounding style.
- Schema changes go through Alembic only (no `create_all`). Autogenerate misses enum changes and turns renames into drop+add, so review every generated file.
- Tests never hit Crusoe, Modal or the bucket: `monkeypatch.setattr(<module>, "fn", fake)` on the `app.services` module. With `ASGITransport`, BackgroundTasks finish before the response returns. Call `await engine.dispose()` at the end of DB tests.
- Default models are `deepseek-ai/Deepseek-V4-Flash` (generation) and `openai/gpt-oss-120b` (judge). Model facts come from the `crusoe-inference` skill.
- `evaluator.MAX_CONCURRENCY=8` applies per run, and Crusoe limits apply per model (30 RPM on free accounts). Concurrent runs hit 429.
- BackgroundTasks aren't durable: a restart leaves runs stuck in `running`. For long jobs, move to a Modal Function or a queue.

## Recipes
- **Endpoint:** add the route plus schemas (`ConfigDict(from_attributes=True)` for ORM output). Use `HTTPException(404, …)` for missing objects.
- **Grader:** extend the `Task.grader` Literal, add a branch to `evaluator.grade()` returning `{score, feedback, sandbox_exit_code}`, and add a unit test. See `llm-eval-loop`.
- **Bucket helper:** add it to `storage.py` only, wrapped in `to_thread`.

## Wiring gotchas
| Symptom | Fix |
| --- | --- |
| `invalid connection option "sslmode"` | Bypassed `Settings.async_database_url`. Use `app.db.engine` |
| `prepared statement … already exists` | `-pooler` host. Use direct, or disable asyncpg statement caches |
| S3 signature/DNS errors, `NoSuchBucket` | Use `storage._client()`. Buckets are per Neon branch |
| Crusoe 401/404/429, empty output | See `crusoe-inference` (key quoting, ID casing, limits, `max_tokens`) |
| Modal `AuthError` | `modal token new`, or both `MODAL_TOKEN_*` set |
