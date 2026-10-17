# crusoe-hack

## Tech stack

| Layer | Tool |
| --- | --- |
| API | [FastAPI](https://fastapi.tiangolo.com) (Python 3.12+, uv) |
| Database | [Neon Postgres](https://neon.com/docs) via SQLAlchemy async + asyncpg, Alembic migrations |
| Object storage | [Neon Object Storage](https://neon.com/docs/storage/overview) (S3-compatible, boto3) |
| LLM inference | [Crusoe Intelligence Foundry](https://console.crusoecloud.com/foundry) (OpenAI-compatible API) |
| Code sandboxes | [Modal Sandboxes](https://modal.com/docs/guide/sandboxes) |
| Web search (harness) | [you.com](https://you.com) API, for the between-generations research step |

## Orchestration harness (`POST /sessions`)

A generational loop that tries to **beat a benchmark with formulae** rather than prompting
alone. Per generation (see `app/services/orchestrator.py`):

1. **Ingest** (session creation): validation CSV (needs a `y` column) + optional seed
   formulae go to the bucket + Postgres; seeds become generation 0, the baseline to beat.
2. **runSubAgent × N**: worker agents on Crusoe propose alternative formulae
   (`predict(row) -> float`), stored with lineage via `store_formulae()`.
3. **Benchmark**: every formula runs against the validation CSV inside Modal sandboxes
   (`block_network=True`, hard timeout); RMSE / MAE / R2 / score land in `benchmarks`.
4. **Summarise**: the best 3 agents' *session thinking* (rationale) is summarised and
   consolidated by a judge model.
5. **Research**: "if the agents knew X…" — a gap query synthesised from the summary, then
   `research_num_sources` (default 20) web sources fetched via you.com and stored.
6. **Rewrite prompt**: the next generation's system prompt = general human task +
   consolidated inherited learnings + research learnings; `GET /sessions/{id}/findings`
   is the RAG read tool workers can call between runs.
7. **Iterate**: `POST /sessions/{id}/iterate {"generations": n}` runs more generations
   with the rewritten prompt; each generation tries to beat `best_score`.

API: `POST /sessions`, `GET /sessions[/{id}]`, `POST /sessions/{id}/iterate`,
`GET /sessions/{id}/findings?q=…`. Tables: `harness_sessions`, `formulae`, `benchmarks`,
`research_sources` (migration `b1a2c3d4e5f6`).
