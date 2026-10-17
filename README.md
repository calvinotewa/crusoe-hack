# Shared Brain

A shared memory layer for a team of ML research agents, built on Crusoe Managed Inference.

Every new agent on a team usually starts from zero and pays again for the same context. In Shared Brain, short-lived researcher agents read a shared memory before they work, and a curator keeps that memory **current, lean and conflict-free**. Each generation starts fresh but smarter, and we measure the gain as validation loss.

## How it works

```mermaid
flowchart TD
    A[Prepare data + measure noise] --> B[Spawn researchers]
    B --> C[Read memory + best train.py]
    C --> D[Run experiments]
    D --> E[Report results]
    E --> F[Curator verifies]
    F --> G[Update memory + best]
    G --> H{Generations left?}
    H -- Yes --> B
    H -- No --> I[Score on test split]
    I --> J[Compare setups]
```

1. Before generation 1, the orchestrator runs the starting `train.py` on 5 seeds to get the baseline loss and the noise level.
2. Each generation, 4 researchers start from the best `train.py` so far plus memory.
3. Each researcher runs experiments to lower val loss, then reports what it learned.
4. The curator re-checks the top claim, settles conflicts and updates memory.
5. At the end, the starting and best `train.py` are scored on the hidden test split.

## Setups

All setups use the same dataset, splits, models, sandbox and total number of runs. Every generation starts from the best `train.py` so far, so **memory is the only difference**.

| Setup | What it shows | What researchers read | Who picks the best `train.py` |
|---|---|---|---|
| One-shot | A single agent, a single edit; this is the floor | Nothing. One run | — |
| No memory | Every agent starts from zero | Nothing | Orchestrator: lowest val loss |
| Raw log | Memory without curation, which bloats and contradicts itself | Every report, appended as-is. Oldest dropped past 100k tokens | Orchestrator: lowest val loss |
| **Shared Brain** | Curated memory: lean and current | Curated memory, 2,000-token cap | Curator check on 3 seeds |

A new best must beat the old one by more than the noise threshold, in every setup.

## Budget

| Item | Value |
|---|---|
| Generations | 5 |
| Researchers per generation | 4 |
| `run_code` calls per researcher | 6 (Shared Brain), 7 (No memory, Raw log) |
| Curator runs per generation | 4: 3 seeds to verify the top claim, 1 for a conflict or stale entry |
| Total runs per generation | 28 in every setup |
| `web_search` per researcher | 2, 5 results each |
| Agent turns per researcher | 30 max |
| One run | 120 s, 2 CPUs, 4 GB RAM, no GPU, no network |
| Seeds | Researcher runs: 0. Curator checks: 1, 2, 3. Test scoring: 1, 2, 3 |

Crashes and timeouts use up a run.

## Data

Each run uses one dataset, split three ways. There are no other datasets in a run.

| Split | Who sees it | Use |
|---|---|---|
| Train | Researchers, curator | Training |
| Val | Researchers, curator (inputs only, never labels) | Scoring during research |
| Test | No agent | Final before-and-after comparison only, run by `test.py` |

- Researchers never see the test split: not its data, its loss, or any score built from it.
- The curator doesn't see it either. It verifies claims on val, because what it learns goes into memory, and researchers read memory.
- Agent code never sees val labels. `train.py` returns predictions and the orchestrator scores them outside the sandbox.

| Step | The sandbox gets | The sandbox never gets |
|---|---|---|
| Research (`run_code`) | Train, val inputs | Val labels, any test data |
| Final scoring (`test.py`) | Train + val (with labels), test inputs | Test labels |

### Data sources

We test on three different kinds of problem: a yes/no prediction, a time-series forecast and a numeric prediction. Each dataset gets its own run and is scored on its own test split. All datasets are free to download without a login.

| Problem | What it predicts | Dataset | Val loss | Split |
|---|---|---|---|---|
| Churn | Will this customer leave? (yes/no) | [OpenML 46371](https://www.openml.org/d/46371) (35k customers) | Log loss | Random 70/15/15, stratified, seed 42 |
| Regression | At what temperature does a material become a superconductor? | [Superconductivity](https://www.openml.org/d/44148) (21k rows, 79 features) | RMSE | Random 70/15/15, seed 42 |
| Inventory | Daily demand per product | [Walmart M5](https://zenodo.org/records/12636070), top 1,000 products by units sold | Pinball loss at 90% | By time: test = last 28 days, val = 28 days before, train = the rest |

Why these:

- **Churn:** clear room to improve (~20%) and stable between runs.
- **Regression:** most room to improve (~47%); 79 features give agents real tuning work.
- **Inventory:** real retail data; the score measures stock levels, not just the forecast. Room to improve not measured yet: `prepare_data.py` measures it, and we drop M5 if it's under 5%.

Order: churn first end-to-end, then superconductivity, then M5 only if there's time.

`scripts/prepare_data.py <name>` downloads a dataset, splits it and uploads to the bucket under `datasets/<name>/`: `train.parquet`, `val_X.parquet`, `val_y.parquet`, `test_X.parquet`, `test_y.parquet` and `meta.json` (target column, loss, split rule).

## Files

| File | Access |
|---|---|
| `train.py` | The only file agents can edit |
| `sandbox/harness.py` | Locked. Runs `train.py` on train, predicts val inputs, returns predictions |
| `sandbox/test.py` | Locked; agents can't see, edit or run it. Only the orchestrator runs it |

### The `train.py` contract

```python
import numpy as np
import pandas as pd


def fit(train: pd.DataFrame, target: str, seed: int) -> object: ...
def predict(model: object, X: pd.DataFrame) -> np.ndarray: ...
```

- Churn returns the chance of leaving (0 to 1). Regression returns the temperature. Inventory returns the 90th-percentile daily demand for each row.
- `test.py` fits on train + val, then predicts the test inputs, for seeds 1, 2 and 3.
- Allowed libraries: numpy, pandas, pyarrow, scikit-learn, lightgbm, xgboost, catboost, torch (CPU). No installs at runtime.

Starting `train.py` per dataset, in `baselines/<name>/train.py`:

| Dataset | Baseline |
|---|---|
| Churn | Logistic regression |
| Regression | Ridge regression |
| Inventory | Per-product 90th percentile of the last 28 days |

## Tools

| Tool | Input | Returns |
|---|---|---|
| `get_memory` | — | Memory as text |
| `run_code` (researcher) | `code` (the whole `train.py`), `note` (one line: what changed) | `run_id`, `status` (ok / error / timeout), `val_loss`, last 2,000 characters of output, `runs_left` |
| `run_code` (curator) | `run_id` to re-check, or `code` | Mean and spread of val loss over seeds 1, 2, 3, and `beats_best` (true if it beats the best by more than the noise threshold). Uses 3 runs |
| `web_search` | `query` | Up to 5 results: title, URL, description |
| `upsert_memory` | `kind` (works / failed / try_next), `claim` (200 characters max), `run_ids`, optional `id` to edit | `id`, tokens used, token cap. Refused if over the cap |
| `remove_memory` | `id`, `reason` | ok |

The memory tools are our own, backed by Neon Postgres. Crusoe has no memory API.

`web_search` uses the [You.com Search API](https://you.com/docs/get-started/quickstart): `POST https://ydc-index.io/v1/search` with header `X-API-Key: $YDC_API_KEY` and body `{"query": "...", "count": 5}`. Results are in `results.web[]` (`title`, `url`, `description`).

## Agents

| Agent | Model | Tools |
|---|---|---|
| **Researcher** (×4) | `nvidia/Nemotron-3-Super-120B-A12B` | `get_memory`, `web_search`, `run_code` |
| **Curator** (×1) | `deepseek-ai/DeepSeek-V4-Pro` | `get_memory`, `run_code` (to re-check), `upsert_memory`, `remove_memory` |
| Waste checker | `openai/gpt-oss-120b` | None |

- Researcher focus areas: features, model type, tuning, regularization and ensembling.
- Researcher settings: temperature 1.0, top_p 0.95, `chat_template_kwargs: {"enable_thinking": true, "force_nonempty_content": true}`, max_tokens 8192.
- Curator settings: `chat_template_kwargs: {"thinking": true}`, max_tokens 8192.
- At start, the orchestrator checks all model IDs against `/v1/models` and stops if one is missing. Nemotron 3 Ultra isn't on Crusoe's serverless list today; if it appears, it can replace the curator model.
- The orchestrator is a plain Python script, not an agent.

### Prompt order

Researchers in a generation share the same start of the prompt, so Crusoe's cache reuses it:

1. Rules and tool list (same for everyone)
2. Dataset card (`meta.json`)
3. Memory
4. Best `train.py` so far
5. This researcher's focus
6. The conversation

### Report

A researcher's final message is its report:

```
Tried: one line per idea
Best run: <run_id>
Learned: what worked or didn't, and why
Next: ideas not yet tried
```

The orchestrator attaches the researcher's run log (note, status and val loss per run) before passing it to the curator. **Ranking uses the run log, not the report.**

## Rules

- **Memory writes:** only the curator writes to memory.
- **`web_search`:** researchers only. Web ideas are hypotheses, not facts. Only results from `run_code` become memories.
- **Budget:** enforced inside `run_code` (time and run limits), so no one wins by training longer.
- **Test split:** no agent can read it or get a score on it, so nothing tuned on it leaks into memory or `train.py`.

## Generation loop

1. **Once per research run:** run the starting `train.py` on seeds 0 to 4. `L0` is the mean. The noise threshold is the larger of 2 × the spread and 0.5% of `L0`.
2. **Each generation:**
   1. Run 4 researchers in parallel from the best `train.py` and the current memory.
   2. Rank them by gain over the generation's starting best, from the run log.
   3. Update memory: the curator (Shared Brain), append reports (Raw log), or nothing (No memory).
   4. Update the best `train.py` if a run beats it by more than the noise threshold.
   5. Score this generation's best on the test split (seeds 1 to 3) for the chart only. Never shown to agents.
   6. Save metrics and a memory snapshot.
3. **At the end:** `test.py` scores the starting and the best `train.py`.

## Curation

Each generation, the 4 researchers are ranked by **gain over the generation's starting best** (not raw val loss). The curator reads every report; rank decides what it does with each one.

| Rank | Curator action |
|---|---|
| **Top 1** | Re-run with `run_code` on 3 seeds. If `beats_best`, add it to memory as a fact; the orchestrator updates the best `train.py` |
| **Middle 2** | Keep only new ideas, as `try_next` |
| **Bottom 1** | Record briefly as `failed` so it isn't retried |

- **Losers are kept on purpose:** failed runs are the "what failed" part of memory.
- **Noise:** gaps smaller than the noise threshold count as ties. Only claims that clear it become facts.
- **Conflicts:** if the top claim contradicts an entry, or an entry is stale, the curator's 4th run settles it.
- **Budget:** re-running is the curator's expensive step, so ranking decides where it's spent.

## Memory

The store holds four kinds of entry:

- **works:** findings, with proof.
- **failed:** dead ends not to retry.
- **try_next:** open questions.
- **best:** the top config and score (set by the orchestrator only).

Each entry has: `id`, `kind`, `claim`, `run_ids`, val change, seeds, `tested_on` (best version), created and updated generation.

Rules for the memory:

- **Lean:** 2,000-token cap (counted as characters ÷ 4). If still over the cap after curation, drop the oldest `try_next`, then the oldest `failed`. Never drop `best`.
- **Current:** an entry tested on a best that is 2 or more versions old is stale. The curator re-tests or removes it.
- **Conflict-free:** only the curator writes to it, and contradictions are settled by an experiment.
- **Versioned:** every change is logged, and memory is saved every generation.
- **Stable text:** memory is rendered in a fixed order (best, works, failed, try_next, then by `id`) so the cache can reuse it.

### Why Crusoe MemoryAlloy

Every researcher's prompt starts with the same memory (see [Prompt order](#prompt-order)), so Crusoe's [MemoryAlloy](https://crusoe.ai/resources/blog/crusoe-memoryalloy-reinventing-kv-caching-for-cluster-scale-inference) reads it once and reuses that work for the other researchers. Sharing memory stays cheap and fast: each researcher pays less and starts answering sooner.

## Scoring

```
gain    = (L0 − Lbest) / L0
fitness = average gain across seeds 1, 2, 3
```

- **`L0`:** loss of the starting `train.py`.
- **`Lbest`:** loss of the best `train.py` reached within the budget.
- During research, both are val loss. For the final score, `test.py` measures both on the test split, with the best `train.py` picked by val loss.

## Evaluation

All setups use the same dataset and budget, and are scored on the same test split. We track four metrics per generation:

- **Quality:** test gain of that generation's best `train.py`.
- **Waste:** runs that repeat a change that already failed earlier in the same research run. `openai/gpt-oss-120b` checks each change against the full run log, with the same rule in every setup.
- **Memory size:** tokens of what researchers read. Curated memory should stay flat while the raw log grows.
- **Cost:** prompt, cached and output tokens, dollars, and time-to-first-token per agent (calls are streamed to measure it; see [Why Crusoe MemoryAlloy](#why-crusoe-memoryalloy)).

Expected size: about 170 sandbox runs and about 2 hours per research run, or about 1,500 runs for all 3 datasets × 4 setups. Run setups in parallel.

## Storage

| Postgres table | Holds |
|---|---|
| `research_runs` | Dataset, setup, config, status, `L0`, noise threshold, fitness |
| `generations` | Index, start and end best val loss, best version, test gain, memory tokens |
| `agent_sessions` | Role, focus, model, report, tokens, cost |
| `experiments` | Seed, status, val loss, runtime, note, output tail, waste flag, code key |
| `memory_entries` | The entry fields above, plus an active flag |
| `memory_events` | Every upsert and remove, with the reason |
| `memory_snapshots` | Rendered memory and token count per generation |

Bucket: `datasets/<name>/…`, `research/<run_id>/code/<experiment_id>.py`, `research/<run_id>/transcripts/<session_id>.jsonl`.

The old prompt-grading tables (`datasets`, `runs`, `run_results`) and code (`evaluator.py`, the LLM judge) are removed in the same migration.

## Code layout

```
app/
  orchestrator.py        # CLI: uv run python -m app.orchestrator --dataset churn --setup shared_brain
  loop.py                # generation loop, ranking, best tracking
  agents/                # researcher.py, curator.py, prompts.py, tools.py
  memory.py              # entries, rendering, cap, staleness
  scoring.py             # losses, noise threshold, gain, fitness
  services/              # inference.py, sandbox.py, storage.py, search.py
  api/routes/            # research.py, dashboard.py, health.py, playground.py
sandbox/                 # harness.py, test.py (locked, copied into the sandbox)
baselines/<dataset>/train.py
scripts/prepare_data.py
```

## API and dashboard

The orchestrator writes to Postgres. The API only reads.

| Endpoint | Returns |
|---|---|
| `GET /` | Dashboard: test gain per generation for each setup, memory size, cost (one HTML page, Chart.js from a CDN) |
| `GET /research-runs` | All research runs |
| `GET /research-runs/{id}` | Generations and metrics |
| `GET /research-runs/{id}/memory?generation=n` | Memory snapshot |
| `GET /health`, `GET /playground/models` | Unchanged |

## Settings

New settings go in `app/config.py` and `.env.example`:

| Setting | Default |
|---|---|
| `YDC_API_KEY` | — |
| `RESEARCHER_MODEL` | `nvidia/Nemotron-3-Super-120B-A12B` |
| `CURATOR_MODEL` | `deepseek-ai/DeepSeek-V4-Pro` |
| `CRUSOE_JUDGE_MODEL` (exists, used for the waste check) | `openai/gpt-oss-120b` |
| `SANDBOX_TIMEOUT_SECONDS` (exists) | `120` |
| `SANDBOX_CPU` | `2` |
| `SANDBOX_MEMORY_MB` | `4096` |

Budget numbers live in a `ResearchConfig` model with the defaults above, and can be overridden from the CLI.

## Build order

1. Churn data prep, baseline, `harness.py`, `run_code`: the baseline gets a val score.
2. One researcher, one generation.
3. 4 researchers, ranking, best tracking: the No memory setup works.
4. Curator and memory: the Shared Brain setup works.
5. Raw log and One-shot setups.
6. `test.py`, per-generation test scores, metrics.
7. API and dashboard.
8. Superconductivity, then M5.

## Done when

- `uv run python -m app.orchestrator --dataset churn --setup shared_brain` finishes 5 generations and saves fitness.
- All 4 setups run on churn and show on the dashboard.
- Tests (no network) cover: val and test labels never reach the research sandbox, run limits, memory cap and dropping order, staleness, ranking with ties, equal total runs across setups, scoring math, raw log trimming.
- `uv run ruff check . && uv run ruff format . && uv run pytest` passes.

## Out of scope

GPUs, fine-tuning, more than one dataset per research run, agents editing anything but `train.py`, repeating whole research runs (add only if time allows).

## Models (Crusoe Managed Inference)

| Role | Model |
|---|---|
| Researcher | `nvidia/Nemotron-3-Super-120B-A12B` |
| Curator | `deepseek-ai/DeepSeek-V4-Pro` |
| Waste checker | `openai/gpt-oss-120b` |

Add a payment method to the Crusoe project: it raises the limit from 30 to 600 requests per minute per model.

## Tech stack

| Layer | Tool |
| --- | --- |
| API | [FastAPI](https://fastapi.tiangolo.com) (Python 3.12+, uv) |
| Database | [Neon Postgres](https://neon.com/docs) via SQLAlchemy async + asyncpg, Alembic migrations |
| Object storage | [Neon Object Storage](https://neon.com/docs/storage/overview) (S3-compatible, boto3) |
| LLM inference | [Crusoe Intelligence Foundry](https://console.crusoecloud.com/foundry) (OpenAI-compatible API) |
| Code sandboxes | [Modal Sandboxes](https://modal.com/docs/guide/sandboxes) |
| Web search | [You.com Search API](https://you.com/docs/get-started/quickstart) |

## Setup

```bash
scripts/setup.sh                    # installs tooling, signs in to Modal/Neon/Crusoe, generates a verified .env
uv run python scripts/check_env.py  # re-verify .env anytime (--skip-sandbox for a fast check)
uv run fastapi dev app/main.py
```

Requires Node.js (for the Neon CLI) and access to the team Neon project `patient-art-99467040` (ask the owner to invite your Neon account).

Built at the Crusoe x Cerebral Valley Recursive Agents Hackathon, SF, Oct 2026.
