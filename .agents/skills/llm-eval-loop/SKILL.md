---
name: llm-eval-loop
description: Method for measuring and improving LLMs in this repo - task/dataset design, graders (exact, code, LLM judge), comparing runs, significance, the automatic improve step, exporting traces as SFT/DPO data.
---

# Eval and improvement loop

Loop: **measure → read failures → change one thing → re-measure → keep it only if it wins.**

## Rules
1. Compare runs only on the same dataset and graders. Change one variable at a time: model, system prompt *or* params (same temperature).
2. There is one dataset with three splits, and no other datasets: `train` for training, `val` for iterating and the improve step, `test` only to confirm the final winner. Test tasks, outputs and scores never go to researchers, the curator, the proposer or training exports.
3. Prefer `exact`/`code` graders. Use `judge` only for open-ended output, and give it a `rubric` or `expected`.
4. Read `/runs/{id}/failures` (the `feedback` field) before changing anything.

## Tasks (`schemas.Task`, one JSONL line)
`{"id","prompt","grader":"exact|code|judge","expected?","test?","rubric?"}`
- IDs are stable slugs, and results are joined across runs by `task_id`. Use a category prefix (`math-…`) to group failures.
- `exact` compares after `strip()`, so the prompt must pin the format.
- `code` runs `extract_code(output)` (first code fence) followed by `test` in a sandbox; exit code 0 = pass. Use `assert`s with edge cases so hardcoded answers fail.
- A `None` score means it couldn't be graded. These are excluded from the average, so count them.

## Graders
To add one: extend the `Literal`, add a branch to `evaluator.grade()` returning `{score, feedback, sandbox_exit_code}`, and add a unit test. Good next ones: `regex`, `json_schema`, `numeric` (with tolerance), `pairwise` (A vs B in random order; more sensitive).
- Judge: temperature 0, JSON output, a different model family from the model under test. Calibrate against 10–20 hand labels before trusting it. Judge calls double the requests per task.

## Is it better?
- With n tasks the standard error ≈ `sqrt(p(1-p)/n)`, so ±13 points at 95% for n=50, p=0.7.
- Compare **paired** runs on the same tasks: count fail→pass (b) vs pass→fail (c) and run a sign test, or bootstrap over `task_id`s. Report latency (`latency_ms`) next to the score.
- No significant win means no win: add tasks or keep the baseline. A per-task diff endpoint (`/runs/compare?a=&b=`) is a good addition.

## Improve step (to build)
1. Get `evaluator.failures(run_id)` on the val split.
2. A strong model (`deepseek-ai/DeepSeek-V4-Pro`) diagnoses the patterns and proposes **one** revised system prompt as JSON.
3. Start a new run with it, and accept it only on a significant paired win. Iterate N times, then confirm once on the test split.
4. Store the lineage (`parent_run_id`, `rationale` columns via migration). Reject candidates that contain verbatim `expected` strings.

## Traces → training data
`runs/<id>/trace.jsonl` holds the prompt, output, score, feedback and model on each line.
- SFT: score==1 rows as `{"messages":[…]}`, written to `exports/sft/`.
- DPO: pass vs fail outputs for the same `task_id`.
- Dedupe by `(task_id, output)`, and exclude test-split tasks.
