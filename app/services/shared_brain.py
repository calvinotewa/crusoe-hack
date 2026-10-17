"""Shared Brain: the generational orchestration loop from the README, as one MVP flow.

Per generation (the user's spec):
1. Run all experiments.
2. Summarize the thinking of the top 10 experiments (curator).
3. Supplement the summary's learnings with an internet search on what could improve it.
4. Generate 100 hypotheses on improvement ideas (curator).
5. Have 100 agents produce experiments based on the hypotheses (repeat).

This module is in-memory (no Postgres / bucket) so the MVP script can run standalone;
the API path (`app/services/orchestrator.py`) persists the same flow. Experiment code
runs through `run_experiment`, which shells out to `python -c` with a timeout — fine
for a local MVP on trusted hardware; production uses Modal Sandboxes (sandbox.run_python).
"""

import asyncio
import json
import logging
import math
import random
import re

import httpx

from app.config import get_settings
from app.services import inference
from app.services.benchmarker import build_program

log = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """You are the curator of a shared brain for a team of ML research agents.

Goal: {goal}

Below are the top {top_k} experiments of this generation, each with the researcher's
own thinking. Summarize what the team was thinking: what approaches were tried, which
worked, which didn't, and what the current best result is. Then write a short
"learnings" section: concrete, actionable lessons for the next generation.

{blobs}"""

SEARCH_QUERY_PROMPT = """Here is a summary of what a team of ML research agents tried and learned:

{summary}

Write ONE web-search query for what these agents were missing — "if they knew X they
could have found a better result". Reply with ONLY the query, no quotes."""

HYPOTHESES_PROMPT = """You are the curator of a shared brain for ML research agents.

Goal: {goal}

Summary of the last generation:
{summary}

Supplementary web learnings:
{research}

Propose {batch} NEW, SPECIFIC, TESTABLE hypotheses for improving the result. Favor
diverse ideas across transforms, interactions, non-linearities and error analysis.
Reply with ONLY a numbered list, one hypothesis per line, no prose."""

AGENT_PROMPT = """You are researcher agent #{n} in a Shared Brain team.

Goal: {goal}

Validation data (you are scored on this data):
{stats}
Columns (predict `y` from the others): {columns}

Your assigned hypothesis for this experiment:
{hypothesis}

Current best score (1/(1+rmse), higher is better): {best}
Inherited learnings:
{memory}

Your experiment must be a CALIBRATED model as Python. fit() sees only the training
rows; your score is measured on rows it has not seen:

```python
def fit(rows: list[dict]) -> object:
    # Derive your model's parameters from the training rows. PREFER closed-form least
    # squares: choose basis functions, build the design matrix, solve the
    # normal equations with Gaussian elimination (pure stdlib math only).
    ...

def predict(row: dict, params) -> float:
    # Predict y for one row using the params fit() returned.
    ...
```

Rules:
- FIT the coefficients from the data. NEVER hand-guess constants: previous
  generations failed by guessing weights for a target on this scale.
- Only the standard library (math, statistics). No network, file access, numpy.
- Feature transforms are encouraged (log10 of skewed magnitudes, interactions,
  squares, hinges). Fit coefficients for your chosen basis, don't tune by hand.
- predict() must return a float; it must not raise.

Reply with ONLY this JSON object:
{{
  "name": "<short name>",
  "expression": "<the model in plain maths>",
  "description": "<one or two sentences>",
  "code": "<the fit() and predict() functions as a python code string>",
  "rationale": "<your thinking: how this tests the hypothesis>"
}}"""


def parse_numbered_list(text: str) -> list[str]:
    """Pull the lines out of a numbered-list completion ("1. ...\\n2. ...")."""
    out = []
    for line in text.strip().splitlines():
        m = re.match(r"\s*\d+[.)]\s*(.+)", line.strip())
        if m and len(m.group(1)) > 8:
            out.append(m.group(1).strip())
    return out


def repair_code(code: str) -> str:
    """Repair agent-generated code before exec.

    Agents often return `code` as a JSON string whose newlines are double-escaped
    (literal `\\n` backslash-n). `exec` then fails with "unexpected character after
    line continuation character" and every experiment scores 0 — seen live when
    34/36 experiments died. Strategy: try to compile as-is; if that fails but the
    unescaped version compiles, use it.
    """
    for candidate in (code, code.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')):
        try:
            compile(candidate, "formula.py", "exec")
            return candidate
        except SyntaxError:
            continue
    return code


NAIVE_CODE = (
    "def fit(rows):\n"
    "    return sum(r['y'] for r in rows) / len(rows)\n\n"
    "def predict(row, params):\n"
    "    return params\n"
)


LINEAR_BASELINE_TEMPLATE = """
def feats(r):
    return [1.0] + [r[c] for c in COLUMNS]

def fit(rows):
    X = [feats(r) for r in rows]
    y = [r["y"] for r in rows]
    k = len(X[0])
    A = [[sum(X[i][a] * X[i][b] for i in range(len(X))) for b in range(k)] for a in range(k)]
    for i in range(k):
        A[i][i] += 1e-9
    v = [sum(X[i][a] * y[i] for i in range(len(X))) for a in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            f = A[j][i] / A[i][i]
            for m in range(i, k):
                A[j][m] -= f * A[i][m]
            v[j] -= f * v[i]
    w = [0.0] * k
    for i in reversed(range(k)):
        w[i] = (v[i] - sum(A[i][m] * w[m] for m in range(i + 1, k))) / A[i][i]
    return w

def predict(row, w):
    return sum(a * b for a, b in zip(w, feats(row)))
"""


def linear_baseline_code(columns: list[str]) -> str:
    """Ordinary least squares on the raw numeric columns: the benchmark a formula must beat."""
    return f"COLUMNS = {list(columns)!r}\n" + LINEAR_BASELINE_TEMPLATE


def split_rows(
    rows: list[dict], seed: int = 42, fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)
) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministic train / validation / test split. Agents never see test rows."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * fractions[0])
    n_val = int(len(shuffled) * fractions[1])
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


async def run_experiment(
    code: str, rows: list[dict], timeout: float = 20.0, train: list[dict] | tuple = ()
) -> dict:
    """Run one experiment: fit on `train`, predict and score `rows`, in a subprocess.

    Returns {metrics..., exit_code, stderr_tail}. A crashing/slow experiment scores 0.
    """
    program = build_program(repair_code(code or ""), rows, train)
    proc = await asyncio.create_subprocess_exec(
        "python",
        "-c",
        program,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        return {"exit_code": -1, "stderr_tail": "timeout", "score": 0.0}
    if proc.returncode != 0:
        return {
            "exit_code": proc.returncode,
            "stderr_tail": stderr.decode()[-500:],
            "score": 0.0,
        }
    try:
        metrics = json.loads(stdout.decode().strip().splitlines()[-1])
        metrics["exit_code"] = 0
        metrics["stderr_tail"] = None
        return metrics
    except (ValueError, IndexError):
        return {"exit_code": 0, "stderr_tail": "unparseable output", "score": 0.0}


async def verify_experiment(
    code: str, rows: list[dict], train: list[dict] | tuple = (), resamples: int = 3, seed: int = 0
) -> list[float]:
    """Re-run an experiment on bootstrap resamples of the validation rows.

    Formulas are deterministic, so the curator's replicate check varies the data, not the seed.
    A crashed or timed-out resample counts as an infinite loss.
    """
    rng = random.Random(seed)
    losses = []
    for _ in range(resamples):
        sample = [rng.choice(rows) for _ in rows]
        res = await run_experiment(code, sample, train=train)
        rmse = res.get("rmse")
        losses.append(float(rmse) if rmse is not None and math.isfinite(rmse) else math.inf)
    return losses


async def web_search(query: str, n: int = 5) -> list[dict]:
    """Search the web for `query`. you.com if YOU_API_KEY is set, else DuckDuckGo's
    keyless HTML endpoint. Returns [{title, url, snippet}]; empty list on failure."""
    s = get_settings()
    if s.you_api_key:
        try:
            return await _youcom_search(s.you_api_key, s.you_api_base, query, n)
        except Exception:
            log.warning("you.com search failed for %r; falling back to DDG", query, exc_info=True)
    try:
        return await _ddg_search(query, n)
    except Exception:
        log.warning("web search failed for %r", query, exc_info=True)
        return []


async def _youcom_search(api_key: str, base: str, query: str, n: int) -> list[dict]:
    """you.com Web Search API: POST {base}/v1/search (see you.com/docs/guides/search)."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{base.rstrip('/')}/v1/search",
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            json={"query": query, "count": n},
        )
        resp.raise_for_status()
        web = resp.json().get("results", {}).get("web", [])
    return [
        {
            "title": h.get("title", ""),
            "url": h.get("url", ""),
            "snippet": (h.get("snippets") or [h.get("description") or ""])[0],
        }
        for h in web[:n]
    ]


async def _ddg_search(query: str, n: int) -> list[dict]:
    """DuckDuckGo HTML (keyless). Best-effort parse; the layout is stable enough for an MVP."""
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        resp = await client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
    results = []
    for m in re.finditer(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', resp.text, re.DOTALL
    ):
        url = m.group(1)
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        results.append({"title": title, "url": url, "snippet": None})
        if len(results) >= n:
            break
    return results


async def summarize_top(goal: str, top: list[dict], top_k: int) -> str:
    """Step 2: the curator summarizes the thinking of the top experiments."""
    blobs = "\n\n".join(
        f"### {e['name']} (score {e.get('score')})\n"
        f"Expression: {e.get('expression')}\n"
        f"Thinking: {e.get('rationale') or '(none)'}\n"
        f"Diagnostics: {e.get('stderr_tail') or 'clean'}"
        for e in top[:top_k]
    )
    return await _curate(SUMMARIZE_PROMPT.format(goal=goal, top_k=top_k, blobs=blobs))


async def _curate(prompt: str) -> str:
    s = get_settings()
    content, _ = await inference.chat(
        [{"role": "user", "content": prompt}], model=s.crusoe_curator_model, temperature=0.2
    )
    return content


async def search_gap_query(summary: str) -> str:
    """Step 3a: what should we search to fill the team's knowledge gaps?

    Empty/unparseable curator answers degrade to a deterministic query built
    from the summary's first words instead of raising."""
    content = await _curate(SEARCH_QUERY_PROMPT.format(summary=summary[:4000]))
    lines = [ln.strip().strip('"') for ln in content.strip().splitlines() if ln.strip()]
    if lines:
        return lines[0][:300]
    words = summary.strip().split()
    return " ".join(words[:8])[:300] or "improve the current best result"


async def generate_hypotheses(
    goal: str,
    summary: str,
    research: str,
    n: int,
    batch: int = 10,
    concurrency: int = 4,
    rounds_cap: int = 10,
) -> list[str]:
    """Generate EXACTLY `n` unique hypotheses (fixed count, same every generation).

    The count is a run-level variable (default 20, see HARNESS_NUM_HYPOTHESES) —
    not something the curator gets to decide. Batches of `batch` in parallel,
    de-duplicated, keeps asking until the count is reached (capped at
    `rounds_cap` rounds); pads with explicit open-question slots if the curator
    stalls, so the count is always exact.
    """
    sem = asyncio.Semaphore(concurrency)
    out: list[str] = []
    seen: set[str] = set()

    async def ask() -> list[str]:
        async with sem:
            try:
                text = await _curate(
                    HYPOTHESES_PROMPT.format(
                        goal=goal, summary=summary[:4000], research=research[:4000], batch=batch
                    )
                )
                return parse_numbered_list(text)
            except Exception:
                # one failed batch (e.g. all retries exhausted) must not kill the
                # phase; the count is made exact by the padding below.
                log.warning("hypothesis batch failed; continuing", exc_info=True)
                return []

    for _ in range(rounds_cap):
        if len(out) >= n:
            break
        need = n - len(out)
        num_asks = max(1, math.ceil(need / batch))
        for lst in await asyncio.gather(*[ask() for _ in range(num_asks)]):
            for h in lst:
                if h not in seen:
                    seen.add(h)
                    out.append(h)
    if len(out) < n:  # keep the count exact even if the curator repeats itself
        while len(out) < n:
            slot = len(out) + 1
            out.append(
                f"Open question: what else could beat the current best result? (slot {slot})"
            )
    return out[:n]


def _parse_agent_json(content: str) -> dict | None:
    """Parse the agent's JSON object, tolerating truncation.

    Reasoning models sometimes run out of tokens mid-JSON (no closing brace).
    First try the normal outermost-braces slice; then try closing the
    truncation: cut at the last complete value and close the open strings and
    braces. Returns None if nothing parses."""
    try:
        start, end = content.index("{"), content.rindex("}")
        parsed = json.loads(content[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, IndexError):
        pass
    start = content.find("{")
    if start == -1:
        return None
    frag = content[start:].rstrip()
    if not frag:
        return None
    # try progressively: as-is, close an open string, then close open braces
    n_open = frag.count("{") - frag.count("}")
    in_string = frag.count('"') % 2 == 1
    candidates = [frag]
    if in_string:
        candidates.append(frag + '"')
        if n_open >= 1:
            candidates.append(frag + '"' + "}" * n_open)
    elif n_open >= 1:
        candidates.append(frag + "}" * n_open)
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            continue
    return None


async def agent_experiment(
    n: int,
    goal: str,
    columns: list[str],
    hypothesis: str,
    memory: str,
    best: float | None,
    stats: str = "",
) -> dict:
    """Step 5: one researcher agent produces one experiment testing its hypothesis.

    Never raises. LLM errors degrade to a scored-0 error stub; unparseable JSON
    gets ONE repair round (a repair prompt with a bigger budget — truncated
    completions from reasoning models are the common failure) before falling
    back to the stub. One dead agent never fails the generation."""
    s = get_settings()
    prompt = AGENT_PROMPT.format(
        n=n,
        goal=goal,
        columns=columns,
        stats=stats or "(not provided)",
        hypothesis=hypothesis or "free choice — pick the most promising idea",
        best=best if best is not None else "unknown",
        memory=memory or "(nothing yet)",
    )
    try:
        content, _ = await inference.chat(
            [{"role": "user", "content": prompt}],
            model=s.crusoe_researcher_model,
            temperature=0.7,
            max_tokens=8192,  # reasoning models think first; don't truncate the JSON
        )
    except Exception as e:
        log.warning("agent %d LLM call failed; recording error stub", n, exc_info=True)
        return {
            "name": f"agent-{n}-error",
            "code": None,
            "rationale": f"llm error: {e}",
            "expression": "",
            "description": None,
        }
    parsed = _parse_agent_json(content)
    needs_repair = parsed is None or not parsed.get("code")
    if needs_repair and content.strip():
        # one repair round: show the broken output, ask for ONLY the JSON
        log.info("agent %d: unparseable/incomplete JSON (len %d); repairing", n, len(content))
        try:
            content, _ = await inference.chat(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                    {
                        "role": "assistant",
                        "content": content[:6000],
                    },
                    {
                        "role": "user",
                        "content": "That reply was truncated or not valid JSON. Reply again with "
                        "ONLY the complete JSON object (name, expression, description, code, "
                        "rationale). Keep the code concise so it fits.",
                    },
                ],
                model=s.crusoe_researcher_model,
                temperature=0.2,
                max_tokens=8192,
            )
            parsed = _parse_agent_json(content)
        except Exception:
            log.warning("agent %d: repair round failed", n, exc_info=True)
    if parsed is None or not parsed.get("code"):
        # still unparseable or unusable after repair: a scored-0 stub
        return {
            "name": f"agent-{n}-unparsed",
            "code": None,
            "rationale": content[:400],
            "expression": "",
            "description": None,
        }
    parsed.setdefault("name", f"agent-{n}")
    return parsed
