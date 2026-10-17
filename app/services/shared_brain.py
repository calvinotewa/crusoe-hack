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

Validation data columns (predict `y`): {columns}

Your assigned hypothesis for this experiment:
{hypothesis}

Current best score (1/(1+rmse)): {best}
Inherited learnings:
{memory}

Write ONE experiment that tests your hypothesis as Python. It must define:

```python
def predict(row: dict) -> float:
    ...
```

Use only arithmetic / math / statistics on `row` values; no network, no file access.

Reply with ONLY this JSON object:
{{
  "name": "<short name>",
  "expression": "<the formula in plain maths>",
  "description": "<one or two sentences>",
  "code": "<the predict() function as a python code string>",
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


async def run_experiment(code: str, rows: list[dict], timeout: float = 20.0) -> dict:
    """Run one experiment (a predict() over the validation rows) in a subprocess.

    Returns {metrics..., exit_code, stderr_tail}. A crashing/slow experiment scores 0.
    """
    program = build_program(code, rows)
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


async def web_search(query: str, n: int = 5) -> list[dict]:
    """Search the web for `query`. you.com if YOU_API_KEY is set, else DuckDuckGo's
    keyless HTML endpoint. Returns [{title, url, snippet}]; empty list on failure."""
    s = get_settings()
    if s.you_api_key:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{s.you_api_base.rstrip('/')}/search",
                params={"q": query, "num_results": n, "safesearch": "Off"},
                headers={"X-API-Key": s.you_api_key},
            )
            resp.raise_for_status()
            return [
                {
                    "title": h.get("title", ""),
                    "url": h.get("url", ""),
                    "snippet": (h.get("snippets") or [""])[0],
                }
                for h in resp.json().get("hits", [])[:n]
            ]
    try:
        return await _ddg_search(query, n)
    except Exception:
        log.warning("web search failed for %r", query, exc_info=True)
        return []


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
    """Step 3a: what should we search to fill the team's knowledge gaps?"""
    content = await _curate(SEARCH_QUERY_PROMPT.format(summary=summary[:4000]))
    return content.strip().strip('"').splitlines()[0][:300]


async def generate_hypotheses(
    goal: str, summary: str, research: str, n: int, batch: int = 10, concurrency: int = 4
) -> list[str]:
    """Step 4: generate `n` hypotheses in batches of `batch` (curator model)."""
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int) -> list[str]:
        async with sem:
            text = await _curate(
                HYPOTHESES_PROMPT.format(
                    goal=goal, summary=summary[:4000], research=research[:4000], batch=batch
                )
            )
        return parse_numbered_list(text)

    batches = [one(i) for i in range(0, n, batch)]
    out: list[str] = []
    for lst in await asyncio.gather(*batches):
        out.extend(lst)
    return out[:n]


async def agent_experiment(
    n: int,
    goal: str,
    columns: list[str],
    hypothesis: str,
    memory: str,
    best: float | None,
) -> dict:
    """Step 5: one researcher agent produces one experiment testing its hypothesis."""
    s = get_settings()
    content, _ = await inference.chat(
        [
            {
                "role": "user",
                "content": AGENT_PROMPT.format(
                    n=n,
                    goal=goal,
                    columns=columns,
                    hypothesis=hypothesis or "free choice — pick the most promising idea",
                    best=best if best is not None else "unknown",
                    memory=memory or "(nothing yet)",
                ),
            }
        ],
        model=s.crusoe_researcher_model,
        temperature=0.7,
    )
    try:
        parsed = json.loads(content[content.index("{") : content.rindex("}") + 1])
    except (ValueError, IndexError):
        return {
            "name": f"agent-{n}-unparsed",
            "code": None,
            "rationale": content[:400],
            "expression": "",
            "description": None,
        }
    parsed.setdefault("name", f"agent-{n}")
    return parsed
