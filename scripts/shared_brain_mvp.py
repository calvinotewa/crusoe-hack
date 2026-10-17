#!/usr/bin/env python3
"""Shared Brain MVP — the full 5-step loop, runnable standalone.

    uv run python scripts/shared_brain_mvp.py [validation.csv] [--quick]

Steps per generation (spec):
  1. Run all experiments.
  2. Summarize the thinking of the top 10 experiments (curator model).
  3. Supplement the summary's learnings with a web search on what could improve it.
  4. Generate 100 hypotheses on improvement ideas (curator model).
  5. Have 100 agents produce experiments based on the hypotheses. Repeat.

Only needs CRUSOE_API_KEY (plus optional YOU_API_KEY for better search; falls back to
keyless DuckDuckGo). No Postgres, no bucket, no Modal: everything is in memory, and
experiment code runs in local `python -c` subprocesses with timeouts. For untrusted
multi-tenant runs use the API path (Modal Sandboxes).

Models (README): researcher nvidia/Nemotron-3-Super-120B-A12B, curator nvidia/Nemotron-3-Ultra-550B.
"""

import argparse
import asyncio
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.services import shared_brain  # noqa: E402

log = logging.getLogger("shared_brain")

DEFAULT_GOAL = "Predict `y` from the other columns with the lowest possible RMSE."


def load_rows(path: str | None) -> list[dict]:
    if path:
        with open(path) as f:
            return list(csv.DictReader(f))
    # Synthetic dataset: y = 2a + 3b + 0.5ab + noise. Something to actually beat.
    rng = random.Random(7)
    rows = []
    for _ in range(60):
        a = rng.uniform(-5, 5)
        b = rng.uniform(-5, 5)
        y = 2 * a + 3 * b + 0.5 * a * b + rng.gauss(0, 0.3)
        rows.append({"a": round(a, 4), "b": round(b, 4), "y": round(y, 4)})
    return rows


async def run_generation(
    gen: int,
    rows: list[dict],
    args: argparse.Namespace,
    hypotheses: list[str],
    memory: str,
    best: float | None,
) -> tuple[float | None, str, str, list[str], list[dict]]:
    """One pass of steps 1-5. Returns (best_score, summary, memory, hypotheses, experiments)."""
    train, val, _ = shared_brain.split_rows(rows)
    columns = sorted(set(rows[0]) - {"y"})
    n_agents = args.agents

    # --- Step 5/1 (same call): N agents produce experiments, hypotheses round-robin ---
    sem = asyncio.Semaphore(args.concurrency)

    async def one(n: int) -> dict:
        hyp = hypotheses[(n - 1) % len(hypotheses)] if hypotheses else ""
        async with sem:
            exp = await shared_brain.agent_experiment(n, args.goal, columns, hyp, memory, best)
        exp["agent"] = n
        exp["hypothesis"] = hyp
        return exp

    t0 = time.perf_counter()
    log.info("gen %d: %d agents producing experiments (this is the slow step)...", gen, n_agents)
    experiments = await asyncio.gather(*[one(i + 1) for i in range(n_agents)])

    # --- Step 1 cont.: run all experiments against the validation rows ---
    async def run_one(exp: dict) -> dict:
        async with sem:
            if not exp.get("code"):
                return {**exp, "score": 0.0, "stderr_tail": "no code"}
            return {**exp, **(await shared_brain.run_experiment(exp["code"], val, train=train))}

    log.info("gen %d: running all %d experiments...", gen, len(experiments))
    experiments = list(await asyncio.gather(*[run_one(e) for e in experiments]))
    experiments.sort(key=lambda e: e.get("score") or 0.0, reverse=True)
    gen_best = experiments[0].get("score") if experiments else None
    best = max(filter(None, [best, gen_best])) if (best or gen_best) else None
    log.info("gen %d: best score %.4f (took %.0fs)", gen, gen_best or 0, time.perf_counter() - t0)

    # --- Step 2: summarize the thinking of the top 10 experiments ---
    log.info("gen %d: curator summarizing top %d...", gen, min(args.top_k, len(experiments)))
    try:
        summary = await shared_brain.summarize_top(args.goal, experiments, args.top_k)
    except Exception as e:
        log.warning("summarize failed (%s); using top-experiment table", e)
        summary = (
            "\n".join(
                f"- {e2.get('name')}: score {e2.get('score')} rmse {e2.get('rmse')}"
                for e2 in experiments[: args.top_k]
            )
            or "(no experiments)"
        )

    # --- Step 3: supplement learnings with a web search on what could improve ---
    # Both steps degrade on failure; a down curator/search never kills the run.
    log.info("gen %d: searching the web for knowledge gaps...", gen)
    try:
        query = await shared_brain.search_gap_query(summary)
        sources = await shared_brain.web_search(query, args.sources)
    except Exception as e:
        log.warning("research failed (%s); continuing without sources", e)
        query, sources = "(research unavailable)", []
    learnings = "\n".join(f"- {s['title']} ({s['url']})" for s in sources) or "(no results)"
    log.info("gen %d: query=%r -> %d sources", gen, query, len(sources))

    # --- Step 4: generate N hypotheses for the next generation ---
    log.info("gen %d: curator generating %d hypotheses...", gen, args.hypotheses)
    try:
        hypotheses = await shared_brain.generate_hypotheses(
            args.goal, summary, learnings, args.hypotheses, concurrency=args.concurrency
        )
    except Exception as e:
        # carry this generation's hypotheses over (already the fixed count)
        log.warning("hypotheses failed (%s); carrying previous over", e)
        hypotheses = list(hypotheses)[: args.hypotheses]
        while len(hypotheses) < args.hypotheses:
            hypotheses.append(f"Open question (slot {len(hypotheses) + 1})")
    log.info("gen %d: %d hypotheses ready", gen, len(hypotheses))

    memory = f"## Last generation's summary\n{summary}\n\n## Web learnings\n{learnings}"
    return best, summary, memory, hypotheses, experiments


async def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("csv", nargs="?", help="validation CSV (needs a `y` column); default: synthetic")
    p.add_argument("--goal", default=DEFAULT_GOAL)
    p.add_argument("--generations", type=int, default=2, help="passes of steps 1-5 (default 2)")
    p.add_argument("--agents", type=int, default=100, help="agents per generation (spec: 100)")
    p.add_argument("--top-k", type=int, default=10, help="experiments to summarize (spec: 10)")
    p.add_argument(
        "--hypotheses", type=int, default=20, help="hypotheses per gen (FIXED count, default 20)"
    )  # noqa: E501
    p.add_argument(
        "--sources", type=int, default=5, help="web sources to fetch (spec step: search)"
    )
    p.add_argument(
        "--concurrency", type=int, default=4, help="parallel inference calls (30 RPM free tier!)"
    )
    p.add_argument("--out", default="shared_brain_results.json")
    p.add_argument(
        "--quick", action="store_true", help="small preset: 6 agents, 10 hypotheses, 1 gen"
    )
    args = p.parse_args()
    if args.quick:
        args.agents, args.hypotheses, args.generations, args.top_k = 6, 10, 1, 3

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    s = get_settings()
    if not s.crusoe_api_key:
        sys.exit("Set CRUSOE_API_KEY in .env first")
    log.info(
        "Shared Brain MVP: researcher=%s curator=%s agents=%d hypotheses=%d generations=%d",
        s.crusoe_researcher_model,
        s.crusoe_curator_model,
        args.agents,
        args.hypotheses,
        args.generations,
    )

    rows = load_rows(args.csv)
    log.info("validation: %d rows, columns %s", len(rows), sorted(rows[0]))

    best, summary, memory, hypotheses, all_experiments = None, "", "", [], []
    for gen in range(1, args.generations + 1):
        best, summary, memory, hypotheses, experiments = await run_generation(
            gen, rows, args, hypotheses, memory, best
        )
        all_experiments.extend(
            {
                "gen": gen,
                "name": e.get("name"),
                "score": e.get("score"),
                "rmse": e.get("rmse"),
                "hypothesis": e.get("hypothesis"),
                "expression": e.get("expression"),
                "rationale": (e.get("rationale") or "")[:300],
            }
            for e in experiments
        )

    print("\n=== Shared Brain final report ===")
    print(f"best score: {best}")
    print(
        f"top experiment: {all_experiments[0]['name'] if all_experiments else '-'}"
        f" (score {all_experiments[0]['score'] if all_experiments else '-'})"
    )
    print("\n--- curator summary (top thinking) ---")
    print(summary[:2000])
    print("\n--- next hypotheses (first 10) ---")
    for h in hypotheses[:10]:
        print(f" * {h}")

    out = {
        "goal": args.goal,
        "best_score": best,
        "generations": args.generations,
        "summary": summary,
        "hypotheses": hypotheses,
        "experiments": all_experiments,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    asyncio.run(main())
