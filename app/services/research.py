"""Step 5: "if these agents knew X they could have found a better result."

Summarise a generation's traces (from Postgres + bucket), ask a subagent what
knowledge was missing, then fetch `research_num_sources` web sources via the you.com
API (or, if no key is set, a stub source pointing at the query so the loop stays runnable).
"""

import logging

import httpx

from app.config import get_settings
from app.db import SessionLocal
from app.models import HarnessSession, ResearchSource
from app.services import subagent

log = logging.getLogger(__name__)

GAP_PROMPT = """You are reviewing a generation of formula-discovery agents.

Goal: {goal}
Current best score (1/(1+rmse)): {best}
Top formulae this generation:
{top}

Write ONE web-search query for knowledge these agents were missing ("if the agents
knew X they could have found a better result"). Reply with ONLY the query, no quotes."""

TOP_FORMULA_PROMPT = """Summarise the session thinking of the 3 best-performing formulae
below. For each: what it does, why it worked, and what the agent apparently did not know
(a knowledge gap). Then list the 3 most valuable things to carry into the next generation
and consolidate them into a single "inherited learnings" section. Be specific and terse.

{blobs}"""


def _stub_sources(query: str, n: int) -> list[dict]:
    return [
        {
            "title": f"Research stub {i + 1} for: {query}",
            "url": f"https://you.com/search?q={query}",
            "snippet": None,
        }
        for i in range(n)
    ]


async def fetch_sources(query: str, n: int) -> list[dict]:
    """Search you.com (YDC API) for sources on `query`."""
    s = get_settings()
    if not s.you_api_key:
        log.warning("YOU_API_KEY not set; returning stub research sources")
        return _stub_sources(query, n)
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{s.you_api_base.rstrip('/')}/search",
            params={"q": query, "num_results": n, "safesearch": "Off"},
            headers={"X-API-Key": s.you_api_key},
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
    return [
        {
            "title": h.get("title", "")[:500],
            "url": h.get("url", "")[:500],
            "snippet": (h.get("snippets") or [""])[0]
            if h.get("snippets")
            else h.get("description"),
        }
        for h in hits[:n]
    ]


async def store_sources(session_id, generation: int, query: str, sources: list[dict]) -> list:
    async with SessionLocal() as session:
        rows = [
            ResearchSource(
                session_id=session_id,
                generation=generation,
                query=query,
                title=s["title"][:500],
                url=s["url"][:500],
                snippet=s.get("snippet"),
            )
            for s in sources
        ]
        session.add_all(rows)
        await session.commit()
        for r in rows:
            await session.refresh(r)
        return rows


async def summarise_session(
    session: HarnessSession, top_formulae: list, top_benchmarks: list
) -> str:
    """Step 4: LLM summary of how the best agents thought (the "consolidated" blob)."""
    blobs = []
    for f, b in zip(top_formulae, top_benchmarks, strict=False):
        blobs.append(
            f"### {f.name} (score {b.score if b else '?'})\n"
            f"Expression: {f.expression}\n"
            f"Description: {f.description or '(none)'}\n"
            f"Agent thinking: {f.rationale or '(none)'}"
        )
    s = get_settings()
    return await subagent.run_sub_agent(
        TOP_FORMULA_PROMPT.format(goal=session.goal, blobs="\n\n".join(blobs)),
        model=s.crusoe_judge_model,
        temperature=0.2,
    )


async def research_gaps(
    session: HarnessSession, top_formulae: list, top_benchmarks: list, generation: int
) -> tuple[str, list]:
    """Full step 5: query synthesis -> you.com fetch -> stored sources.

    Returns (query, sources). Sources are persisted with the session for the RAG tool.
    """
    s = get_settings()
    content = await subagent.run_sub_agent(
        GAP_PROMPT.format(
            goal=session.goal,
            best=session.best_score,
            top="\n".join(
                f"- {f.name}: {f.expression} (score {(b.score if b else '?')})"
                for f, b in zip(top_formulae, top_benchmarks, strict=False)
            ),
        ),
        model=s.crusoe_judge_model,
        temperature=0.2,
        max_tokens=2048,
    )
    query = content.strip().strip('"').splitlines()[0][:2000]
    sources = await fetch_sources(query, s.research_num_sources)
    stored = await store_sources(session.id, generation, query, sources) if sources else []
    return query, stored
