"""The Shared Brain loop as an inspectable orchestration (dashboard path).

Per generation: agents -> benchmark -> summarize -> research -> hypotheses.
Instrumented for the dashboard: phase events, per-generation results, and
persistence into the SQLite store (`app/services/dashboard_store.py`) with
logical separation: every run belongs to a project; every generation and
experiment belongs to its run.

`scripts/shared_brain_mvp.py` is the plain CLI twin of this loop.
"""

import asyncio
import logging
import random
from datetime import UTC, datetime

from app.services import shared_brain

log = logging.getLogger(__name__)

PHASES = ["agents", "benchmark", "summarize", "research", "hypotheses"]
MAX_EVENTS = 300


def coerce_row(row: dict) -> dict:
    """CSV strings -> numbers where possible ('3.14' -> 3.14)."""
    out = {}
    for k, v in row.items():
        if isinstance(v, str):
            try:
                out[k] = float(v.strip())
            except ValueError:
                out[k] = v
        else:
            out[k] = v
    return out


class HarnessRunner:
    """One orchestration run against one project's dataset. Persists via `store`.

    Continued runs inherit state from the run they continue (see the dashboard's
    POST /api/runs/{id}/continue): the benchmark to beat (`initial_best_score`),
    the last generation's hypotheses, and the last summary + web learnings as
    the starting shared memory.
    """

    def __init__(
        self,
        params: dict,
        goal: str,
        rows: list[dict],
        store,
        run_id: str,
        *,
        initial_hypotheses: list[str] | None = None,
        initial_memory: str = "",
        initial_best_score: float | None = None,
    ):
        self.params = params
        self.goal = goal
        self.rows = rows
        self.store = store
        self.run_id = run_id
        self.initial_hypotheses = initial_hypotheses or []
        self.initial_memory = initial_memory
        self.state: dict = {
            "id": run_id,
            "params": params,
            "status": "running",
            "current_phase": "starting",
            "generation": 0,
            "best_score": initial_best_score,
            "events": [],
            "generations": [],
            "started_at": datetime.now(UTC).isoformat(),
            "finished_at": None,
            "error": None,
        }

    async def log(self, phase: str, msg: str) -> None:
        ev = {
            "ts": datetime.now(UTC).isoformat(),
            "phase": phase,
            "gen": self.state["generation"],
            "msg": msg,
        }
        self.state["events"].append(ev)
        self.state["events"] = self.state["events"][-MAX_EVENTS:]
        log.info("[%s] %s", phase, msg)
        await self._persist()

    async def _persist(self) -> None:
        await self.store.update_run(
            self.run_id,
            status=self.state["status"],
            current_phase=self.state["current_phase"],
            generation=self.state["generation"],
            best_score=self.state["best_score"],
            events=self.state["events"],
            error=self.state["error"],
            finished_at=self.state["finished_at"],
        )

    async def _set_phase(self, phase: str) -> None:
        self.state["current_phase"] = phase
        await self._persist()

    async def run(self) -> dict:
        p = self.params
        try:
            await self.log(
                "start",
                f"run {self.run_id}: {p['agents']} agents x {p['generations']} generations,"
                f" {p['hypotheses']} hypotheses/gen, top_k={p['top_k']}",
            )
            hypotheses: list[str] = list(self.initial_hypotheses)
            memory = self.initial_memory
            columns = sorted(set(self.rows[0]) - {"y"}) if self.rows else []
            sem = asyncio.Semaphore(p["concurrency"])

            for gen in range(1, p["generations"] + 1):
                self.state["generation"] = gen
                gen_hypotheses, gen_memory = hypotheses, memory

                # --- phase 1: agents produce experiments (testing hypotheses) ---
                await self._set_phase("agents")
                await self.log("agents", f"gen {gen}: spawning {p['agents']} researcher agents")
                done = 0

                async def agent_one(n: int, hyps: list[str], mem: str) -> dict:
                    async with sem:
                        hyp = hyps[(n - 1) % len(hyps)] if hyps else ""
                        exp = await shared_brain.agent_experiment(
                            n, self.goal, columns, hyp, mem, self.state["best_score"]
                        )
                        exp["hypothesis"] = hyp
                        return exp

                tasks = [
                    asyncio.create_task(agent_one(i + 1, gen_hypotheses, gen_memory))
                    for i in range(p["agents"])
                ]
                experiments: list[dict] = []
                for coro in asyncio.as_completed(tasks):
                    exp = await coro
                    done += 1
                    experiments.append(exp)
                    await self.log(
                        "agents", f"gen {gen}: agent {done}/{p['agents']} -> {exp.get('name')}"
                    )

                # --- phase 2: benchmark all experiments against the validation rows ---
                await self._set_phase("benchmark")
                await self.log("benchmark", f"gen {gen}: running {len(experiments)} experiments")
                bdone = 0

                async def bench_one(exp: dict) -> dict:
                    async with sem:
                        if not exp.get("code"):
                            return {**exp, "score": 0.0, "stderr_tail": "no code"}
                        res = await shared_brain.run_experiment(exp["code"], self.rows)
                        return {**exp, **res}

                btasks = [asyncio.create_task(bench_one(e)) for e in experiments]
                scored: list[dict] = []
                for coro in asyncio.as_completed(btasks):
                    res = await coro
                    bdone += 1
                    scored.append(res)
                    await self.log(
                        "benchmark",
                        f"gen {gen}: benchmark {bdone}/{len(experiments)}"
                        f" -> {res.get('name')}: score {res.get('score')}",
                    )
                scored.sort(key=lambda e: e.get("score") or 0.0, reverse=True)
                gen_best = scored[0].get("score") if scored else None
                known = [x for x in (self.state["best_score"], gen_best) if x is not None]
                self.state["best_score"] = max(known) if known else None

                # --- phase 3: curator summarizes the thinking of the top-k ---
                await self._set_phase("summarize")
                top = scored[: p["top_k"]]
                summary = await shared_brain.summarize_top(self.goal, top, p["top_k"])
                await self.log(
                    "summarize", f"gen {gen}: curator summarized top {len(top)} (best {gen_best})"
                )

                # --- phase 4: web search supplements the learnings ---
                await self._set_phase("research")
                query = await shared_brain.search_gap_query(summary)
                sources = await shared_brain.web_search(query, p.get("sources", 5))
                learnings = (
                    "\n".join(f"- {s['title']} ({s['url']})" for s in sources) or "(no results)"
                )
                await self.log("research", f"gen {gen}: query={query!r} -> {len(sources)} sources")

                # --- phase 5: exactly p['hypotheses'] for the next generation ---
                await self._set_phase("hypotheses")
                next_hypotheses = await shared_brain.generate_hypotheses(
                    self.goal, summary, learnings, p["hypotheses"], concurrency=p["concurrency"]
                )
                await self.log(
                    "hypotheses",
                    f"gen {gen}: {len(next_hypotheses)} hypotheses for gen {gen + 1}",
                )

                gen_record = {
                    "gen": gen,
                    "best_score": gen_best,
                    "num_experiments": len(scored),
                    "summary": summary,
                    "query": query,
                    "sources": sources,
                    "hypotheses": next_hypotheses,
                }
                self.state["generations"].append(gen_record)
                await self.store.add_generation(
                    self.run_id,
                    gen_record,
                    [
                        {
                            "name": e.get("name"),
                            "score": e.get("score"),
                            "rmse": e.get("rmse"),
                            "expression": e.get("expression"),
                            "hypothesis": e.get("hypothesis"),
                            "rationale": (e.get("rationale") or "")[:400],
                        }
                        for e in scored
                    ],
                )
                await self._persist()
                hypotheses = next_hypotheses
                memory = f"## Last generation's summary\n{summary}\n\n## Web learnings\n{learnings}"

            self.state["status"] = "completed"
            await self.log("done", f"run finished, best score {self.state['best_score']}")
        except asyncio.CancelledError:
            self.state["status"] = "stopped"
            await self.log("stop", "run cancelled")
        except Exception as e:
            log.exception("harness run failed")
            self.state["status"] = "failed"
            self.state["error"] = f"{type(e).__name__}: {e}"
            await self.log("error", f"run failed: {e}")
        finally:
            self.state["current_phase"] = "done"
            self.state["finished_at"] = datetime.now(UTC).isoformat()
            await self._persist()
        return self.state


def synthetic_ab_rows(n: int = 60, seed: int = 7) -> list[dict]:
    """Small built-in dataset: y = 2a + 3b + 0.5ab + noise (used by CLI, not the store)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a = rng.uniform(-5, 5)
        b = rng.uniform(-5, 5)
        y = 2 * a + 3 * b + 0.5 * a * b + rng.gauss(0, 0.3)
        rows.append({"a": round(a, 4), "b": round(b, 4), "y": round(y, 4)})
    return rows
