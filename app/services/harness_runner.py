"""The Shared Brain loop as an inspectable orchestration (dashboard path).

Per generation: agents -> benchmark -> curate -> research -> hypotheses.
The curate phase is the curator gate (`app/services/curation.py`): it ranks experiments,
verifies the top claim on resamples, and only a claim that replicates moves the best score.
Shared memory is the curated store, not a free-form summary.
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

from app.services import curation, shared_brain

log = logging.getLogger(__name__)

PHASES = ["agents", "benchmark", "curate", "research", "hypotheses"]
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


def row_stats(rows: list[dict]) -> str:
    """Compact data-scale summary for agent prompts (targets/features span wide
    ranges on real datasets — agents need the scales to design their basis)."""
    if not rows:
        return "(empty)"
    ys = [float(r["y"]) for r in rows if isinstance(r.get("y"), (int, float))]
    ymean = sum(ys) / len(ys)
    parts = [f"{len(rows)} rows; target y: min {min(ys):.3g}, max {max(ys):.3g}, mean {ymean:.3g}"]
    for c in sorted(set(rows[0]) - {"y"}):
        vals = [float(r[c]) for r in rows if isinstance(r.get(c), (int, float))]
        if vals:
            parts.append(f"{c}: {min(vals):.3g}..{max(vals):.3g}")
    return "; ".join(parts)


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
        self.memory = curation.Memory()
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
            columns = sorted(set(self.rows[0]) - {"y"}) if self.rows else []
            stats = row_stats(self.rows)
            sem = asyncio.Semaphore(p["concurrency"])

            for gen in range(1, p["generations"] + 1):
                self.state["generation"] = gen
                gen_hypotheses = hypotheses
                gen_memory = "\n\n".join(
                    x for x in (self.initial_memory, self.memory.render()) if x
                )

                # --- phase 1: agents produce experiments (testing hypotheses) ---
                await self._set_phase("agents")
                await self.log("agents", f"gen {gen}: spawning {p['agents']} researcher agents")
                done = 0

                async def agent_one(n: int, hyps: list[str], mem: str) -> dict:
                    async with sem:
                        hyp = hyps[(n - 1) % len(hyps)] if hyps else ""
                        exp = await shared_brain.agent_experiment(
                            n, self.goal, columns, hyp, mem, self.state["best_score"], stats=stats
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
                        # 60s: fitting (e.g. least squares over all rows) can be slower
                        res = await shared_brain.run_experiment(
                            exp["code"], self.rows, timeout=60.0
                        )
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

                # --- phase 3: curator gate; only a replicated claim moves the best ---
                await self._set_phase("curate")
                incumbent = self.state["best_score"]
                incumbent_loss = (1.0 / incumbent - 1.0) if incumbent else None
                promoted = await curation.curate(
                    self.memory,
                    scored,
                    gen,
                    incumbent_loss,
                    lambda code: shared_brain.verify_experiment(code, self.rows),
                )
                if promoted is not None:
                    self.state["best_score"] = 1.0 / (1.0 + promoted.loss)
                summary = self.memory.render() or "(no verified learnings yet)"
                await self.log(
                    "curate",
                    f"gen {gen}: best {self.state['best_score']}, "
                    + (
                        f"promoted {promoted.experiment.get('name')}"
                        f" (resampled rmse {promoted.loss:.4f})"
                        if promoted
                        else "no claim replicated"
                    ),
                )

                # --- phase 4: web search supplements the learnings ---
                # A transient LLM/search failure degrades (no sources this gen),
                # it never fails the run.
                await self._set_phase("research")
                try:
                    query = await shared_brain.search_gap_query(summary)
                    sources = await shared_brain.web_search(query, p.get("sources", 5))
                except Exception as e:
                    log.warning("research phase failed; degrading", exc_info=True)
                    await self.log("research", f"gen {gen}: DEGRADED ({e}); no sources this gen")
                    query, sources = "(research unavailable)", []
                learnings = (
                    "\n".join(f"- {s['title']} ({s['url']})" for s in sources) or "(no results)"
                )
                await self.log("research", f"gen {gen}: query={query!r} -> {len(sources)} sources")

                # --- phase 5: exactly p['hypotheses'] for the next generation ---
                # If the curator is down, carry this generation's hypotheses over
                # (already the fixed count) so the loop still invests a full gen.
                await self._set_phase("hypotheses")
                try:
                    next_hypotheses = await shared_brain.generate_hypotheses(
                        self.goal, summary, learnings, p["hypotheses"], concurrency=p["concurrency"]
                    )
                except Exception as e:
                    log.warning("hypotheses phase failed; degrading", exc_info=True)
                    await self.log(
                        "hypotheses", f"gen {gen}: DEGRADED ({e}); carrying hypotheses over"
                    )
                    next_hypotheses = list(gen_hypotheses)
                    while len(next_hypotheses) < p["hypotheses"]:
                        next_hypotheses.append(
                            f"Open question: what else could beat the current best result?"
                            f" (slot {len(next_hypotheses) + 1})"
                        )
                    next_hypotheses = next_hypotheses[: p["hypotheses"]]
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
