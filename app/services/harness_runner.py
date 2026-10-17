"""Runs the Shared Brain loop as an inspectable, persistable orchestration.

`scripts/shared_brain_mvp.py` prints and exits; this module is the same 5-step
flow instrumented for the dashboard (`scripts/dashboard_server.py`): phase
events, per-generation results, and JSON persistence under `data/harness_runs/`.

Per generation: agents -> benchmark -> summarize -> research -> hypotheses.

Storage note: Crusoe is a stateless inference API - it stores nothing. Runs made
here are persisted locally as JSON files; the API path (`orchestrator.py`)
additionally persists to Neon Postgres + Object Storage.
"""

import asyncio
import json
import logging
import random
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.services import shared_brain

log = logging.getLogger(__name__)

PHASES = ["agents", "benchmark", "summarize", "research", "hypotheses"]
MAX_EVENTS = 300


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _coerce(row: dict) -> dict:
    """CSV strings -> numbers where possible ('3.14' -> 3.14)."""
    out = {}
    for k, v in row.items():
        if isinstance(v, str):
            s = v.strip()
            try:
                out[k] = float(s)
            except ValueError:
                out[k] = v
        else:
            out[k] = v
    return out


def load_dataset(name: str) -> tuple[str, list[dict]]:
    """Built-in datasets: 'churn' (examples/churn_synthetic.csv) or 'ab' (synthetic)."""
    if name == "churn":
        import csv

        path = Path(__file__).resolve().parent.parent.parent / "examples" / "churn_synthetic.csv"
        with path.open() as f:
            rows = [_coerce(r) for r in csv.DictReader(f)]
        goal = (
            "Predict customer churn: y=1 means the customer churned, y=0 means they "
            "stayed. Features: tenure_months, monthly_charges, support_calls, "
            "satisfaction, contract_months. Return the churn probability (0..1) per "
            "row. Scored by RMSE against the 0/1 labels (Brier score) - lower is better."
        )
        return goal, rows
    # 'ab': synthetic y = 2a + 3b + 0.5ab + noise
    rng = random.Random(7)
    rows = []
    for _ in range(60):
        a = rng.uniform(-5, 5)
        b = rng.uniform(-5, 5)
        y = 2 * a + 3 * b + 0.5 * a * b + rng.gauss(0, 0.3)
        rows.append({"a": round(a, 4), "b": round(b, 4), "y": round(y, 4)})
    return "Predict `y` from the other columns with the lowest possible RMSE.", rows


class HarnessRunner:
    """One orchestration run: state dict + the loop, persisted as JSON per phase."""

    def __init__(self, params: dict, rows: list[dict], store_dir: Path):
        self.params = params
        self.rows = rows
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        if params.get("goal"):
            self.goal = params["goal"]
        else:
            self.goal, _ = load_dataset(params["dataset"])
        self.state: dict = {
            "id": uuid.uuid4().hex[:12],
            "params": params,
            "status": "running",
            "current_phase": "starting",
            "phases_done": [],
            "generation": 0,
            "best_score": None,
            "events": [],
            "generations": [],
            "started_at": _now(),
            "finished_at": None,
            "error": None,
        }
        self.persist()

    # -- persistence ------------------------------------------------------

    @property
    def path(self) -> Path:
        return self.store_dir / f"{self.state['id']}.json"

    def persist(self) -> None:
        self.path.write_text(json.dumps(self.state, indent=2))

    # -- events ----------------------------------------------------------

    def log(self, phase: str, msg: str) -> None:
        ev = {"ts": _now(), "phase": phase, "gen": self.state["generation"], "msg": msg}
        self.state["events"].append(ev)
        self.state["events"] = self.state["events"][-MAX_EVENTS:]
        log.info("[%s] %s", phase, msg)

    # -- the loop ---------------------------------------------------------

    async def _set_phase(self, phase: str) -> None:
        self.state["current_phase"] = phase
        self.persist()

    async def run(self) -> dict:
        p = self.params
        try:
            self.log(
                "start",
                f"run {self.state['id']}: dataset={p['dataset']} "
                f"{p['agents']} agents x {p['generations']} generations, "
                f"{p['hypotheses']} hypotheses/gen, top_k={p['top_k']}",
            )
            hypotheses: list[str] = []
            memory = ""
            columns = sorted(set(self.rows[0]) - {"y"}) if self.rows else []
            sem = asyncio.Semaphore(p["concurrency"])

            for gen in range(1, p["generations"] + 1):
                self.state["generation"] = gen
                self.state["phases_done"] = []
                gen_hypotheses, gen_memory = hypotheses, memory

                # --- phase 1: agents produce experiments (testing hypotheses) ---
                await self._set_phase("agents")
                self.log("agents", f"gen {gen}: spawning {p['agents']} researcher agents")
                done = 0

                async def agent_one(n: int, hyps: list[str] = None, mem: str = "") -> dict:
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
                    self.log(
                        "agents",
                        f"gen {gen}: agent {done}/{p['agents']} -> {exp.get('name')}",
                    )

                # --- phase 2: benchmark all experiments against the rows ---
                await self._set_phase("benchmark")
                self.log("benchmark", f"gen {gen}: running {len(experiments)} experiments")
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
                    self.log(
                        "benchmark",
                        f"gen {gen}: benchmark {bdone}/{len(experiments)} "
                        f"-> {res.get('name')}: score {res.get('score')}",
                    )
                scored.sort(key=lambda e: e.get("score") or 0.0, reverse=True)
                gen_best = scored[0].get("score") if scored else None
                known = [x for x in (self.state["best_score"], gen_best) if x is not None]
                self.state["best_score"] = max(known) if known else None

                # --- phase 3: curator summarizes the thinking of the top-k ---
                await self._set_phase("summarize")
                top = scored[: p["top_k"]]
                summary = await shared_brain.summarize_top(self.goal, top, p["top_k"])
                self.log(
                    "summarize", f"gen {gen}: curator summarized top {len(top)} (best {gen_best})"
                )

                # --- phase 4: web search supplements the learnings ---
                await self._set_phase("research")
                query = await shared_brain.search_gap_query(summary)
                sources = await shared_brain.web_search(query, p.get("sources", 5))
                learnings = (
                    "\n".join(f"- {s['title']} ({s['url']})" for s in sources) or "(no results)"
                )
                self.log("research", f"gen {gen}: query={query!r} -> {len(sources)} sources")

                # --- phase 5: curator generates next-generation hypotheses ---
                await self._set_phase("hypotheses")
                next_hypotheses = await shared_brain.generate_hypotheses(
                    self.goal, summary, learnings, p["hypotheses"], concurrency=p["concurrency"]
                )
                self.log(
                    "hypotheses",
                    f"gen {gen}: {len(next_hypotheses)} hypotheses for gen {gen + 1}",
                )

                self.state["generations"].append(
                    {
                        "gen": gen,
                        "best_score": gen_best,
                        "num_experiments": len(scored),
                        "top": {
                            "name": scored[0].get("name"),
                            "score": scored[0].get("score"),
                            "rmse": scored[0].get("rmse"),
                            "expression": scored[0].get("expression"),
                        },
                        "experiments": [
                            {
                                "name": e.get("name"),
                                "score": e.get("score"),
                                "rmse": e.get("rmse"),
                                "expression": e.get("expression"),
                                "hypothesis": e.get("hypothesis"),
                                "rationale": (e.get("rationale") or "")[:300],
                            }
                            for e in scored
                        ],
                        "summary": summary,
                        "query": query,
                        "sources": sources,
                        "hypotheses": next_hypotheses,
                    }
                )
                self.state["phases_done"] = PHASES[:]
                self.persist()
                hypotheses, memory = (
                    next_hypotheses,
                    (f"## Last generation's summary\n{summary}\n\n## Web learnings\n{learnings}"),
                )

            self.state["status"] = "completed"
            self.log("done", f"run finished, best score {self.state['best_score']}")
        except asyncio.CancelledError:
            self.state["status"] = "stopped"
            self.log("stop", "run cancelled")
        except Exception as e:
            log.exception("harness run failed")
            self.state["status"] = "failed"
            self.state["error"] = f"{type(e).__name__}: {e}"
            self.log("error", f"run failed: {e}")
        finally:
            self.state["current_phase"] = "done"
            self.state["finished_at"] = _now()
            self.persist()
        return self.state
