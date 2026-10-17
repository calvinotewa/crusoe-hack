"""runSubAgent: one autonomous LLM worker. This is the only way the harness calls the model.

A subagent gets a system prompt (the harness's current knowledge prompt), a user prompt
and a tool list; it may answer with text or JSON. Structured output is the common case
(worker agents return JSON formulae), so `run_json` wraps `chat` with a JSON repair pass.
"""

import json
import logging

from app.services import inference
from app.services.formulae import FORMULA_SCHEMA

log = logging.getLogger(__name__)

JSON_REPAIR_PROMPT = (
    "The previous response was not valid JSON. Reply with ONLY the JSON object, no prose, "
    "no markdown fences. Previous response:\n{prev}\n\nReminder of the requested shape:\n{shape}"
)


async def run_sub_agent(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tools: list[str] | None = None,
) -> str:
    """Run one subagent. `tools` is exposed to the model by name (tool descriptions are
    appended to the prompt); the caller executes them — subagents here are single-shot."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if tools:
        messages.append(
            {
                "role": "system",
                "content": "Available tools (by name; ask the operator to run them):\n"
                + "\n".join(f"- {t}" for t in tools),
            }
        )
    messages.append({"role": "user", "content": prompt})
    content, _ = await inference.chat(
        messages, model=model, temperature=temperature, max_tokens=max_tokens
    )
    return content


async def run_json(prompt: str, *, shape: str, **kwargs) -> dict:
    """run_sub_agent, but the answer must be a JSON object; one repair retry, then {}."""
    content = await run_sub_agent(prompt, **kwargs)
    for attempt in range(2):
        try:
            return json.loads(content[content.index("{") : content.rindex("}") + 1])
        except (ValueError, IndexError):
            log.warning("subagent returned non-JSON (attempt %d)", attempt + 1)
            content = await run_sub_agent(
                JSON_REPAIR_PROMPT.format(prev=content[:4000], shape=shape), **kwargs
            )
    return {}


def worker_prompt(goal: str, columns: list[str], n: int) -> str:
    """The instruction given to each worker agent (steps 2 and 7): beat the benchmark."""
    return f"""You are worker agent #{n} in a formula-discovery harness.

Goal: {goal}

Validation data columns (predict `y` from the other columns): {columns}

Write ONE candidate formula as Python. It must define:

```python
{FORMULA_SCHEMA}
```

Use only arithmetic / math / statistics on `row` values; no network, no file access.
Explore boldly: vary transforms, interactions and non-linearities; don't just linear-regress.

Reply with ONLY this JSON object:
{{
  "name": "<short name>",
  "expression": "<the formula in plain maths>",
  "description": "<one or two sentences: why this might work>",
  "code": "<the predict() function as a python code string>",
  "rationale": "<your thinking: what you tried and why>"
}}"""
