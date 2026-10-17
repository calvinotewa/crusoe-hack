"""Crusoe Intelligence Foundry inference. The API is OpenAI-compatible."""

import time
from functools import lru_cache

from openai import AsyncOpenAI

from app.config import get_settings


@lru_cache
def get_client() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(
        api_key=s.crusoe_api_key, base_url=s.crusoe_base_url, max_retries=s.crusoe_max_retries
    )


async def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    **kwargs,
) -> tuple[str, int]:
    """Returns (content, latency_ms).

    Reasoning models spend `max_tokens` on thinking first; empty content with
    finish_reason == "length" means the budget was too small.
    """
    started = time.perf_counter()
    resp = await get_client().chat.completions.create(
        model=model or get_settings().crusoe_default_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    return resp.choices[0].message.content or "", latency_ms


async def list_models() -> list[str]:
    models = await get_client().models.list()
    return [m.id for m in models.data]
