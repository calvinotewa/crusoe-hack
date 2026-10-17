"""Crusoe Intelligence Foundry inference. The API is OpenAI-compatible."""

import asyncio
import logging
import time
from collections import deque
from functools import lru_cache

from openai import AsyncOpenAI, RateLimitError

from app.config import get_settings

log = logging.getLogger(__name__)


class RateLimiter:
    """Sliding-window requests-per-minute limiter (Crusoe free tier: 30 RPM)."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.calls: deque[float] = deque()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        if not self.rpm:
            return
        async with self.lock:
            now = time.monotonic()
            while self.calls and now - self.calls[0] > 60:
                self.calls.popleft()
            if len(self.calls) >= self.rpm:
                wait = 60 - (now - self.calls[0]) + 0.05
                await asyncio.sleep(wait)
                now = time.monotonic()
                while self.calls and now - self.calls[0] > 60:
                    self.calls.popleft()
            self.calls.append(time.monotonic())


@lru_cache
def get_client() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(
        api_key=s.crusoe_api_key, base_url=s.crusoe_base_url, max_retries=s.crusoe_max_retries
    )


@lru_cache
def _limiter() -> RateLimiter:
    return RateLimiter(get_settings().crusoe_rate_limit_rpm)


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

    Rate limits: the client RPM limiter paces requests, and a 429 that survives
    the SDK's own retries (empirically the account window is tighter than the
    documented 30 RPM) triggers an adaptive cooldown + retry so one burst can't
    fail a whole harness run.
    """
    attempts = 3
    for attempt in range(attempts):
        await _limiter().acquire()
        started = time.perf_counter()
        try:
            resp = await get_client().chat.completions.create(
                model=model or get_settings().crusoe_default_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            return resp.choices[0].message.content or "", latency_ms
        except RateLimitError:
            if attempt == attempts - 1:
                raise
            log.warning("429 after SDK retries; cooling down 60s (attempt %d)", attempt + 1)
            await asyncio.sleep(60)
    return "", 0  # unreachable


async def list_models() -> list[str]:
    models = await get_client().models.list()
    return [m.id for m in models.data]
