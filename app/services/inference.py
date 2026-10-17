"""Crusoe Intelligence Foundry inference. The API is OpenAI-compatible.

Resilience (see `chat`): the OpenAI SDK already retries 429/5xx/connection errors
with backoff (`CRUSOE_MAX_RETRIES`). On top, `chat` adds:

- client-side RPM pacing (`CRUSOE_RATE_LIMIT_RPM`),
- extra attempts with exponential backoff for transient errors
  (`CRUSOE_CHAT_ATTEMPTS`, `CRUSOE_CHAT_BACKOFF_SECONDS`; 429s get a longer
  cooldown),
- a model fallback chain (`CRUSOE_FALLBACK_MODELS`) so a model that 404s/400s
  or exhausts its retries doesn't fail the call,
- one budget-doubling retry when a reasoning model burns `max_tokens` on
  thinking and returns empty content.
"""

import asyncio
import logging
import time
from collections import deque
from functools import lru_cache

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

from app.config import get_settings

log = logging.getLogger(__name__)

# Transient: retry with backoff, then fall through to the next fallback model.
TRANSIENT_ERRORS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)
# The requested model is wrong for this account: skip straight to fallbacks.
MODEL_ERRORS = (NotFoundError, BadRequestError)


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


def _fallback_chain(model: str | None) -> list[str]:
    """Requested model (or default), then the configured fallbacks, deduped."""
    s = get_settings()
    primary = model or s.crusoe_default_model
    chain = [primary]
    for m in (x.strip() for x in s.crusoe_fallback_models.split(",")):
        if m and m not in chain:
            chain.append(m)
    return chain


async def _create_once(model: str, messages: list, temperature: float, max_tokens: int, **kwargs):
    return await get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


async def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    **kwargs,
) -> tuple[str, int]:
    """Returns (content, latency_ms).

    Reasoning models spend `max_tokens` on thinking first; finish_reason ==
    "length" means the answer was truncated (or empty) — retried with a doubled
    budget, keeping the longest partial as a last-resort answer.

    Error handling: for each model in the fallback chain, the SDK retries
    transient failures first (max_retries); then `CRUSOE_CHAT_ATTEMPTS` extra
    attempts with exponential backoff (RateLimitError: longer cooldown).
    A model that 404s/400s or never succeeds raises through only after every
    fallback has been tried — one flaky model never fails a harness run.
    """
    s = get_settings()
    attempts = max(1, s.crusoe_chat_attempts)
    backoff = max(0.0, s.crusoe_chat_backoff_seconds)
    chain = _fallback_chain(model)
    started = time.perf_counter()
    last_error: Exception | None = None
    partial = ""  # longest truncated-but-nonempty answer seen

    for mi, current_model in enumerate(chain):
        budget = max_tokens
        for attempt in range(attempts + 1):  # +1: the length-budget retry
            await _limiter().acquire()
            try:
                resp = await _create_once(current_model, messages, temperature, budget, **kwargs)
            except TRANSIENT_ERRORS as e:
                last_error = e
                if attempt == attempts - 1:
                    break  # next fallback model
                wait = 60.0 if isinstance(e, RateLimitError) else backoff * (2**attempt)
                log.warning(
                    "transient %s on %s (attempt %d); backing off %.0fs",
                    type(e).__name__,
                    current_model,
                    attempt + 1,
                    wait,
                )
                await asyncio.sleep(wait)
                continue
            except MODEL_ERRORS as e:
                last_error = e
                log.warning("model error on %s: %s; trying fallbacks", current_model, e)
                break  # next fallback model
            latency_ms = int((time.perf_counter() - started) * 1000)
            content = resp.choices[0].message.content or ""
            finish = getattr(resp.choices[0], "finish_reason", None)
            if finish == "length" and attempt < attempts:
                # Truncated (reasoning models burn budget on thinking first):
                # harness prompts need complete output, so retry with a doubled
                # budget — keeping the longest partial as a last-resort answer.
                log.warning(
                    "truncated content (finish=length) from %s; retrying with %d tokens",
                    current_model,
                    budget * 2,
                )
                if len(content) > len(partial):
                    partial = content
                budget *= 2
                continue
            if mi > 0:
                log.info("served by fallback model %s", current_model)
            return (content, latency_ms) if len(content) >= len(partial) else (partial, latency_ms)
    if partial:
        return partial, int((time.perf_counter() - started) * 1000)  # truncated, but usable
    raise last_error if last_error else RuntimeError("chat exhausted all models without an error")


async def list_models() -> list[str]:
    models = await get_client().models.list()
    return [m.id for m in models.data]
