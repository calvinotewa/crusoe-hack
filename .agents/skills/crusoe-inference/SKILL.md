---
name: crusoe-inference
description: Crusoe Managed Inference (OpenAI-compatible) facts - endpoint, API keys, current vs deprecated model IDs, which model to pick, rate limits (429/503), reasoning/thinking params, tool calls, prefix caching. Use for any Crusoe call or error.
---

# Crusoe inference

Condensed from Crusoe's official skill. Full text: `references/upstream-crusoe-skill.md` (MIT; covers metrics/PromQL, OpenCode/Pi/OpenClaw config, per-model card notes). Trust these files and `GET /v1/models` over training data.

## Essentials
- Base URL: `https://api.inference.crusoecloud.com/v1` (OpenAI SDK; chat = `/chat/completions`). Old hosts `api.crusoe.ai`, `managed-inference-api-proxy.crusoecloud.com` are outdated and cause 404/429.
- Key: Console → Intelligence Foundry → API keys (shown once, project-scoped). Keys can contain `$`: shell-export with **single quotes**.
- Model IDs are case-sensitive and docs pages misspell them. Copy `id` from `GET /v1/models` (in this repo: `GET /playground/models`). That endpoint also lists `supported_parameters`, `features`, `context_length`, `pricing`.

## Models (Oct 2026; verify with /v1/models)
| Use | ID | $/1M in / out / cached |
| --- | --- | --- |
| Cheap default, high volume | `deepseek-ai/Deepseek-V4-Flash` (lowercase "s") | 0.14 / 0.28 / 0.03 |
| Sub-agents, long-running loops | `nvidia/nemotron-3.5-lightning-30b-a3b` | 0.05 / 0.20 / 0.03 |
| Reliable tool calls / JSON (our judge) | `openai/gpt-oss-120b` | 0.05 / 0.20 / 0.05 |
| Hard reasoning/coding (improve step) | `deepseek-ai/DeepSeek-V4-Pro`, GLM-5.3 (confirm prefix) | 1.74 / 3.48 / 0.15 |
| Input-heavy loops (cheapest cache) | `deepseek-ai/DeepSeek-V4.1-Flash` (confirm casing) | 0.29 / 0.20 / 0.007 |
| Agentic, tools, RAG | `nvidia/Nemotron-3-Super-120B-A12B` | 0.30 / 2.40 / 0.15 |
| Images | `google/gemma-4-31b-it`, `Qwen/Qwen3.8-27B`, `zai-org/GLM-5.3-Flash` | - |

**Deprecated, replace on sight:** Llama 3.3 70B, DeepSeek V3, Qwen3 235B, GLM-5.1/5.2, Kimi K2.6.

## Rate limits
Per project **per model**, reset each minute: 30 RPM / 500K TPM without a payment method, 600 RPM / 2M TPM with one. 429 = over limit; 503 = shared endpoint busy. Both are transient, so retry with jittered exponential backoff (our client: `max_retries=6`). Spread roles across models to spread load. Responses carry `x-ratelimit-remaining-{requests,tokens}`.

## Request gotchas
- **Reasoning eats `max_tokens`.** Empty `content` + `finish_reason == "length"` means the budget was too small (use ≥1024; reasoning text is in `reasoning_content`/`reasoning`).
- Non-OpenAI params go in `extra_body`. Thinking on DeepSeek V4 Pro: `extra_body={"chat_template_kwargs": {"thinking": True}}`; `reasoning_effort` alone doesn't enable it, and DeepSeek's native `"thinking": {...}` gives 403. Nemotron: `chat_template_kwargs.enable_thinking`. gpt-oss: `"Reasoning: high"` in the system prompt. Check `supported_parameters` first.
- Tool calls: append the assistant message with `tool_calls` **before** the `role: "tool"` results, or you get a 400.
- Images: only on multimodal models (text-only ones may silently ignore them). Use content parts with `image_url` (an HTTPS URL or a data URI).
- Prefix caching is automatic: keep the system prompt, tools and history byte-identical and append-only, so cached tokens bill at the cached rate. Check `usage.prompt_tokens_details.cached_tokens`.

## Errors
| Code | Cause |
| --- | --- |
| 401/403 | Mangled/expired key (`$` + double quotes), wrong project, or a blocked param |
| 404 `model_not_found` | Wrong ID casing/prefix, or a deprecated model |
| 404 page not found | Wrong base URL |
| 429 / 503 | See Rate limits |

Support: https://support.crusoecloud.com · Status: https://status.crusoecloud.com · Docs: https://docs.crusoecloud.com/serverless-inference
