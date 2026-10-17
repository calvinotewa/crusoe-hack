"""Direct access to the underlying providers, handy for debugging."""

from fastapi import APIRouter

from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse, SandboxRequest, SandboxResult
from app.services import inference, sandbox

router = APIRouter(prefix="/playground", tags=["playground"])


@router.get("/models")
async def models() -> list[str]:
    return await inference.list_models()


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    model = body.model or get_settings().crusoe_default_model
    content, latency_ms = await inference.chat(
        body.messages, model=model, temperature=body.temperature, max_tokens=body.max_tokens
    )
    return ChatResponse(model=model, content=content, latency_ms=latency_ms)


@router.post("/sandbox", response_model=SandboxResult)
async def run_sandbox(body: SandboxRequest):
    return await sandbox.run_python(body.code, body.timeout)
