"""Run untrusted (model-generated) code in Modal Sandboxes."""

import os

import modal

from app.config import get_settings
from app.schemas import SandboxResult

# Extend with whatever the generated code is allowed to import.
SANDBOX_IMAGE = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "pytest")

_app: modal.App | None = None


async def _get_app() -> modal.App:
    global _app
    if _app is None:
        s = get_settings()
        # Modal reads os.environ / ~/.modal.toml, not our .env; bridge tokens loaded by Settings.
        if s.modal_token_id and s.modal_token_secret:
            os.environ.setdefault("MODAL_TOKEN_ID", s.modal_token_id)
            os.environ.setdefault("MODAL_TOKEN_SECRET", s.modal_token_secret)
        _app = await modal.App.lookup.aio(s.modal_app_name, create_if_missing=True)
    return _app


async def run_python(code: str, timeout: int | None = None) -> SandboxResult:
    timeout = timeout or get_settings().sandbox_timeout_seconds
    sb = await modal.Sandbox.create.aio(
        app=await _get_app(),
        image=SANDBOX_IMAGE,
        timeout=timeout + 30,
        block_network=True,
    )
    try:
        p = await sb.exec.aio("python", "-c", code, timeout=timeout)
        stdout = await p.stdout.read.aio()
        stderr = await p.stderr.read.aio()
        await p.wait.aio()
        return SandboxResult(stdout=stdout, stderr=stderr, exit_code=p.returncode)
    finally:
        await sb.terminate.aio()
