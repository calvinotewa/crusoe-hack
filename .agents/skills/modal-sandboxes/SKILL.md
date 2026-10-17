---
name: modal-sandboxes
description: Running untrusted model-generated code in Modal Sandboxes (app/services/sandbox.py, the "code" grader). Use for sandbox create/exec, exit codes, images, isolation, scaling evals, Modal auth errors.
---

# Modal Sandboxes

The `modal` SDK is 1.6.x. Full verified API notes (signatures, filesystem API, snapshots, warm pools, costs): `references/full-guide.md`. Verify any parameter before using it: `uv run python -c "import modal,inspect; print(inspect.signature(modal.Sandbox.create))"`. Old docs show `sb.open()`/`sb.ls()`, which no longer exist; use `sb.filesystem.*`.

## Rules
1. Inside FastAPI, use only `.aio` variants (`Sandbox.create.aio`, `sb.exec.aio`, `p.stdout.read.aio`, `p.wait.aio`, `sb.terminate.aio`).
2. Create before `try:`, call `await sb.terminate.aio()` in `finally:`. A leaked sandbox bills until its `timeout`.
3. Model output is hostile: `block_network=True`, an exec `timeout` plus a slightly longer sandbox `timeout`, `cpu=(req, limit)`/`memory=(req, limit)` caps, **no** `secrets=` or credential `env=`, no real-data volumes. `block_network` can't be combined with allowlists or ports.
4. Tests never call Modal: monkeypatch `app.services.sandbox.run_python`, and keep `evaluator` calling it through the module.

## Pattern
```python
sb = await modal.Sandbox.create.aio(app=app, image=IMAGE, timeout=t + 30, block_network=True,
                                    cpu=(0.5, 2.0), memory=(512, 2048), workdir="/sandbox")
try:
    await sb.filesystem.write_text.aio(code, "/sandbox/main.py")   # better than `python -c` for big code
    p = await sb.exec.aio("python", "main.py", timeout=t)
    out, err = await asyncio.gather(p.stdout.read.aio(), p.stderr.read.aio())
    code = await p.wait.aio()
finally:
    await sb.terminate.aio()
```
- Exit codes: process code normally; **`-1` = exec timeout (no exception, output truncated)**; `128+N` = killed by a signal (`137` = SIGKILL/OOM/terminate). `sb.wait()` raises `SandboxTimeoutError` when the sandbox lifetime runs out.
- Images: define them at module level, because they build lazily on the first `create`, and changing them causes a rebuild. The network is blocked at run time, so pre-install every package the generated code may import. To avoid rebuilds during requests, `build` + `publish` a named image at deploy time and use `Image.from_name`.

## Auth
`modal token new` writes `~/.modal.toml`; alternatively set `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`. Values in `.env` reach Modal only because `sandbox._get_app()` copies them from `Settings` into `os.environ` before the first `App.lookup`. Keep that bridge.

## Scaling evals
One sandbox per task (current design) gives the cleanest isolation. Reusing one sandbox for several execs is cheaper but leaks state between them. Warm pools, `snapshot_filesystem` and tags for cleanup are in the full guide. Cap fan-out with an `asyncio.Semaphore`; the ceiling is your plan's container limit. Billing is per second on max(request, usage).
