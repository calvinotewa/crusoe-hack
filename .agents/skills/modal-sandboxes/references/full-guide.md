
# Modal Sandboxes for the eval loop

The project grades "code" tasks by running `extract_code(output) + task.test` in a Modal
Sandbox and scoring `1.0` when the exit code is `0`. The integration is
`app/services/sandbox.py` (`run_python(code, timeout=None) -> SandboxResult`), called from
`grade()` in `app/services/evaluator.py`. Settings live in `app/config.py`
(`modal_app_name`, `sandbox_timeout_seconds`).

Installed SDK: `modal` 1.6.2 (`pyproject.toml` pins `modal>=1.0`). Everything below was
checked against 1.6.2's signatures.

## 1. Rules

1. **Use async inside FastAPI.** Every blocking call has a `.aio` variant:
   `await modal.Sandbox.create.aio(...)`, `await sb.exec.aio(...)`,
   `await p.stdout.read.aio()`, `await p.wait.aio()`, `await sb.terminate.aio()`,
   `await sb.filesystem.write_text.aio(...)`, `await modal.App.lookup.aio(...)`.
   Calling the sync form inside the event loop blocks every request.
2. **Always terminate in `finally`.** A leaked sandbox keeps billing until its `timeout`
   (the default is 300 s). Put `create` before `try:` and `await sb.terminate.aio()` in `finally:`.
3. **Treat generated code as hostile:**
   - `block_network=True` (already set)
   - an exec `timeout`, plus a sandbox `timeout` a bit longer than that
   - capped `cpu=(req, limit)` / `memory=(req, limit)`
   - **never** pass `secrets=` or sensitive `env=` to a sandbox that runs model output
   - never mount Volumes that hold real data
4. **Keep providers faked in tests.** Never call real Modal from pytest (section 8).
5. **Verify before you write.** The Modal API changes often: for example, `sb.open()` and
   `sb.ls()` from older docs **do not exist** in 1.6.2. Check the signature before you use a
   parameter:
   ```bash
   uv run python -c "import modal, inspect; print(modal.__version__); print(inspect.signature(modal.Sandbox.create))"
   ```
   Also check `https://modal.com/docs/sdk/py/latest/Sandbox.md`. Don't use underscore-prefixed
   `_experimental_*` APIs unless the user asks for them.

## 2. Auth and setup

- **Local dev:** `uv run modal token new` opens a browser and writes `~/.modal.toml`.
  Check it with `uv run modal token info`.
- **Servers/CI:** set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` as real process
  environment variables. Modal reads `os.environ` and `~/.modal.toml` directly.
  - `.env` / `.env.local` also work: `Settings` declares `modal_token_id`/`modal_token_secret`
    and `sandbox._get_app()` copies them into `os.environ` (via `setdefault`) before the first
    `App.lookup`. pydantic-settings alone does **not** export to `os.environ`, so keep that bridge
    if you refactor, and don't create Modal objects before `_get_app()` runs.
- **App:** sandboxes must belong to an App. `sandbox.py` lazily caches one:
  ```python
  _app = await modal.App.lookup.aio(get_settings().modal_app_name, create_if_missing=True)
  ```
  `MODAL_APP_NAME` (default `crusoe-hack-sandboxes`) groups the sandboxes in the dashboard,
  where you can inspect them. You don't need `modal deploy` or `modal run`: the FastAPI
  process creates the sandboxes directly.

## 3. Core lifecycle (verified, async)

```python
import asyncio
import modal

SANDBOX_IMAGE = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "pytest")


async def run_python(code: str, timeout: int = 120) -> SandboxResult:
    sb = await modal.Sandbox.create.aio(
        app=await _get_app(),
        image=SANDBOX_IMAGE,
        timeout=timeout + 30,  # max lifetime of the sandbox, seconds (default 300, max 24h)
        idle_timeout=None,  # optional: kill after N idle seconds (exists in 1.6.2)
        cpu=(0.5, 2.0),  # (request, hard limit) in physical cores
        memory=(512, 2048),  # (request, hard limit) in MiB
        workdir="/sandbox",  # default cwd for the container
        block_network=True,
    )
    try:
        await sb.filesystem.write_text.aio(code, "/sandbox/main.py")
        p = await sb.exec.aio(
            "python",
            "main.py",
            timeout=timeout,  # per-exec limit, seconds
            workdir="/sandbox",  # must be absolute
            env={"PYTHONUNBUFFERED": "1"},
        )
        stdout, stderr = await asyncio.gather(p.stdout.read.aio(), p.stderr.read.aio())
        exit_code = await p.wait.aio()  # wait() returns the exit code; also p.returncode
        return SandboxResult(stdout=stdout, stderr=stderr, exit_code=exit_code)
    finally:
        await sb.terminate.aio()
```

Reference signatures (1.6.2):

- `Sandbox.create(*args, app, name, tags, image, env, secrets, timeout=300, idle_timeout=None, workdir, gpu, cpu, memory, runtime: 'gvisor'|'vm'|None, block_network=False, outbound_cidr_allowlist, outbound_domain_allowlist, inbound_cidr_allowlist, volumes={}, readiness_probe, verbose, ...)`
  - Positional `*args` set the container CMD. Omit them for an exec-driven sandbox.
- `sb.exec(*args, stdout=StreamType.PIPE, stderr=StreamType.PIPE, timeout=None, workdir=None, env=None, secrets=None, text=True, bufsize=-1, pty=False)` returns a `ContainerProcess`.
- `ContainerProcess`: `.stdout` / `.stderr` (StreamReader: `read()`, async-iterable line by line), `.stdin`, `wait() -> int`, `poll() -> int | None`, `returncode`.
- `sb.terminate(*, wait=False) -> int | None`, `sb.wait(raise_on_termination=True)`, `sb.poll()`, `sb.returncode`, `sb.object_id`, `sb.get_dashboard_url()`, `Sandbox.from_id(id)`, `Sandbox.list(app_id=, tags=)`, `sb.set_tags(...)`, `sb.detach()`.
- `text=True` (the default) gives `str` output. Use `text=False` for bytes.
- `StreamType` lives in `modal.stream_type` (`PIPE`, `STDOUT`, `DEVNULL`). Use `DEVNULL` when you don't need the output.

**Exit codes (verified in the SDK source, `modal/container_process.py`):**

| Situation | What you see |
|---|---|
| Normal exit | the process's own code (`0`, `1` for an uncaught exception/assert, ...) |
| Killed by signal N | `128 + N`. SIGKILL is `137` (this is also what `sb.terminate()` gives `sb.returncode`) |
| **Exec `timeout` exceeded** | **`-1`**. No exception is raised, and the stdout/stderr read just ends early |
| Sandbox lifetime `timeout` hit | `await sb.wait.aio()` raises `modal.exception.SandboxTimeoutError` |
| Sandbox terminated externally | `sb.wait()` raises `SandboxTerminatedError` (unless `raise_on_termination=False`) |

In the grader, any non-zero code (including `-1` and `137`) scores 0. `evaluator.grade()` puts
"timed out" or "killed (OOM or terminated)" at the start of `feedback` for those codes.

## 4. Images

- Builders verified on `modal.Image`:
  - `debian_slim(python_version=...)`, `from_registry(tag, ...)`, `from_name(name)`, `from_id(id)`
  - `.pip_install(*pkgs)`, `.uv_pip_install(*pkgs, requirements=...)`, `.apt_install(...)`,
    `.run_commands(...)`, `.env({...})`, `.workdir(path)`
  - `.add_local_file(local, remote, copy=False)`, `.add_local_dir(local, remote, copy=False, ignore=[...])`,
    `.add_local_python_source(*modules)`
  - `.build(app)`, `.publish(name)`
- **Define the image at module level** (like `SANDBOX_IMAGE`). The definition is lazy: Modal
  builds it the first time a sandbox uses it (or when you call `.build(app)` eagerly) and
  reuses that build afterwards. A module constant keeps one stable definition for the whole
  process. Changing the definition (e.g. the package list) invalidates it, and the next
  `create` blocks on a rebuild.
- To see build logs while debugging, wrap the call: `with modal.enable_output(): ...`.
- **Avoid blocking requests on rebuilds** (recommended by Modal): build and publish a
  named image in a deploy/CI step, then reference it by name at runtime:
  ```python
  # scripts/build_sandbox_image.py (run on deploy)
  app = await modal.App.lookup.aio("crusoe-hack-sandboxes", create_if_missing=True)
  with modal.enable_output():
      await SANDBOX_IMAGE.build.aio(app)
  await SANDBOX_IMAGE.publish.aio("crusoe-hack-sandbox-runtime")
  # runtime
  image = modal.Image.from_name("crusoe-hack-sandbox-runtime")
  ```
- External registry tags are treated as immutable once pulled. Bump the tag to pick up changes.
- Extend `pip_install` only with packages the generated code is *allowed* to import.
  The network is blocked at run time, so the code can't `pip install` its own packages.

## 5. Getting files in and out

`sb.filesystem` (1.6.2) has these methods, each with an `.aio` variant. Every remote path
must be absolute.

| Method | Notes |
|---|---|
| `write_text(data, remote_path)` / `write_bytes(data, remote_path)` | creates parent dirs, overwrites |
| `read_text(remote_path)` / `read_bytes(remote_path)` | raises `SandboxFilesystemFileTooLargeError` past the read limit |
| `copy_from_local(local_path, remote_path)` / `copy_to_local(remote_path, local_path)` | single files |
| `list_files(path)`, `stat(path)`, `make_directory(path, create_parents=True)`, `remove(path, recursive=False)`, `watch(path, ...)` | |

Errors are in `modal.exception` (`SandboxFilesystemNotFoundError`,
`SandboxFilesystemPermissionError`, ... all subclass `SandboxFilesystemError`).

**Prefer files over `python -c`.** The current code passes the whole program as an argv
string (`exec("python", "-c", code)`). For larger programs, or a solution plus a test
suite, write the files and run them:

```python
await sb.filesystem.write_text.aio(solution, "/sandbox/solution.py")
await sb.filesystem.write_text.aio(task.test, "/sandbox/test_solution.py")
p = await sb.exec.aio("python", "-m", "pytest", "-q", "-x", workdir="/sandbox", timeout=60)
```

This also gives real file names and line numbers in tracebacks, which helps the feedback
text that `grade()` stores.

For data shared across many sandboxes, bake it into the image (`add_local_dir`).
Modal Volumes (`volumes={...}`) also work, but don't mount writable shared volumes into
untrusted sandboxes.

## 6. Networking and security for untrusted code

- **Isolation:** sandboxes run under gVisor (userspace syscall layer) or a VM
  (`runtime="vm"`, own kernel via KVM). Leave `runtime` unset unless you need Docker,
  FUSE, or nested cgroups. Sandboxes **can't access other resources in the Modal
  workspace** (unlike Functions), so the blast radius is the sandbox itself.
- **Outbound controls** (choose one):
  - `block_network=True` drops all outbound traffic, DNS included. This is the right
    default for graders.
  - `outbound_cidr_allowlist=[...]` allows only those IP ranges, any protocol.
  - `outbound_domain_allowlist=[...]` (Beta) allows only TLS on port 443 to listed SNI
    names, with `*.` wildcards. It's vulnerable to domain fronting, and DNS stays open.
  - With `block_network=True` you **cannot** also pass the allowlists or
    `inbound_cidr_allowlist`.
- By default a sandbox accepts **no inbound connections**. Don't add `encrypted_ports`,
  `unencrypted_ports`, or `h2_ports` for graders.
- **No secrets:** don't pass `secrets=[...]` or credential-bearing `env=` to
  `Sandbox.create` or `exec` for untrusted code. Anything injected is readable by the
  code and could end up in `stdout`, which is persisted as feedback.
- **Resource caps:** without a limit, sandboxes can burst above their request.
  `cpu=(0.5, 2.0)` and `memory=(512, 2048)` set hard caps, which matters when model
  code fork-bombs or allocates without bound.
- Truncate what you store: the grader already keeps only the last 2000 chars of output.
  Keep a cap like that on anything persisted.

## 7. Performance for evals at scale

- **One sandbox per task (current design):** strongest isolation, no cross-task
  contamination, and simple cleanup. It pays a container start for each graded output.
- **Reuse one sandbox for several execs:** create once, run multiple
  `sb.exec.aio(...)` calls (e.g. all samples of one task, or one eval run), then
  terminate. It's cheaper and faster, but state leaks between execs (files, background
  processes). Use only for outputs you'd trust to share a box. Give each exec a fresh
  directory, `remove(..., recursive=True)` it afterwards, and keep a per-exec `timeout`.
  `idle_timeout` is activity-based (a running exec, stdin writes, or an open tunnel
  connection counts as activity), so an idle reused sandbox gets reaped.
- **Warm pool** (Modal example `sandbox_pool`):
  - Pre-create N sandboxes (stored by `object_id`, e.g. in a `modal.Queue`) and
    re-attach with `Sandbox.from_id`.
  - Track `expires_at` yourself, because Modal doesn't expose remaining lifetime.
  - Health-check before handing one out, and terminate pooled sandboxes that fail
    readiness, or they run untracked until timeout.
  - Don't set `idle_timeout` on pooled sandboxes (they're idle by design).
  - Readiness: `readiness_probe=modal.Probe.with_exec("cmd", ...)` or
    `modal.Probe.with_tcp(port)`, then `await sb.wait_until_ready.aio(timeout=...)`.
- **Snapshots:**
  - `await sb.snapshot_filesystem.aio(ttl=...)` returns an `Image`. Use it as `image=`
    for new sandboxes (forking from a prepared state). The default TTL is 30 days, and
    `ttl=None` keeps it indefinitely. Volumes aren't included.
  - `snapshot_directory(path)` + `mount_image(path, image)` / `unmount_image(path)`
    work on a single directory.
  - Memory snapshots are `_experimental_` only (they terminate the sandbox and expire
    after 7 days). Avoid them.
  - For this project, a published named image is usually the better optimization.
- **Concurrency:**
  - The parallel ceiling is your plan's concurrent container limit. The Harbor evals
    example runs 8 by default and scales to `--n-parallel 100`.
  - Bound fan-out in the app with an `asyncio.Semaphore` around `run_python` so a large
    eval run doesn't hit limits.
  - Combine `tags={"run_id": ...}` on create with `Sandbox.list(tags=...)` to find and
    terminate stragglers for a run.
  - When managing many handles, `await sb.detach.aio()` releases client-side
    connections. It doesn't terminate the sandbox.
- **Cost** (sandbox-resources page):
  - Billing is per second on `max(request, actual usage)`. `cpu` is in physical cores
    (1 core = 2 vCPU), and `memory` is in MiB.
  - Request roughly p50-75 of observed CPU and p90-95 of memory, and let bursting cover
    peaks. Confirm with the App's Sandboxes metrics tab (CPU Pressure, Exit Reasons).
  - Disk size can't be increased.
  - GPU sandboxes need `runtime="gvisor"` and can be preempted. They aren't needed here.

## 8. Testing

Tests never touch Modal. `tests/test_integration.py` monkeypatches the module attribute
that `evaluator` calls through (`sandbox.run_python`, via `from app.services import sandbox`):

```python
from app.schemas import SandboxResult
from app.services import sandbox


async def run_python(code, timeout=None):
    ok = "return a + b" in code
    return SandboxResult(stdout="", stderr="" if ok else "AssertionError", exit_code=0 if ok else 1)


monkeypatch.setattr(sandbox, "run_python", run_python)
```

- Keep `evaluator.py` calling `sandbox.run_python(...)` through the module, not
  `from app.services.sandbox import run_python`, or the monkeypatch won't take effect.
- If you change `run_python`'s signature or `SandboxResult` fields, update the fakes in
  `tests/` in the same change.
- To unit-test sandbox.py's own logic (exit-code mapping, cleanup in `finally`), fake
  `modal.Sandbox.create.aio` with a stub object exposing `exec.aio`, `terminate.aio`,
  `filesystem.write_text.aio`, etc. Assert that `terminate` is called even when `exec`
  raises.
- Real-Modal smoke checks belong in a manually run script, not in pytest, and need a
  token (section 2).

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `AuthError` / "token missing" from the FastAPI process but the CLI works | Check that both `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` are set (the `.env` bridge in `_get_app()` needs both), or that `~/.modal.toml` exists. Code that calls Modal before `_get_app()` bypasses the bridge |
| First sandbox after an image change is slow | Image rebuild on first `create`. Pre-build and publish a named image, and use `modal.enable_output()` to see the logs |
| Exit code `-1`, truncated or empty output | Exec `timeout` hit. Raise `timeout`/`SANDBOX_TIMEOUT_SECONDS` or fix the infinite loop |
| Exit code `137` | SIGKILL: memory limit exceeded or sandbox terminated. Raise the `memory` limit or check the dashboard Exit Reasons |
| `SandboxTimeoutError` from `sb.wait()` | Sandbox lifetime `timeout` shorter than the work. Keep `create(timeout=)` above the exec timeout |
| `ModuleNotFoundError` inside the sandbox | Package not in `SANDBOX_IMAGE`. Add it to `pip_install` (pip can't run at exec time with the network blocked) |
| Generated code fails on network/DNS | Expected with `block_network=True`. Don't loosen it for graders |
| `AttributeError: 'Sandbox' object has no attribute 'open'` | Old API. Use `sb.filesystem.write_text/read_text/...` |
| `InvalidError` combining `block_network` with an allowlist or ports | `block_network=True` can't be combined with `outbound_*`/`inbound_cidr_allowlist` or `*_ports`. Pick one |
| Sandboxes still running in the dashboard after errors | Missing `finally: terminate`. Clean up with `Sandbox.list(app_id=..., tags=...)` plus `terminate`, or `modal app stop` / `modal container stop` |
| Event loop stalls under load | Sync Modal calls in async code. Switch to `.aio` |
| `SandboxFilesystemFileTooLargeError` | Read limit exceeded. Write smaller outputs or stream via exec stdout |

## 10. Sources

Compiled 2026-10-17 against `modal` 1.6.2 (installed in `.venv`).

- https://modal.com/docs/guide/sandbox
- https://modal.com/docs/guide/sandbox-spawn
- https://modal.com/docs/guide/sandbox-files
- https://modal.com/docs/guide/sandbox-networking
- https://modal.com/docs/guide/sandbox-snapshots
- https://modal.com/docs/guide/sandbox-resources
- https://modal.com/docs/guide/secrets
- https://modal.com/docs/sdk/py/latest/Sandbox
- https://modal.com/docs/examples/sandbox_pool
- https://modal.com/docs/examples/harbor_evals
- SDK source: `.venv/lib/python3.12/site-packages/modal/container_process.py` (exit-code semantics), `modal/config.py` (token env vars)
