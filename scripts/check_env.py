"""Verify that .env is complete and every service works with it.

    uv run python scripts/check_env.py [--skip-sandbox]

Exits non-zero (listing what's wrong) unless Postgres, the Neon bucket, Crusoe and Modal
all work with the values in .env. Uses the app's own clients, so passing here means the
app will work.
"""

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REQUIRED = {
    "Neon Postgres": ["DATABASE_URL", "DATABASE_URL_UNPOOLED"],
    "Neon bucket": [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ENDPOINT_URL_S3",
        "AWS_REGION",
        "S3_BUCKET",
    ],
    "Crusoe": ["CRUSOE_API_KEY", "CRUSOE_BASE_URL", "CRUSOE_DEFAULT_MODEL", "CRUSOE_JUDGE_MODEL"],
    "Modal": ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"],
}

OK, FAIL = "  \033[32m✓\033[0m", "  \033[31m✗\033[0m"


async def check_postgres() -> str:
    from sqlalchemy import text

    from app.config import get_settings
    from app.db import engine, make_engine

    async with engine.connect() as conn:
        await conn.execute(text("select 1"))
    await engine.dispose()
    direct = make_engine(get_settings().migration_database_url)
    async with direct.connect() as conn:
        version = (await conn.execute(text("select version_num from alembic_version"))).scalar()
    await direct.dispose()
    return f"pooled + direct connections work, schema at {version}"


async def check_bucket() -> str:
    from app.services import storage

    key = f"_setup-check/{uuid.uuid4()}.txt"
    await storage.put_bytes(key, b"ok", "text/plain")
    try:
        assert await storage.get_bytes(key) == b"ok", "read back different bytes"
    finally:
        await storage.delete(key)
    return "write/read/delete round trip works"


async def check_crusoe() -> str:
    from app.config import get_settings
    from app.services import inference

    s = get_settings()
    available = set(await inference.list_models())
    missing = [m for m in (s.crusoe_default_model, s.crusoe_judge_model) if m not in available]
    if missing:
        raise RuntimeError(f"model(s) not available to this key: {', '.join(missing)}")
    await inference.chat([{"role": "user", "content": "Reply with: ok"}], max_tokens=512)
    return f"key valid, chat works on {s.crusoe_default_model}"


async def check_modal() -> str:
    from app.services import sandbox

    res = await sandbox.run_python("print('ok')", timeout=60)
    if res.exit_code != 0 or res.stdout.strip() != "ok":
        raise RuntimeError(f"exit {res.exit_code}: {res.stderr[-300:]}")
    return "sandbox ran code"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sandbox", action="store_true", help="don't start a Modal sandbox")
    args = parser.parse_args()

    env_path = Path(".env")
    if not env_path.exists():
        print(f"{FAIL} .env not found. Run scripts/setup.sh")
        return 1
    values = dotenv_values(env_path)
    failures, incomplete = 0, set()
    for service, keys in REQUIRED.items():
        empty = [k for k in keys if not values.get(k)]
        if empty:
            failures += 1
            incomplete.add(service)
            print(f"{FAIL} {service}: missing in .env: {', '.join(empty)}")

    checks = [
        ("Neon Postgres", check_postgres),
        ("Neon bucket", check_bucket),
        ("Crusoe", check_crusoe),
    ]
    if not args.skip_sandbox and "Modal" not in incomplete:
        print("  … Modal: starting a sandbox (the first run builds the image, ~1-2 min)")
        checks.append(("Modal", check_modal))
    for name, check in checks:
        if name in incomplete:
            continue
        try:
            print(f"{OK} {name}: {await check()}")
        except Exception as e:  # report every failing service, not just the first
            failures += 1
            print(f"{FAIL} {name}: {type(e).__name__}: {e}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
