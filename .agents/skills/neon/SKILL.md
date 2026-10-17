---
name: neon
description: Neon Postgres + Neon Object Storage (S3-compatible buckets) + branches, as used from Python (SQLAlchemy/asyncpg, boto3, Alembic). Use for DATABASE_URL, pooled vs direct, migrations, buckets/credentials, neon CLI, test branches.
---

# Neon (Postgres, buckets, branches)

Condensed for this Python repo. Full official skills (Apache-2.0, TypeScript-oriented): `references/upstream/{neon,neon-postgres,neon-object-storage,neon-postgres-branches}.md`; read them for neon.ts, auth, search, MCP and diagnostics. Docs as markdown: append `.md` to any neon.com/docs URL; index at https://neon.com/docs/llms.txt.

## Postgres
- Neon URLs look like `postgresql://…/neondb?sslmode=require&channel_binding=require`. asyncpg rejects those params: strip them and pass `connect_args={"ssl": "require"}` (done in `app/config.py` / `app/db.py`).
- **Direct** host for migrations, `pg_dump`, LISTEN/NOTIFY and session state. **Pooled** (`-pooler` host, PgBouncer transaction mode) for many short app connections. With asyncpg behind the pooler, disable statement caches (`prepared_statement_cache_size=0`, `statement_cache_size=0`).
- Computes scale to zero: expect a cold first query, and use `pool_pre_ping=True`.
- Manage the schema only through migrations (Alembic here). Test them on a branch first.

## Object Storage (beta; aws us-east-1/2, eu-central-1, ap-southeast-1)
- S3 API, **path-style + SigV4 only**. boto3: `Config(signature_version="s3v4", s3={"addressing_style": "path"})`.
- Env: `AWS_ACCESS_KEY_ID` (credential `token_id`), `AWS_SECRET_ACCESS_KEY` (`s3_secret_access_key`, shown once), `AWS_ENDPOINT_URL_S3` (branch endpoint), `AWS_REGION`.
- Buckets are **per branch**: a child branch inherits its parent's objects copy-on-write, and its writes never touch the parent. Store object **keys** in Postgres rows on the same branch and presign on read.
- Access is `private` (default) or `public_read`. Put a CDN in front of public assets.
- Not every S3 operation exists; see https://neon.com/docs/storage/s3-compatibility.md.

## CLI (needs the current `neon` CLI: `npm i -g neon`; Homebrew `neonctl` 2.x lacks storage/env/skills)
```bash
neon link --project-id <id> -y              # once; writes .neon, pulls env
neon checkout <branch> --create             # branch per feature/test; pulls that branch's env
neon env pull --file .env -s postgres       # DATABASE_URL (pooled), DATABASE_URL_UNPOOLED, NEON_BRANCH
neon env pull --file .env -s object-storage # AWS_*; issues a NEW credential on every call
neon connection-string <branch>             # direct URL by default
neon credentials create --scope storage:read --scope storage:write
neon bucket create <name> | list | delete ; neon bucket object put|get|list|delete
neon branches create --name t --parent main --expires-at <ISO>   # ephemeral test DB
neon branches reset <branch> --parent       # discard drift, re-copy parent
neon diff                                   # schema diff vs parent
```
Don't run `neon login` unattended. Never point tests or experimental migrations at the main branch. Use an expiring branch.
