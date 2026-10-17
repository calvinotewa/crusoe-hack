from fastapi.testclient import TestClient

from app.config import Settings, asyncpg_engine_args
from app.main import app
from app.schemas import Task
from app.services import evaluator
from app.services.storage import signing_region


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_neon_url_is_converted_for_asyncpg():
    url, args = asyncpg_engine_args(
        "postgresql://u:p@ep-x.us-east-2.aws.neon.tech/neondb"
        "?sslmode=require&channel_binding=require"
    )
    assert url == "postgresql+asyncpg://u:p@ep-x.us-east-2.aws.neon.tech/neondb"
    assert args == {"ssl": "require"}


def test_pooled_neon_url_disables_statement_caches():
    url, args = asyncpg_engine_args(
        "postgresql://u:p@ep-x-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"
    )
    assert url.endswith("/neondb?prepared_statement_cache_size=0")
    assert args == {"ssl": "require", "statement_cache_size": 0}


def test_migrations_prefer_unpooled_url():
    s = Settings(database_url="postgresql://pooled", database_url_unpooled="postgresql://direct")
    assert s.migration_database_url == "postgresql://direct"


def test_extract_code():
    assert evaluator.extract_code("Sure:\n```python\nx = 1\n```\n") == "x = 1\n"
    assert evaluator.extract_code("x = 1") == "x = 1"


async def test_exact_grader():
    task = Task(id="t1", prompt="2+2?", grader="exact", expected="4")
    assert (await evaluator.grade(task, " 4\n"))["score"] == 1.0
    assert (await evaluator.grade(task, "5"))["score"] == 0.0


def test_signing_region():
    endpoint = "https://br-x.storage.c-2.us-east-2.aws.neon.tech"
    assert signing_region("eu-central-1", endpoint) == "eu-central-1"
    assert signing_region("c-2.us-east-2.aws.neon.tech", "") == "us-east-2"
    assert signing_region("", endpoint) == "us-east-2"
    assert signing_region("", "") == "us-east-2"
