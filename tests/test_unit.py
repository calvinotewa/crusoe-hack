from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.schemas import Task
from app.services import evaluator


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_neon_url_is_converted_for_asyncpg():
    s = Settings(
        database_url="postgresql://u:p@ep-x.us-east-2.aws.neon.tech/neondb"
        "?sslmode=require&channel_binding=require"
    )
    assert s.async_database_url == "postgresql+asyncpg://u:p@ep-x.us-east-2.aws.neon.tech/neondb"
    assert s.database_requires_ssl


def test_extract_code():
    assert evaluator.extract_code("Sure:\n```python\nx = 1\n```\n") == "x = 1\n"
    assert evaluator.extract_code("x = 1") == "x = 1"


async def test_exact_grader():
    task = Task(id="t1", prompt="2+2?", grader="exact", expected="4")
    assert (await evaluator.grade(task, " 4\n"))["score"] == 1.0
    assert (await evaluator.grade(task, "5"))["score"] == 0.0
