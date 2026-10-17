from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), extra="ignore")

    # Neon Postgres. `neon env pull` writes a pooled DATABASE_URL (app) and a direct
    # DATABASE_URL_UNPOOLED (migrations).
    database_url: str = "postgresql://localhost/crusoe_hack"
    database_url_unpooled: str = ""

    # Neon Object Storage (S3 compatible)
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_endpoint_url_s3: str = ""
    aws_region: str = "us-east-2"
    s3_bucket: str = "artifacts"

    # Crusoe inference
    crusoe_api_key: str = ""
    crusoe_base_url: str = "https://api.inference.crusoecloud.com/v1"
    crusoe_default_model: str = "deepseek-ai/Deepseek-V4-Flash"
    crusoe_judge_model: str = "openai/gpt-oss-120b"
    # SDK retries 429/5xx with exponential backoff (honours Retry-After)
    crusoe_max_retries: int = 6

    # Modal sandboxes (also read from ~/.modal.toml if unset)
    modal_token_id: str = ""
    modal_token_secret: str = ""
    modal_app_name: str = "crusoe-hack-sandboxes"
    sandbox_timeout_seconds: int = 120

    @property
    def migration_database_url(self) -> str:
        return self.database_url_unpooled or self.database_url


def asyncpg_engine_args(url: str) -> tuple[str, dict]:
    """Turn a Neon connection string into (SQLAlchemy URL, connect_args) for asyncpg.

    Neon URLs carry `?sslmode=require&channel_binding=require`, which asyncpg rejects, so
    strip the query and pass SSL via connect_args. The `-pooler` host is PgBouncer in
    transaction mode, where asyncpg's prepared-statement caches must be off.
    """
    parts = urlsplit(url)
    connect_args: dict = {}
    if "neon.tech" in parts.netloc or "sslmode=require" in parts.query:
        connect_args["ssl"] = "require"
    query = ""
    if "-pooler" in parts.netloc:
        connect_args["statement_cache_size"] = 0
        query = "prepared_statement_cache_size=0"
    return urlunsplit(("postgresql+asyncpg", parts.netloc, parts.path, query, "")), connect_args


@lru_cache
def get_settings() -> Settings:
    return Settings()
