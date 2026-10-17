from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), extra="ignore")

    # Neon Postgres
    database_url: str = "postgresql://localhost/crusoe_hack"

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
    def async_database_url(self) -> str:
        """Neon hands out `postgresql://...?sslmode=require&channel_binding=require`.

        asyncpg doesn't understand those query params, so strip them and pass SSL via
        connect_args instead (see db.py).
        """
        parts = urlsplit(self.database_url)
        scheme = "postgresql+asyncpg"
        return urlunsplit((scheme, parts.netloc, parts.path, "", ""))

    @property
    def database_requires_ssl(self) -> bool:
        return "neon.tech" in self.database_url or "sslmode=require" in self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
