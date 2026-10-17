"""Neon Object Storage client. Neon buckets speak the S3 API (path-style, SigV4 only)."""

import asyncio
import json
from functools import lru_cache
from typing import Any

import boto3
from botocore.config import Config

from app.config import get_settings


@lru_cache
def _client():
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.aws_endpoint_url_s3 or None,
        region_name=s.aws_region,
        aws_access_key_id=s.aws_access_key_id or None,
        aws_secret_access_key=s.aws_secret_access_key or None,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def _bucket() -> str:
    return get_settings().s3_bucket


async def put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    await asyncio.to_thread(
        _client().put_object, Bucket=_bucket(), Key=key, Body=data, ContentType=content_type
    )
    return key


async def get_bytes(key: str) -> bytes:
    obj = await asyncio.to_thread(_client().get_object, Bucket=_bucket(), Key=key)
    return await asyncio.to_thread(obj["Body"].read)


async def put_jsonl(key: str, rows: list[dict[str, Any]]) -> str:
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode()
    return await put_bytes(key, body, "application/x-ndjson")


async def get_jsonl(key: str) -> list[dict[str, Any]]:
    raw = (await get_bytes(key)).decode()
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


async def list_keys(prefix: str = "") -> list[str]:
    resp = await asyncio.to_thread(_client().list_objects_v2, Bucket=_bucket(), Prefix=prefix)
    return [o["Key"] for o in resp.get("Contents", [])]


async def presigned_url(key: str, expires_in: int = 3600) -> str:
    return await asyncio.to_thread(
        _client().generate_presigned_url,
        "get_object",
        Params={"Bucket": _bucket(), "Key": key},
        ExpiresIn=expires_in,
    )


async def ping() -> bool:
    await asyncio.to_thread(_client().head_bucket, Bucket=_bucket())
    return True
