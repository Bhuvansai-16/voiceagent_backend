"""Object storage on Neon's S3-compatible endpoint.

The Python equivalent of the @aws-sdk/client-s3 snippet: `forcePathStyle: true`
becomes `addressing_style: "path"`, and the endpoint comes from
AWS_ENDPOINT_URL_S3 rather than being derived from a region.

boto3 is synchronous. Every public function here is async and dispatches the
blocking call through `asyncio.to_thread`, because these are called from
`async def` request handlers and a blocking socket read there stalls every other
request on the event loop. That is the same class of bug as the synchronous
httpx client this codebase used for OCR.

Uploads are fire-and-store: the original document is kept so a re-index does not
require the caller to upload again, while the extracted text and vectors live in
Postgres and Pinecone.
"""

import asyncio
import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger("service-desk.storage")

BUCKET = os.getenv("S3_BUCKET", "assets")
PRESIGN_TTL = int(os.getenv("S3_PRESIGN_TTL", "3600"))


def configured() -> bool:
    """Whether real credentials are present.

    Checks for placeholder values too. A .env carrying `<generated-access-key>`
    is not configured, but every call still reaches the network and fails with
    InvalidAccessKeyId, which reads like an outage rather than a setup step.
    """
    endpoint = os.getenv("AWS_ENDPOINT_URL_S3", "").strip()
    key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    secret = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    if not (endpoint and key and secret):
        return False
    return not any(v.startswith(("<", "{", "[")) for v in (key, secret))


def missing() -> list[str]:
    out = [
        name
        for name in ("AWS_ENDPOINT_URL_S3", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
        if not os.getenv(name, "").strip()
    ]
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if os.getenv(name, "").strip().startswith(("<", "{", "[")) and name not in out:
            out.append(f"{name} (placeholder, not a real key)")
    return out


@lru_cache(maxsize=1)
def _client():
    """One client for the process. Building one per call adds a TLS handshake."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3"),
        region_name=os.getenv("AWS_REGION") or "us-east-1",
        config=Config(
            # forcePathStyle: true — Neon's endpoint does not do virtual-host
            # style buckets.
            s3={"addressing_style": "path"},
            connect_timeout=10,
            read_timeout=30,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def reset_client() -> None:
    """Drop the cached client. Tests change credentials between cases."""
    _client.cache_clear()


async def put(key: str, body: bytes, content_type: str = "") -> str:
    """Store bytes and return the key. Raises on failure — an upload that
    silently did not happen is worse than one that errors."""
    kwargs: dict[str, Any] = {"Bucket": BUCKET, "Key": key, "Body": body}
    if content_type:
        kwargs["ContentType"] = content_type
    await asyncio.to_thread(lambda: _client().put_object(**kwargs))
    logger.info("stored %s (%d bytes)", key, len(body))
    return key


async def get(key: str) -> bytes:
    def _read() -> bytes:
        return _client().get_object(Bucket=BUCKET, Key=key)["Body"].read()

    return await asyncio.to_thread(_read)


async def presign(key: str, expires_in: int | None = None) -> str:
    """A time-limited URL for reading one object.

    This is computed locally — it is an HMAC over the request, with no network
    call — so a presigned URL comes back successfully even when the credentials
    are wrong. It proves nothing about whether the object exists or the key
    works. Check `configured()` for that.
    """
    def _sign() -> str:
        return _client().generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET, "Key": key},
            ExpiresIn=expires_in or PRESIGN_TTL,
        )

    return await asyncio.to_thread(_sign)


async def delete(key: str) -> None:
    await asyncio.to_thread(lambda: _client().delete_object(Bucket=BUCKET, Key=key))


async def ensure_bucket() -> bool:
    """Create the bucket if it is absent. True if it exists afterwards."""
    def _ensure() -> bool:
        client = _client()
        try:
            client.head_bucket(Bucket=BUCKET)
            return True
        except Exception:
            pass
        try:
            client.create_bucket(Bucket=BUCKET)
            return True
        except Exception as exc:
            logger.warning("could not create bucket %s: %s", BUCKET, exc)
            return False

    return await asyncio.to_thread(_ensure)
