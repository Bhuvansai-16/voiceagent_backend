"""Neon Auth: verifying the caller's JWT.

Three things here are easy to get wrong, and all three fail in ways that look
like something else.

**The algorithm is EdDSA, not RS256.** Neon Auth signs with Ed25519 (`kty: OKP`).
Almost every FastAPI JWT example hardcodes RS256; passing the wrong algorithm
list makes every token look invalid. PyJWT needs its `crypto` extra for Ed25519.

**The JWKS URL is not the auth URL.** It is `<AUTH_URL>/.well-known/jwks.json`.
Pointing at AUTH_URL itself returns a 404 JSON body, which `PyJWKClient` reports
as a key-fetch failure rather than a configuration mistake.

**The issuer is the origin, not the full URL.** Neon's docs check `iss` against
`scheme://host` of the auth URL, dropping the `/neondb/auth` path. Comparing
against the full URL rejects every valid token.

Keys are cached by PyJWKClient, so verification does not make a network call per
request — only on a key id it has not seen.
"""

import logging
import os
from urllib.parse import urlsplit

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

logger = logging.getLogger("backend.api.auth")

ALGORITHMS = ["EdDSA"]

_jwk_client: PyJWKClient | None = None


def auth_base_url() -> str:
    return os.getenv("AUTH_URL", "").strip().rstrip("/")


def jwks_url() -> str:
    """`JWKS_URL` if it actually points at a key set, else derived from AUTH_URL.

    The derivation is not a convenience: a value that names the auth base rather
    than the key set is a common paste error, and honouring it verbatim turns
    every request into a 401 with no useful message.
    """
    configured = (os.getenv("JWKS_URL") or os.getenv("JWTKS_URL") or "").strip()
    if configured.endswith(("jwks.json", "/jwks")):
        return configured
    base = auth_base_url()
    return f"{base}/.well-known/jwks.json" if base else ""


def expected_issuer() -> str:
    """Origin only. Neon's docs compare `iss` against scheme://host."""
    parts = urlsplit(auth_base_url())
    return f"{parts.scheme}://{parts.netloc}" if parts.netloc else ""


def configured() -> bool:
    return bool(jwks_url())


def _client() -> PyJWKClient:
    global _jwk_client
    if _jwk_client is None:
        _jwk_client = PyJWKClient(jwks_url(), cache_keys=True, lifespan=3600)
    return _jwk_client


def reset_client() -> None:
    """Drop the cached JWKS client. For tests that change AUTH_URL."""
    global _jwk_client
    _jwk_client = None


def verify(token: str) -> dict:
    """Decoded claims, or raise `jwt.PyJWTError`."""
    signing_key = _client().get_signing_key_from_jwt(token).key
    return jwt.decode(
        token,
        signing_key,
        algorithms=ALGORITHMS,
        issuer=expected_issuer() or None,
        options={"verify_aud": False},
    )


# --- FastAPI wiring ---------------------------------------------------------

# auto_error=False so a missing header reaches our own handler and produces a
# message that says what is wrong, rather than a bare 403 from the dependency.
_bearer = HTTPBearer(auto_error=False)


async def current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """Require a valid token. Returns the claims; `sub` is the user id."""
    if not configured():
        raise HTTPException(
            503, "Authentication is not configured. Set AUTH_URL in .env."
        )
    if credentials is None or not credentials.credentials:
        raise HTTPException(401, "Missing bearer token.")
    try:
        claims = verify(credentials.credentials)
    except jwt.PyJWTError as exc:
        # The reason goes to the log, not to the caller: telling an unauthorised
        # client whether a token was expired, malformed or signed by the wrong
        # key is more help than it needs.
        logger.info("token rejected: %s", exc)
        raise HTTPException(401, "Invalid or expired token.") from exc
    request.state.user = claims
    return claims


async def optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict | None:
    """Claims when a valid token is present, None otherwise. Never raises.

    For endpoints that should work signed-out but scope data per user when
    someone is signed in.
    """
    if credentials is None or not credentials.credentials or not configured():
        return None
    try:
        claims = verify(credentials.credentials)
    except jwt.PyJWTError:
        return None
    request.state.user = claims
    return claims
