"""Neon Auth token verification.

The three mistakes these guard against all fail as "401, no idea why":
wrong algorithm family, JWKS URL pointing at the auth base, and comparing the
issuer against the full URL instead of the origin.
"""

import os

import jwt
import pytest

from backend.api import auth


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setenv("AUTH_URL", "https://ep-test.neonauth.example.com/neondb/auth")
    monkeypatch.delenv("JWKS_URL", raising=False)
    monkeypatch.delenv("JWTKS_URL", raising=False)
    auth.reset_client()
    yield
    auth.reset_client()


def test_jwks_url_is_derived_from_the_auth_url():
    """AUTH_URL alone is what people actually have in .env."""
    assert auth.jwks_url() == (
        "https://ep-test.neonauth.example.com/neondb/auth/.well-known/jwks.json"
    )


def test_a_configured_jwks_url_that_names_a_key_set_is_honoured(monkeypatch):
    monkeypatch.setenv("JWKS_URL", "https://elsewhere.example.com/keys/jwks.json")
    assert auth.jwks_url() == "https://elsewhere.example.com/keys/jwks.json"


def test_a_jwks_url_pointing_at_the_auth_base_is_corrected(monkeypatch):
    """The paste error that started this: JWTKS_URL set equal to AUTH_URL.

    Honouring it verbatim gives a 404 from the key fetch and a 401 on every
    request, with nothing saying the URL is the problem.
    """
    monkeypatch.setenv("JWTKS_URL", "https://ep-test.neonauth.example.com/neondb/auth")
    assert auth.jwks_url().endswith("/.well-known/jwks.json")


def test_issuer_is_the_origin_not_the_full_url():
    """Neon compares `iss` against scheme://host. Including /neondb/auth would
    reject every valid token."""
    assert auth.expected_issuer() == "https://ep-test.neonauth.example.com"


def test_the_algorithm_is_eddsa():
    """Neon Auth signs with Ed25519 (kty OKP). An RS256 default would make
    every token fail verification for a reason nothing reports."""
    assert auth.ALGORITHMS == ["EdDSA"]


def test_unconfigured_auth_is_not_silently_open(monkeypatch):
    monkeypatch.delenv("AUTH_URL", raising=False)
    auth.reset_client()
    assert auth.configured() is False


def test_verify_rejects_a_token_signed_with_the_wrong_key(monkeypatch):
    """End to end over a real Ed25519 keypair, no network: a token this service
    did not issue must not verify."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    attacker = Ed25519PrivateKey.generate()
    ours = Ed25519PrivateKey.generate()

    token = jwt.encode(
        {"sub": "u1", "iss": auth.expected_issuer()},
        attacker,
        algorithm="EdDSA",
    )

    monkeypatch.setattr(
        auth, "_client",
        lambda: type("K", (), {"get_signing_key_from_jwt":
                               staticmethod(lambda t: type("S", (), {"key": ours.public_key()})())})(),
    )
    with pytest.raises(jwt.PyJWTError):
        auth.verify(token)


def test_verify_accepts_a_correctly_signed_token(monkeypatch):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    token = jwt.encode(
        {"sub": "user-42", "iss": auth.expected_issuer()}, key, algorithm="EdDSA"
    )

    monkeypatch.setattr(
        auth, "_client",
        lambda: type("K", (), {"get_signing_key_from_jwt":
                               staticmethod(lambda t: type("S", (), {"key": key.public_key()})())})(),
    )
    assert auth.verify(token)["sub"] == "user-42"


def test_verify_rejects_a_token_from_another_issuer(monkeypatch):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    token = jwt.encode(
        {"sub": "u1", "iss": "https://someone-elses-project.neonauth.example.com"},
        key, algorithm="EdDSA",
    )
    monkeypatch.setattr(
        auth, "_client",
        lambda: type("K", (), {"get_signing_key_from_jwt":
                               staticmethod(lambda t: type("S", (), {"key": key.public_key()})())})(),
    )
    with pytest.raises(jwt.InvalidIssuerError):
        auth.verify(token)


@pytest.mark.skipif(
    not os.getenv("AUTH_URL_LIVE"), reason="set AUTH_URL_LIVE=1 to hit the real JWKS"
)
def test_the_real_jwks_endpoint_serves_an_eddsa_key():
    """Opt-in live check against the configured Neon Auth project."""
    import httpx
    from dotenv import load_dotenv

    # override=True matters: the autouse fixture above has already set a fake
    # AUTH_URL, and load_dotenv leaves existing variables alone by default.
    load_dotenv(".env", override=True)
    auth.reset_client()
    body = httpx.get(auth.jwks_url(), timeout=15).json()
    assert body["keys"], "JWKS served no keys"
    assert any(k.get("alg") == "EdDSA" or k.get("kty") == "OKP" for k in body["keys"])
