"""Object storage configuration checks.

The live paths need real credentials, so what is worth testing without them is
the thing that actually went wrong: telling a placeholder apart from a key.

A .env carrying `<generated-access-key-id>` is not configured, but every call
still reaches the network and comes back InvalidAccessKeyId, which reads like an
outage rather than a setup step you have not finished.
"""

import pytest

from backend.core import storage


@pytest.fixture(autouse=True)
def reset():
    storage.reset_client()
    yield
    storage.reset_client()


@pytest.fixture
def env(monkeypatch):
    def _set(endpoint="https://x.storage.neon.tech", key="AKIAREAL", secret="realsecret"):
        for name, value in (
            ("AWS_ENDPOINT_URL_S3", endpoint),
            ("AWS_ACCESS_KEY_ID", key),
            ("AWS_SECRET_ACCESS_KEY", secret),
        ):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
    return _set


def test_real_looking_credentials_count_as_configured(env):
    env()
    assert storage.configured() is True
    assert storage.missing() == []


def test_absent_credentials_are_reported_by_name(env):
    env(key=None, secret=None)
    assert storage.configured() is False
    assert "AWS_ACCESS_KEY_ID" in storage.missing()


@pytest.mark.parametrize("placeholder", [
    "<generated-access-key-id>",
    "{{ACCESS_KEY}}",
    "[your-key-here]",
])
def test_placeholder_credentials_are_not_configured(env, placeholder):
    """Both keys in the real .env were 25-character strings starting with '<'.
    Treating those as configured is how a setup step becomes a runtime mystery."""
    env(key=placeholder)
    assert storage.configured() is False
    assert any("placeholder" in m for m in storage.missing())


def test_missing_names_the_placeholder_variable(env):
    env(secret="<generated-secret>")
    assert any(m.startswith("AWS_SECRET_ACCESS_KEY") for m in storage.missing())


@pytest.mark.asyncio
async def test_presign_is_local_and_proves_nothing_about_credentials(env):
    """A presigned URL is an HMAC computed locally, with no network call, so it
    succeeds against credentials that cannot actually read anything. It must
    never be used as a connectivity check."""
    env()
    url = await storage.presign("uploads/file.txt")
    assert url.startswith("https://x.storage.neon.tech/")
    assert "X-Amz-Signature" in url


@pytest.mark.asyncio
async def test_presign_honours_a_custom_expiry(env):
    env()
    assert "X-Amz-Expires=60" in await storage.presign("k", expires_in=60)
