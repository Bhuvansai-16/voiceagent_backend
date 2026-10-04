"""Every no-argument endpoint answers without a server error.

This exists because of a real miss. Splitting the 830-line debug router into
five files left `routers/connectors.py` importing `backend.worker.connectors`,
which had moved to `backend.core`. The whole suite stayed green — no test calls
`/_debug/connectors` — and the endpoint raised ImportError on first request.

A 500 here means a module-level mistake: a missing import, a name that moved, a
router that was never registered. A 4xx is fine and often correct: 503 from an
endpoint whose credentials are absent is an answer, not a crash.
"""

import pytest

# GET endpoints that need no path or query parameters.
GET_PATHS = [
    "/_debug/agents",
    "/_debug/models",
    "/_debug/voices",
    "/_debug/providers",
    "/_debug/logs",
    "/_debug/analytics",
    "/_debug/traces",
    "/_debug/quality-analytics",
    "/_debug/integrations",
    "/_debug/state",
    "/_debug/telephony",
    "/_debug/connectors",
    "/_debug/flows",
    "/_debug/livekit-token",
    "/documents",
]


@pytest.mark.parametrize("path", GET_PATHS)
def test_endpoint_does_not_raise_a_server_error(client, path):
    response = client.get(path)
    assert response.status_code < 500, f"{path} -> {response.status_code}: {response.text[:300]}"


def test_log_ingest_accepts_an_event(client):
    response = client.post("/_debug/log", json={"kind": "smoke", "detail": {"ms": 1}})
    assert response.status_code == 204
