"""Service desk API contract tests."""

import json
import uuid

import pytest
from psycopg.types.json import Jsonb

from backend.api import main
from backend.api.main import _recent, publish


# --- GET /employees/{id} -----------------------------------------------------


def test_get_employee_returns_seeded_row(client):
    r = client.get("/employees/E1042")
    assert r.status_code == 200
    assert r.json() == {
        "id": "E1042",
        "name": "Priya N",
        "email": "priya.n@corp.com",
        "locked": False,
    }


def test_get_unknown_employee_is_404(client):
    assert client.get("/employees/E9999").status_code == 404


def test_seed_includes_a_locked_employee(client):
    assert client.get("/employees/E1045").json()["locked"] is True


# --- GET /tickets ------------------------------------------------------------


def test_open_tickets_excludes_closed(client):
    tickets = client.get("/tickets", params={"employee_id": "E1042"}).json()["tickets"]
    assert {t["id"] for t in tickets} == {"INC0041", "INC0042"}
    assert all(t["status"] == "open" for t in tickets)


def test_status_all_includes_closed(client):
    tickets = client.get(
        "/tickets", params={"employee_id": "E1042", "status": "all"}
    ).json()["tickets"]
    assert "INC0043" in {t["id"] for t in tickets}


def test_tickets_for_employee_with_none(client):
    assert client.get("/tickets", params={"employee_id": "E9999"}).json() == {"tickets": []}


# --- POST /tickets -----------------------------------------------------------


def test_create_ticket_returns_201_and_is_listed(client):
    r = client.post(
        "/tickets",
        json={"employee_id": "E1046", "title": "Mouse not working", "description": "left click dead"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "open"
    ticket_id = body["id"]

    listed = client.get("/tickets", params={"employee_id": "E1046"}).json()["tickets"]
    assert ticket_id in {t["id"] for t in listed}


def test_create_ticket_for_unknown_employee_is_404(client):
    r = client.post("/tickets", json={"employee_id": "E9999", "title": "x"})
    assert r.status_code == 404


# --- POST /password-reset ----------------------------------------------------


def test_password_reset_returns_202_with_masked_recipient(client):
    r = client.post(
        "/password-reset",
        json={"employee_id": "E1042", "idempotency_key": str(uuid.uuid4())},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "sent"
    assert body["sent_to"] == "p***@corp.com"
    assert body["reset_id"].startswith("PR-")


def test_same_idempotency_key_returns_same_reset_id(client):
    """Plan section 4: this is what makes barge-in-during-tool-call safe."""
    key = str(uuid.uuid4())
    payload = {"employee_id": "E1043", "idempotency_key": key}

    first = client.post("/password-reset", json=payload)
    second = client.post("/password-reset", json=payload)

    assert first.status_code == 202
    assert second.status_code == 200
    assert first.json()["reset_id"] == second.json()["reset_id"]

    resets = client.get("/_debug/state").json()["resets"]
    assert sum(1 for r in resets if r["idempotency_key"] == key) == 1


def test_different_keys_create_different_resets(client):
    a = client.post("/password-reset", json={"employee_id": "E1044", "idempotency_key": str(uuid.uuid4())})
    b = client.post("/password-reset", json={"employee_id": "E1044", "idempotency_key": str(uuid.uuid4())})
    assert a.json()["reset_id"] != b.json()["reset_id"]


def test_password_reset_for_unknown_employee_is_404(client):
    r = client.post(
        "/password-reset", json={"employee_id": "E9999", "idempotency_key": str(uuid.uuid4())}
    )
    assert r.status_code == 404


def test_missing_idempotency_key_is_rejected(client):
    assert client.post("/password-reset", json={"employee_id": "E1042"}).status_code == 422


# --- /_debug/events ----------------------------------------------------------


@pytest.fixture
def short_stream(monkeypatch):
    """Breaking out of iter_lines does not cancel the generator, so the request
    is not finished until the deadline. One second instead of thirty."""
    monkeypatch.setattr(main, "MAX_STREAM_SECONDS", 1)


def _read_events(client, headers=None, want=1):
    """Pull `want` data lines off the SSE feed, then hang up."""
    seen = []
    with client.stream("GET", "/_debug/events", headers=headers or {}) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("data: "):
                seen.append(json.loads(line[6:]))
                if len(seen) >= want:
                    break
    return seen


def test_events_replay_carries_a_sequence_id(client, short_stream):
    publish({"kind": "test-marker", "n": 1})
    assert _read_events(client)[-1]["seq"] > 0


def test_last_event_id_skips_what_the_client_already_saw(client, short_stream):
    publish({"kind": "test-seen", "n": 1})
    already = _recent[-1]["seq"]
    publish({"kind": "test-fresh", "n": 2})

    # Without the header the client would replay the buffer and show both again.
    got = _read_events(client, headers={"Last-Event-ID": str(already)})
    assert [e["kind"] for e in got] == ["test-fresh"]


# --- /_debug/integrations ----------------------------------------------------


def test_integrations_names_the_missing_variable(client, monkeypatch):
    """A user id alone is not an endpoint: the server id is what is missing."""
    for var in ("COMPOSIO_MCP_URL", "COMPOSIO_MCP_SERVER_ID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("COMPOSIO_USER_ID", "voiceagent-user")
    monkeypatch.setenv("COMPOSIO_API_KEY", "ak_secret_value")

    composio = next(
        i for i in client.get("/_debug/integrations").json()["integrations"]
        if i["id"] == "composio"
    )
    assert composio["connected"] is False
    assert composio["missing"] == ["COMPOSIO_MCP_SERVER_ID"]
    assert "voiceagent-user" in composio["detail"]


def test_integrations_never_returns_a_key_value(client, monkeypatch):
    """This endpoint is open on localhost and the UI renders it verbatim."""
    monkeypatch.setenv("COMPOSIO_API_KEY", "ak_secret_value")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly_secret_value")
    monkeypatch.setenv("COMPOSIO_MCP_URL", "https://backend.composio.dev/v3/mcp/abc")

    body = client.get("/_debug/integrations").text
    assert "ak_secret_value" not in body
    assert "tvly_secret_value" not in body


def test_saving_a_provider_without_its_key_is_refused(client, monkeypatch):
    """Otherwise the panel accepts it and the failure surfaces as a dead call."""
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    r = client.put("/_debug/agents", json={
        "id": "research", "provider": "nvidia", "model": "nvidia/nemotron-nano-9b-v2",
    })
    assert r.status_code == 400
    assert "NVIDIA_API_KEY" in r.json()["error"]


def test_saving_a_provider_with_its_key_is_allowed(client, monkeypatch, tmp_path):
    from backend.core import personas

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    r = client.put("/_debug/agents", json={"id": "research", "provider": "nvidia"})
    assert r.status_code == 200


def test_a_partial_save_does_not_wipe_the_rest(client, store):
    """A PUT of {id, model} used to drop that agent's purpose, skills and tools.

    The store merges rather than replaces, but this stays an endpoint test
    because that is the layer the bug reached the user through.
    """
    import asyncio

    asyncio.run(store.upsert_agent("research", {
        "label": "Research", "purpose": "Look things up.",
        "skills": ["citing-sources"], "tools": ["search_web", "end_call"],
    }))

    assert client.put(
        "/_debug/agents", json={"id": "research", "model": "gemini-2.5-flash"}
    ).status_code == 200

    saved = asyncio.run(store.agent_overrides())["research"]
    assert saved["model"] == "gemini-2.5-flash"
    assert saved["purpose"] == "Look things up."
    assert saved["skills"] == ["citing-sources"]
    assert saved["tools"] == ["search_web", "end_call"]


# --- /_debug/logs and /_debug/analytics ---------------------------------------


def test_logs_returns_only_what_the_client_has_not_seen(client):
    """The poll is only cheap if `since` actually narrows it."""
    publish({"kind": "probe-old"})
    seen = _recent[-1]["seq"]
    publish({"kind": "probe-new"})

    body = client.get("/_debug/logs", params={"since": seen}).json()
    assert [e["kind"] for e in body["events"]] == ["probe-new"]
    assert body["seq"] >= seen + 1


@pytest.fixture
def traced():
    """Insert call rows directly, and take them out again afterwards.

    A plain synchronous connection on purpose: the client fixture already holds
    an async pool bound to its own event loop, and borrowing it from a sync test
    fails with "attached to a different loop".
    """
    import psycopg

    from backend.store.pool import direct_dsn

    made: list[str] = []

    def write(call_id, meta, rows):
        made.append(call_id)
        with psycopg.connect(direct_dsn(), connect_timeout=30, autocommit=True) as conn:
            conn.execute(
                "INSERT INTO calls (id, agent, model, provider) VALUES (%s,%s,%s,%s)"
                " ON CONFLICT (id) DO NOTHING",
                (call_id, meta.get("agent"), meta.get("model"), meta.get("provider")),
            )
            for row in rows:
                payload = {k: v for k, v in row.items() if k not in ("stage", "ms")}
                conn.execute(
                    "INSERT INTO call_events (call_id, stage, ms, payload)"
                    " VALUES (%s,%s,%s,%s)",
                    (call_id, row["stage"], row.get("ms"), Jsonb(payload)),
                )

    yield write

    with psycopg.connect(direct_dsn(), connect_timeout=30, autocommit=True) as conn:
        for call_id in made:
            conn.execute("DELETE FROM calls WHERE id = %s", (call_id,))


def test_analytics_groups_calls_by_the_model_that_produced_them(client, traced):
    call_id = f"t-{uuid.uuid4().hex[:8]}"
    traced(call_id, {"model": "model-a", "provider": "google", "agent": "it-support"}, [
        {"stage": "user.speech", "ms": 900, "start": 100.0, "end": 101.0},
        {"stage": "tts.first_byte", "ms": 200, "start": 101.2, "end": 101.4},
        {"stage": "llm.first_token", "ms": 640},
    ])

    body = client.get("/_debug/analytics").json()
    entry = next(m for m in body["models"] if m["model"] == "model-a")
    assert entry["calls"] == 1
    assert entry["agents"] == ["it-support"]
    assert entry["stats"]["first_token"]["p50"] == 640
    # 101.2 + 0.2s of ttfb, measured from the 101.0 end of speech = 400ms.
    assert entry["stats"]["reply"]["p50"] == 400.0


def test_call_detail_returns_the_conversation_and_its_timings(client, traced):
    """The drill-down the four separate views could not answer: what happened
    on this particular call."""
    call_id = f"t-{uuid.uuid4().hex[:8]}"
    traced(call_id, {"model": "model-d", "agent": "it-support", "provider": "nvidia"}, [
        {"stage": "turn.user", "ms": None, "turn": 1, "text": "what are my tickets",
         "intent": "get_tickets"},
        {"stage": "turn.tool_call", "ms": None, "turn": 1, "tool": "get_tickets",
         "args": {"employee_id": "E1042"}, "result": "2 open", "success": True,
         "args_valid": True},
        {"stage": "turn.agent", "ms": None, "turn": 1, "text": "You have two open tickets."},
        {"stage": "user.speech", "ms": 900, "start": 10.0, "end": 11.0},
        {"stage": "tts.first_byte", "ms": 250, "start": 11.3, "end": 11.6},
        {"stage": "llm.first_token", "ms": 640},
    ])

    body = client.get(f"/_debug/calls/{call_id}").json()
    assert body["agent"] == "it-support"
    assert [t["who"] for t in body["transcript"]] == ["caller", "agent"]
    assert body["tools"][0]["tool"] == "get_tickets"
    assert body["tools"][0]["success"] is True
    # speech ends at 11.0, audio starts at 11.3 plus 250ms of ttfb = 550ms
    assert body["reply_ms"]["p50"] == 550.0
    assert body["reply_ms"]["over_budget"] == 0
    assert body["timings"]["llm.first_token"]["p50"] == 640.0


def test_call_detail_404s_for_an_unknown_call(client):
    assert client.get("/_debug/calls/does-not-exist").status_code == 404


def test_quality_analytics_reads_events_from_postgres(client, traced):
    """The endpoint returns early when there are no quality events, so the
    smoke test passed while the populated path raised KeyError on a call record
    built with the wrong key. This exercises the path that actually computes."""
    call_id = f"t-{uuid.uuid4().hex[:8]}"
    traced(call_id, {"model": "model-q", "agent": "it-support"}, [
        {"stage": "turn.user", "ms": None, "turn": 1, "text": "my tickets?",
         "intent": "get_tickets"},
        {"stage": "turn.tool_call", "ms": None, "turn": 1, "tool": "get_tickets",
         "args": {"employee_id": "E1042"}, "result": "2 tickets",
         "success": True, "args_valid": True},
    ])

    body = client.get("/_debug/quality-analytics").json()
    assert body["calls_with_quality_data"] >= 1
    mine = next(c for c in body["per_call"] if c["filename"] == call_id)
    assert mine["model"] == "model-q"
    assert body["aggregate"]["task_success_rate"]["value"] is not None


def test_analytics_does_not_invent_a_model_for_unattributed_calls(client, traced):
    """A call whose metadata never landed - the worker died before writing it -
    must not have its numbers filed under some real model's name."""
    call_id = f"t-{uuid.uuid4().hex[:8]}"
    traced(call_id, {}, [{"stage": "llm.first_token", "ms": 700}])

    body = client.get("/_debug/analytics").json()
    assert "unattributed" in [m["model"] for m in body["models"]]
