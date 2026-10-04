"""Composio wiring needs two values, and half of it must not fail quietly."""

import pytest

from backend.worker import main


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("COMPOSIO_MCP_URL", "COMPOSIO_API_KEY", "COMPOSIO_ALLOWED_TOOLS",
                "EXTRA_MCP_URLS", "COMPOSIO_MCP_SERVER_ID", "COMPOSIO_USER_ID"):
        monkeypatch.delenv(var, raising=False)


def test_nothing_configured_attaches_nothing():
    assert main.build_mcp_servers() == []


def test_key_without_url_warns_instead_of_silently_doing_nothing(monkeypatch, caplog):
    """The failure this prevents: a key is set, no tools appear, and the logs
    say nothing at all about why."""
    monkeypatch.setenv("COMPOSIO_API_KEY", "ak_test")
    with caplog.at_level("WARNING"):
        assert main.build_mcp_servers() == []
    assert "COMPOSIO_MCP_URL" in caplog.text


def test_url_without_key_fails_loudly(monkeypatch):
    monkeypatch.setenv("COMPOSIO_MCP_URL", "https://backend.composio.dev/v3/mcp/abc")
    with pytest.raises(RuntimeError, match="COMPOSIO_API_KEY"):
        main.build_mcp_servers()


def test_both_values_attach_one_server(monkeypatch):
    monkeypatch.setenv("COMPOSIO_MCP_URL", "https://backend.composio.dev/v3/mcp/abc?user_id=u1")
    monkeypatch.setenv("COMPOSIO_API_KEY", "ak_test")
    assert len(main.build_mcp_servers()) == 1


def test_extra_servers_are_attached_alongside(monkeypatch):
    monkeypatch.setenv("EXTRA_MCP_URLS", "https://a.example/mcp, https://b.example/mcp")
    assert len(main.build_mcp_servers()) == 2


def test_blank_entries_in_the_extra_list_are_ignored(monkeypatch):
    monkeypatch.setenv("EXTRA_MCP_URLS", " , https://a.example/mcp ,, ")
    assert len(main.build_mcp_servers()) == 1


# --- Composio URL assembly ----------------------------------------------------


def test_server_id_plus_user_id_builds_the_url(monkeypatch):
    """The pieces are easier to paste correctly than a URL with a query string."""
    monkeypatch.setenv("COMPOSIO_MCP_SERVER_ID", "srv_abc123")
    monkeypatch.setenv("COMPOSIO_USER_ID", "voiceagent-user")
    assert main.composio_mcp_url() == (
        "https://backend.composio.dev/v3/mcp/srv_abc123?user_id=voiceagent-user"
    )


def test_a_full_url_gains_the_user_id(monkeypatch):
    monkeypatch.setenv("COMPOSIO_MCP_URL", "https://backend.composio.dev/v3/mcp/srv_x")
    monkeypatch.setenv("COMPOSIO_USER_ID", "voiceagent-user")
    assert main.composio_mcp_url().endswith("/srv_x?user_id=voiceagent-user")


def test_an_existing_user_id_is_not_doubled(monkeypatch):
    """Two user_id params is a request Composio cannot resolve unambiguously."""
    monkeypatch.setenv("COMPOSIO_MCP_URL", "https://backend.composio.dev/v3/mcp/s?user_id=someone")
    monkeypatch.setenv("COMPOSIO_USER_ID", "voiceagent-user")
    assert main.composio_mcp_url().count("user_id=") == 1


def test_user_id_is_url_encoded(monkeypatch):
    monkeypatch.setenv("COMPOSIO_MCP_SERVER_ID", "s")
    monkeypatch.setenv("COMPOSIO_USER_ID", "a b&c")
    assert main.composio_mcp_url().endswith("?user_id=a%20b%26c")


def test_a_user_id_alone_configures_nothing(monkeypatch):
    """Without a server id there is no endpoint, so this must stay off rather
    than build a URL that 404s."""
    monkeypatch.setenv("COMPOSIO_USER_ID", "voiceagent-user")
    monkeypatch.setenv("COMPOSIO_API_KEY", "ak_test")
    assert main.composio_mcp_url() == ""
    assert main.build_mcp_servers() == []
