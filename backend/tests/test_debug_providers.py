"""The console must not be able to offer a provider the worker cannot build."""

import pytest

from backend.core import personas, providers


def test_providers_endpoint_lists_every_kind(client):
    body = client.get("/_debug/providers").json()["providers"]
    assert set(body) == {"llm", "stt", "tts"}
    assert {p["name"] for p in body["llm"]} >= {"google", "nvidia", "openai-compatible"}


def test_providers_report_their_required_variable_never_its_value(client, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "super-secret")
    entry = next(p for p in client.get("/_debug/providers").json()["providers"]["llm"]
                 if p["name"] == "google")
    assert entry["requires"] == "GOOGLE_API_KEY"
    assert entry["key_present"] is True
    assert "super-secret" not in client.get("/_debug/providers").text


def test_every_offered_model_names_a_registered_provider(client):
    """The drift guard. A model whose provider is not in the registry would be
    offered in the picker and fail on the call."""
    for m in client.get("/_debug/models").json()["models"]:
        assert providers.get("llm", m["provider"]) is not None, m["id"]


def test_measurements_survive_the_move_to_the_registry(client):
    """Provider checks recorded that gemini-2.5-flash-lite wrote without asking.
    No API can report that, so it must not be lost in the refactor."""
    models = {m["id"]: m for m in client.get("/_debug/models").json()["models"]}
    assert models["gemini-2.5-flash-lite"]["safe"] is False
    assert models["gemini-3.5-flash-lite"]["ms"] == 695


def test_voices_are_scoped_to_their_tts_provider(client):
    """A Deepgram voice name sent to Riva fails on the call, so the picker must
    not offer them together."""
    aura = client.get("/_debug/voices?provider=deepgram").json()["voices"]
    magpie = client.get("/_debug/voices?provider=nvidia").json()["voices"]
    assert all(v["id"].startswith("aura-2-") for v in aura)
    assert all(v["id"].startswith("Magpie-") for v in magpie)
    assert not {v["id"] for v in aura} & {v["id"] for v in magpie}


def test_voices_default_to_deepgram_for_backwards_compatibility(client):
    assert client.get("/_debug/voices").json()["voices"][0]["id"].startswith("aura-2-")


# --- save-time validation ---------------------------------------------------


@pytest.fixture
def agents_file(monkeypatch):
    """An empty store, so a save cannot touch the real gallery."""
    from backend.store import MemoryStore, set_store

    set_store(MemoryStore())
    yield
    set_store(None)
    personas.apply_overrides({})


def test_save_refuses_an_unknown_stt_provider(client, agents_file):
    r = client.put("/_debug/agents", json={"id": "it-support", "stt_provider": "whisper"})
    assert r.status_code == 400
    assert "whisper" in r.json()["error"]


def test_save_refuses_a_speech_provider_whose_key_is_absent(client, agents_file, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    r = client.put("/_debug/agents", json={"id": "it-support", "tts_provider": "nvidia"})
    assert r.status_code == 400
    assert "NVIDIA_API_KEY" in r.json()["error"]


def test_save_refuses_openai_compatible_without_a_key_variable(client, agents_file):
    r = client.put("/_debug/agents", json={
        "id": "it-support", "provider": "openai-compatible",
        "base_url": "https://api.groq.com/openai/v1",
    })
    assert r.status_code == 400
    assert "api_key_env" in r.json()["error"]


def test_save_refuses_openai_compatible_with_a_bad_base_url(client, agents_file, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gr")
    r = client.put("/_debug/agents", json={
        "id": "it-support", "provider": "openai-compatible",
        "base_url": "not-a-url", "api_key_env": "GROQ_API_KEY",
    })
    assert r.status_code == 400
    assert "base_url" in r.json()["error"]


def test_save_accepts_a_complete_openai_compatible_config(client, agents_file, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gr")
    r = client.put("/_debug/agents", json={
        "id": "it-support", "provider": "openai-compatible",
        "model": "llama-3.3-70b",
        "base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY",
    })
    assert r.status_code == 200


def test_save_refuses_a_voice_the_chosen_provider_does_not_offer(client, agents_file, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nk")
    r = client.put("/_debug/agents", json={
        "id": "it-support", "tts_provider": "nvidia", "voice": "aura-2-thalia-en",
    })
    assert r.status_code == 400
    assert "aura-2-thalia-en" in r.json()["error"]
