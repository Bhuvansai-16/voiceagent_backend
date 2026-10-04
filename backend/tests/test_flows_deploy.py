"""The flow compiler is the second path that writes agent configuration, and it
dropped speech settings the same way the persona loader did."""

import pytest

from backend.core import personas


@pytest.fixture
def isolated():
    """An empty store. Deploy writes an agent row, not a file."""
    from backend.store import MemoryStore, set_store

    impl = MemoryStore()
    set_store(impl)
    yield impl
    set_store(None)
    personas.apply_overrides({})


def _flow():
    return {
        "id": "flow-abc", "name": "Speechy", "description": "d",
        "nodes": [
            {"type": "agent_brain", "data": {"config": {"purpose": "You answer questions."}}},
            {"type": "voice_input", "data": {"config": {
                "stt_provider": "nvidia", "model": "parakeet-x"}}},
            {"type": "voice_output", "data": {"config": {
                "tts_provider": "nvidia", "voice": "Magpie-Multilingual.EN-US.Leo"}}},
        ],
    }


def test_deploy_carries_speech_config_into_the_agent(client, isolated):
    r = client.post("/_debug/flows/flow-abc/deploy", json=_flow())
    assert r.status_code == 200
    agent = r.json()["agent"]
    assert agent["stt_provider"] == "nvidia"
    assert agent["stt_model"] == "parakeet-x"
    assert agent["tts_provider"] == "nvidia"
    assert agent["voice"] == "Magpie-Multilingual.EN-US.Leo"


@pytest.mark.asyncio
async def test_deploy_persists_speech_config(client, isolated):
    client.post("/_debug/flows/flow-abc/deploy", json=_flow())
    saved = await isolated.agent_overrides()
    assert saved["abc"]["stt_provider"] == "nvidia"


def test_a_flow_without_speech_nodes_omits_the_fields(client, isolated):
    """Absent nodes must not overwrite an existing agent's speech config with
    nulls."""
    flow = _flow()
    flow["nodes"] = [n for n in flow["nodes"] if not n["type"].startswith("voice_")]
    agent = client.post("/_debug/flows/flow-abc/deploy", json=flow).json()["agent"]
    assert "stt_provider" not in agent
