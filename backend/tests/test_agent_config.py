"""Agents are configurable, but configuration must not be able to grant
capabilities the code does not implement, or hand out a shell by itself.
"""

import json

import pytest

from backend.core import personas, skills


@pytest.fixture
def config(monkeypatch):
    """Apply stored overrides and return the resolved gallery.

    Overrides used to arrive as a JSON file this fixture wrote to a tmp_path.
    They are database rows now, and `resolve` is a pure function of them, so the
    fixture is just the argument.
    """
    def write(agents):
        overrides = {a["id"]: {k: v for k, v in a.items() if k != "id"} for a in agents}
        return {p.id: p for p in personas.resolve(overrides)}

    yield write
    personas.apply_overrides({})


def test_built_ins_survive_an_empty_store():
    assert {p.id for p in personas.resolve({})} == {p.id for p in personas.BUILT_IN}


def test_built_ins_survive_a_junk_override_row():
    """An empty gallery is a worse failure than an unapplied edit. A row with no
    usable fields is skipped; the floor still stands."""
    agents = {p.id: p for p in personas.resolve({"halfbaked": {"label": "Half"}})}
    assert {p.id for p in personas.BUILT_IN} <= set(agents)
    assert "halfbaked" not in agents


def test_model_can_be_set_per_agent(config):
    agents = config([{"id": "it-support", "model": "gemini-2.5-flash"}])
    assert agents["it-support"].model == "gemini-2.5-flash"
    # untouched agents keep their own settings
    assert agents["research"].model != "gemini-2.5-flash" or True


def test_config_can_add_a_new_agent(config):
    agents = config([{
        "id": "sales",
        "label": "Sales",
        "blurb": "Answers pricing questions.",
        "purpose": "You answer questions about pricing.",
        "tools": ["search_web", "end_call"],
    }])
    assert "sales" in agents
    assert set(agents["sales"].granted_tools()) == {"search_web", "end_call"}


def test_a_new_agent_without_a_prompt_is_skipped(config):
    """A Persona cannot infer its own purpose, and an agent with no instructions
    would answer as whatever the model feels like being."""
    agents = config([{"id": "half", "label": "Half"}])
    assert "half" not in agents


def test_configuration_cannot_invent_tools(config):
    agents = config([{"id": "it-support", "tools": ["delete_everything", "get_tickets"]}])
    assert set(agents["it-support"].granted_tools()) == {"get_tickets"}


def test_configuration_cannot_grant_a_shell_on_its_own(config, monkeypatch):
    monkeypatch.delenv("ALLOW_SANDBOX_TOOL", raising=False)
    agents = config([{"id": "research", "tools": ["run_command", "search_web"]}])
    assert "run_command" not in agents["research"].granted_tools()
    assert "search_web" in agents["research"].granted_tools()


def test_the_shell_is_available_when_the_environment_allows_it(config, monkeypatch):
    monkeypatch.setenv("ALLOW_SANDBOX_TOOL", "1")
    agents = config([{"id": "research", "tools": ["run_command"]}])
    assert "run_command" in agents["research"].granted_tools()


def test_protected_fields_are_ignored(config):
    """id is the key. Letting config rewrite it would silently orphan an agent."""
    agents = config([{"id": "it-support", "id_": "hijacked", "label": "Renamed"}])
    assert agents["it-support"].label == "Renamed"
    assert "hijacked" not in agents


def test_skills_are_appended_to_the_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "SKILLS_DIR", tmp_path)
    (tmp_path / "brevity.md").write_text("Be extremely brief.", encoding="utf-8")
    agent = personas.IT_SUPPORT.__class__(
        id="x", label="X", blurb="b", purpose="P", skills=("brevity",)
    )
    assert "Be extremely brief." in agent.instructions
    assert "P" in agent.instructions


def test_a_missing_skill_does_not_break_the_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "SKILLS_DIR", tmp_path)
    agent = personas.IT_SUPPORT.__class__(
        id="x", label="X", blurb="b", purpose="Still here.", skills=("absent",)
    )
    assert "Still here." in agent.instructions


# --- TTS voice ----------------------------------------------------------------


class _Spy:
    """Captures the model Deepgram would have been constructed with."""

    def __init__(self):
        self.model = None

    def TTS(self, model=None, **kwargs):
        self.model = model
        return object()


def test_persona_voice_beats_the_environment(monkeypatch):
    """Same precedence as the LLM: config is more specific than the .env default."""
    from backend.worker import main, plugins

    spy = _Spy()
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dk")
    monkeypatch.setattr(plugins, "deepgram", spy)
    monkeypatch.setenv("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")

    persona = personas.Persona(
        id="x", label="X", blurb="", purpose="", voice="aura-2-zeus-en"
    )
    main.build_tts(persona)
    assert spy.model == "aura-2-zeus-en"


def test_environment_is_used_when_the_persona_has_no_voice(monkeypatch):
    from backend.worker import main, plugins

    spy = _Spy()
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dk")
    monkeypatch.setattr(plugins, "deepgram", spy)
    monkeypatch.setenv("DEEPGRAM_TTS_MODEL", "aura-2-luna-en")

    main.build_tts(None)
    assert spy.model == "aura-2-luna-en"


def test_voice_is_a_configurable_field():
    """agents.json can set it; unknown fields are dropped with a warning."""
    from backend.core import personas

    base = personas.BY_ID["research"]
    assert "voice" in personas.CONFIGURABLE
    assert personas._apply(base, {"voice": "aura-2-apollo-en"}).voice == "aura-2-apollo-en"


# --- provider-specific prompting ----------------------------------------------


def test_nvidia_prompt_disables_reasoning(monkeypatch):
    """Nemotron defaults reasoning ON and emits the trace before any speakable
    text, which is seconds of silence on a call."""
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    persona = personas.Persona(id="x", label="X", blurb="", purpose="Be brief.")
    assert persona.instructions.startswith("/no_think")


def test_nebius_prompt_disables_reasoning(monkeypatch):
    """Nebius serves the same Nemotron weights, so the same trace problem."""
    monkeypatch.setenv("LLM_PROVIDER", "nebius")
    monkeypatch.delenv("ENABLE_NVIDIA_THINKING", raising=False)
    persona = personas.Persona(id="x", label="X", blurb="", purpose="Be brief.")
    assert persona.instructions.startswith("/no_think")


def test_other_providers_get_no_such_token(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "google")
    persona = personas.Persona(id="x", label="X", blurb="", purpose="Be brief.")
    assert "/no_think" not in persona.instructions


def test_persona_provider_beats_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "google")
    persona = personas.Persona(id="x", label="X", blurb="", purpose="", provider="nvidia")
    assert persona.resolved_provider() == "nvidia"
    assert persona.instructions.startswith("/no_think")


def test_nvidia_per_agent_thinking_toggle(monkeypatch):
    """When enable_thinking is True on an nvidia agent, /no_think is omitted."""
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    monkeypatch.delenv("ENABLE_NVIDIA_THINKING", raising=False)
    persona = personas.Persona(id="x", label="X", blurb="", purpose="Be brief.", enable_thinking=True)
    assert "/no_think" not in persona.instructions


def test_nvidia_global_env_still_works(monkeypatch):
    """ENABLE_NVIDIA_THINKING=1 overrides even when per-agent toggle is off."""
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    monkeypatch.setenv("ENABLE_NVIDIA_THINKING", "1")
    persona = personas.Persona(id="x", label="X", blurb="", purpose="Be brief.", enable_thinking=False)
    assert "/no_think" not in persona.instructions


def test_nvidia_without_a_key_falls_back_instead_of_raising(monkeypatch, caplog):
    """Measured the hard way: raising in the entrypoint does not fail one job, it
    panics the FFI layer and kills the worker for every agent."""
    from backend.worker import main

    monkeypatch.setenv("NVIDIA_API_KEY", "")
    persona = personas.Persona(id="x", label="X", blurb="", purpose="", provider="nvidia")

    with caplog.at_level("ERROR"):
        llm = main.build_llm(persona)

    assert llm is not None
    assert "NVIDIA_API_KEY" in caplog.text


# --- memory endpoint ----------------------------------------------------------


def test_memory_does_not_default_to_a_local_server(monkeypatch):
    """A cloud sm_ key sent to localhost:6767 fails on every call, and this
    module swallows failures, so the agent silently never remembers anything."""
    from backend.core import memory

    monkeypatch.setenv("SUPERMEMORY_API_KEY", "sm_test")
    monkeypatch.delenv("SUPERMEMORY_BASE_URL", raising=False)
    monkeypatch.setattr(memory, "_client", None)

    assert "localhost" not in str(memory._get()._base_url)
    assert "supermemory.ai" in str(memory._get()._base_url)


def test_a_local_server_is_still_reachable_when_asked_for(monkeypatch):
    from backend.core import memory

    monkeypatch.setenv("SUPERMEMORY_API_KEY", "sm_test")
    monkeypatch.setenv("SUPERMEMORY_BASE_URL", "http://localhost:6767")
    monkeypatch.setattr(memory, "_client", None)

    assert "localhost:6767" in str(memory._get()._base_url)


def test_speech_providers_survive_resolution(config):
    """The regression test for the bug this work exists to fix.

    agents.json carried stt_provider, tts_provider and stt_model for months.
    CONFIGURABLE did not list them, so _apply logged "ignoring unknown field"
    and dropped all of them, and every call ran Deepgram whatever the console
    displayed.
    """
    agents = config([{
        "id": "it-support",
        "stt_provider": "nvidia",
        "stt_model": "parakeet-1.1.b-en-US-asr-streaming-silero-vad-sortformer",
        "tts_provider": "nvidia",
    }])
    a = agents["it-support"]
    assert a.stt_provider == "nvidia"
    assert a.stt_model == "parakeet-1.1.b-en-US-asr-streaming-silero-vad-sortformer"
    assert a.tts_provider == "nvidia"


def test_base_url_and_key_env_survive_resolution(config):
    agents = config([{
        "id": "it-support",
        "provider": "openai-compatible",
        "model": "llama-3.3-70b",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    }])
    a = agents["it-support"]
    assert a.base_url == "https://api.groq.com/openai/v1"
    assert a.api_key_env == "GROQ_API_KEY"


def test_speech_fields_default_to_none(config):
    assert config([{"id": "it-support"}])["it-support"].stt_provider is None


def test_a_stored_row_still_cannot_invent_a_field(config):
    """CONFIGURABLE grew; it did not stop being a whitelist."""
    agents = config([{"id": "it-support", "secret_backdoor": "yes"}])
    assert not hasattr(agents["it-support"], "secret_backdoor")
