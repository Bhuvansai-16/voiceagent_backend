"""The provider table is the single source of truth for what the worker can
build. Configuration selects from it and cannot add to it."""

import pytest

from backend.core import providers
from backend.worker import plugins


def test_every_registered_provider_has_a_factory():
    """The table and the factories are two dicts in two layers. A provider in
    one and not the other is a call that fails at dispatch, so assert the key
    sets are identical rather than trusting anyone to update both."""
    assert set(providers.REGISTRY) == set(plugins.FACTORIES)


def test_every_registry_key_matches_its_entry():
    for (kind, name), spec in providers.REGISTRY.items():
        assert spec.kind == kind
        assert spec.name == name


def test_known_llm_providers_are_registered():
    assert set(providers.names("llm")) >= {
        "google", "nvidia", "openai", "anthropic", "openai-compatible",
    }


def test_get_is_case_and_space_insensitive():
    assert providers.get("llm", "  NVIDIA ") is providers.get("llm", "nvidia")


def test_get_returns_none_for_an_unknown_provider():
    assert providers.get("llm", "whisper") is None


def test_credential_reads_the_specs_own_env_var(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "gk")
    assert providers.credential(providers.get("llm", "google"), None) == "gk"


def test_credential_is_none_when_the_env_var_is_blank(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "   ")
    assert providers.credential(providers.get("llm", "google"), None) is None


def test_openai_compatible_takes_its_key_from_the_named_variable(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gr")
    spec = providers.get("llm", "openai-compatible")
    assert spec.env_key is None
    assert providers.credential(spec, "GROQ_API_KEY") == "gr"


def test_build_uses_the_named_provider(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nk")
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return "llm"

    monkeypatch.setattr(plugins.openai, "LLM", fake)
    assert plugins.build("llm", "nvidia", default="google", model="m") == "llm"
    assert seen["model"] == "m"
    assert seen["api_key"] == "nk"


def test_build_falls_back_when_the_credential_is_absent(monkeypatch):
    """A missing key must not raise: an exception in the entrypoint kills the
    worker, not the job."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(plugins.google, "LLM", lambda **kw: "fallback")
    assert plugins.build("llm", "nvidia", default="google") == "fallback"


def test_build_falls_back_for_an_unknown_provider(monkeypatch):
    monkeypatch.setattr(plugins.google, "LLM", lambda **kw: "fallback")
    assert plugins.build("llm", "not-a-provider", default="google") == "fallback"


def test_build_never_raises_when_the_factory_explodes(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nk")

    def boom(**kwargs):
        raise RuntimeError("provider is down")

    monkeypatch.setattr(plugins.openai, "LLM", boom)
    monkeypatch.setattr(plugins.google, "LLM", lambda **kw: "fallback")
    assert plugins.build("llm", "nvidia", default="google") == "fallback"


# --- per-agent speech wiring ------------------------------------------------

from backend.worker import main as agent_main  # noqa: E402
from backend.core.personas import Persona  # noqa: E402


def _persona(**kw):
    base = dict(id="t", label="T", blurb="b", purpose="p")
    base.update(kw)
    return Persona(**base)


def test_build_stt_defaults_to_deepgram(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dk")
    monkeypatch.delenv("STT_PROVIDER", raising=False)
    monkeypatch.setattr(plugins.deepgram, "STTv2", lambda **kw: ("deepgram", kw))
    kind, kwargs = agent_main.build_stt(_persona())
    assert kind == "deepgram"
    assert kwargs["model"] == "flux-general-en"


def test_persona_selects_nvidia_stt(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nk")
    monkeypatch.setattr(plugins.nvidia, "STT", lambda **kw: ("nvidia", kw))
    kind, kwargs = agent_main.build_stt(_persona(stt_provider="nvidia", stt_model="parakeet-x"))
    assert kind == "nvidia"
    assert kwargs["model"] == "parakeet-x"


def test_persona_selects_sarvam_stt_on_saaras_v4(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "sk")
    monkeypatch.setattr(plugins.sarvam, "STT", lambda **kw: ("sarvam", kw))
    kind, kwargs = agent_main.build_stt(_persona(stt_provider="sarvam"))
    assert kind == "sarvam"
    assert kwargs["model"] == "saaras:v4"


def test_sarvam_ignores_a_leaked_deepgram_model_id(monkeypatch):
    """The flow editor pre-fills flux-general-en; Sarvam would reject it."""
    monkeypatch.setenv("SARVAM_API_KEY", "sk")
    monkeypatch.setattr(plugins.sarvam, "STT", lambda **kw: ("sarvam", kw))
    _, kwargs = agent_main.build_stt(_persona(stt_provider="sarvam", stt_model="flux-general-en"))
    assert kwargs["model"] == "saaras:v4"


def test_nebius_llm_points_at_token_factory(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "nb")
    monkeypatch.delenv("NEBIUS_BASE_URL", raising=False)
    seen = {}
    monkeypatch.setattr(plugins.openai, "LLM", lambda **kw: seen.update(kw) or "llm")
    assert plugins.build("llm", "nebius", default="google") == "llm"
    assert seen["base_url"] == "https://api.tokenfactory.nebius.com/v1"
    assert seen["model"] == "nvidia/Nemotron-3_5-Lightning"
    assert seen["api_key"] == "nb"


def test_persona_selects_nvidia_tts_with_its_own_voice(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nk")
    monkeypatch.setattr(plugins.nvidia, "TTS", lambda **kw: ("nvidia", kw))
    kind, kwargs = agent_main.build_tts(
        _persona(tts_provider="nvidia", voice="Magpie-Multilingual.EN-US.Leo")
    )
    assert kind == "nvidia"
    assert kwargs["voice"] == "Magpie-Multilingual.EN-US.Leo"


def test_stt_falls_back_to_deepgram_without_an_nvidia_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dk")
    monkeypatch.setattr(plugins.deepgram, "STTv2", lambda **kw: ("deepgram", kw))
    kind, _ = agent_main.build_stt(_persona(stt_provider="nvidia"))
    assert kind == "deepgram"


def test_tts_persona_voice_still_beats_the_environment(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dk")
    monkeypatch.setenv("DEEPGRAM_TTS_MODEL", "aura-2-zeus-en")
    monkeypatch.setattr(plugins.deepgram, "TTS", lambda **kw: ("deepgram", kw))
    _, kwargs = agent_main.build_tts(_persona(voice="aura-2-luna-en"))
    assert kwargs["model"] == "aura-2-luna-en"


def test_turn_detection_is_stt_for_a_streaming_provider(monkeypatch):
    monkeypatch.delenv("STT_PROVIDER", raising=False)
    assert agent_main.turn_detection_for(_persona()) == "stt"


def test_turn_detection_downgrades_for_a_non_streaming_provider(monkeypatch, caplog):
    """A non-streaming STT has no end-of-turn to give. Leaving turn_detection
    on "stt" would be a silent latency regression with no error."""
    monkeypatch.setitem(
        providers.REGISTRY, ("stt", "batchy"),
        providers.Provider("batchy", "stt", None, streaming=False),
    )
    with caplog.at_level("WARNING"):
        assert agent_main.turn_detection_for(_persona(stt_provider="batchy")) == "vad"
    assert "batchy" in caplog.text


def test_turn_detection_is_stt_for_an_unknown_provider(monkeypatch):
    """Unknown providers fall back to Deepgram, which streams."""
    assert agent_main.turn_detection_for(_persona(stt_provider="nope")) == "stt"
