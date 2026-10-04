"""Turning a provider table entry into a live LiveKit plugin.

The *table* — which providers exist, which key each needs, which voices each
accepts — lives in `backend/core/providers.py` and imports no livekit. This
module holds the other half: the factory per entry, and the `build` that picks
one.

Two tables keyed alike, each owned by one layer. The worker does not register
factories into the core table at startup: that would be import-time global
mutation, and it would make `core`'s behaviour depend on whether a worker had
booted. `test_every_registered_provider_has_a_factory` asserts the two key sets
match, so a provider cannot be added to one and forgotten in the other.

Imported here at module scope, and never inside a build function. Importing a
plugin registers it, and registration is only legal on the main thread; a lazy
import runs on the job thread and dies with "Plugins must be registered on the
main thread" on the first call.
"""

import logging
import os
from typing import Any, Callable

from livekit.plugins import anthropic, cartesia, deepgram, google, nvidia, openai, sarvam

from backend.core import providers

logger = logging.getLogger("service-desk.providers")


# --- factories -------------------------------------------------------------
# Every factory takes the same keyword arguments and ignores what it does not
# use. Uniform signatures are what let `build` stay short.

def _google_llm(*, model=None, base_url=None, api_key=None, **_):
    # Measured, not assumed. gemini-3.5-flash-lite is the only model tried that
    # is both fast enough (695ms to first token) and safe (asked before writing
    # 3/3). gemini-*2.5*-flash-lite is a different thing entirely and wrote
    # without asking in repeated provider checks. The version number is doing real work.
    return google.LLM(model=model or "gemini-3.5-flash-lite")


def _openai_llm(*, model=None, base_url=None, api_key=None, **_):
    return openai.LLM(model=model or "gpt-5-mini")


def _anthropic_llm(*, model=None, base_url=None, api_key=None, **_):
    return anthropic.LLM(model=model or "claude-haiku-4-5-20251001")


def _nvidia_llm(*, model=None, base_url=None, api_key=None, **_):
    # build.nvidia.com is OpenAI-compatible, so the openai plugin drives it with
    # a different base URL. The id "nvidia/nvidia-nemotron-nano-9b-v2" has the
    # vendor twice and both are load bearing: other hosts spell the same weights
    # "nvidia/nemotron-nano-9b-v2" and that returns a bare-text 404 here.
    # GET /v1/models is the authority.
    return openai.LLM(
        model=model or "nvidia/nemotron-3.5-lightning-30b-a3b",
        base_url=base_url or os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key=api_key,
    )


def _nebius_llm(*, model=None, base_url=None, api_key=None, **_):
    # Nebius Token Factory is OpenAI-compatible too. Same Nemotron weights as
    # the nvidia entry, different spelling: Nebius uses "nvidia/Nemotron-3_5-Lightning",
    # and the build.nvidia.com id 404s here. GET /v1/models is the authority.
    return openai.LLM(
        model=model or "nvidia/Nemotron-3_5-Lightning",
        base_url=base_url or os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"),
        api_key=api_key,
    )


def _compatible_llm(*, model=None, base_url=None, api_key=None, **_):
    """Any host speaking the OpenAI protocol: Groq, Together, Fireworks,
    OpenRouter, Ollama, vLLM, LM Studio. The caller supplies both halves."""
    return openai.LLM(model=model, base_url=base_url, api_key=api_key)


def _deepgram_stt(*, model=None, sample_rate=None, **_):
    # Flux's own end-of-turn model is why this is the default (plan section 2).
    kwargs = {}
    if sample_rate is not None:
        kwargs["sample_rate"] = sample_rate
    return deepgram.STTv2(
        model=model or os.getenv("DEEPGRAM_STT_MODEL", "flux-general-en"),
        eager_eot_threshold=float(os.getenv("EAGER_EOT_THRESHOLD", "0.4")),
        **kwargs,
    )


def _nvidia_stt(*, model=None, **_):
    return nvidia.STT(
        model=model or "parakeet-1.1.b-en-US-asr-streaming-silero-vad-sortformer",
        language_code=os.getenv("NVIDIA_STT_LANGUAGE", "en-US"),
    )


def _sarvam_stt(*, model=None, sample_rate=None, **_):
    # sarvam.STT, not STTRealtime: the realtime class is hard-locked to
    # saaras:v3-realtime. This one streams saaras:v4 over a websocket and emits
    # END_OF_SPEECH from Sarvam's server VAD, so turn_detection="stt" still holds.
    # ponytail: a non-saaras model means a Deepgram id leaked in (the flow editor
    # pre-fills flux-general-en), so it is ignored rather than sent to Sarvam.
    if not (model or "").startswith("saaras"):
        model = "saaras:v4"
    kwargs = {}
    if sample_rate is not None:
        kwargs["sample_rate"] = sample_rate
    return sarvam.STT(
        model=model,
        language=os.getenv("SARVAM_STT_LANGUAGE", "en-IN"),
        **kwargs,
    )


def _deepgram_tts(*, voice=None, **_):
    # Deepgram's TTS model *is* its voice, which is why there is no tts_model.
    return deepgram.TTS(model=voice or os.getenv("DEEPGRAM_TTS_MODEL") or "aura-2-thalia-en")


def _cartesia_tts(*, voice=None, **_):
    kwargs = {"model": os.getenv("CARTESIA_MODEL", "sonic-3")}
    if voice:
        kwargs["voice"] = voice
    return cartesia.TTS(**kwargs)


def _nvidia_tts(*, voice=None, **_):
    return nvidia.TTS(
        voice=voice or "Magpie-Multilingual.EN-US.Leo",
        language_code=os.getenv("NVIDIA_TTS_LANGUAGE", "en-US"),
    )


# Keyed exactly as core.providers.REGISTRY is. The parity test enforces it.
FACTORIES: dict[tuple[str, str], Callable[..., Any]] = {
    ("llm", "google"): _google_llm,
    ("llm", "nvidia"): _nvidia_llm,
    ("llm", "nebius"): _nebius_llm,
    ("llm", "openai"): _openai_llm,
    ("llm", "anthropic"): _anthropic_llm,
    ("llm", "openai-compatible"): _compatible_llm,
    ("stt", "deepgram"): _deepgram_stt,
    ("stt", "nvidia"): _nvidia_stt,
    ("stt", "sarvam"): _sarvam_stt,
    ("tts", "deepgram"): _deepgram_tts,
    ("tts", "cartesia"): _cartesia_tts,
    ("tts", "nvidia"): _nvidia_tts,
}


def build(
    kind: str,
    name: str | None,
    *,
    default: str,
    model: str | None = None,
    voice: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    **extra,
):
    """Construct a plugin, falling back to `default` if that is not possible.

    This function does not raise, and that is deliberate rather than sloppy.
    Raising inside the LiveKit entrypoint kills the whole worker rather than one
    job: the framework is already holding a room handle, and the exception
    leaves it waiting for a ready signal that never comes until the FFI layer
    times out and panics. Measured once — one misconfigured agent took down all
    three. A caller on a fallback model is better off than a caller on a dead
    worker, and /_debug/agents refuses the bad configuration at save time so
    this path stays rare.
    """
    spec = providers.get(kind, name)
    if spec is None:
        if name:
            logger.error("unknown %s provider %r, falling back to %s", kind, name, default)
        return _fallback(kind, default, voice=voice, **extra)

    key = providers.credential(spec, api_key_env)
    if spec.env_key is not None and key is None:
        logger.error(
            "agent asked for the %s provider %r but %s is empty, so this call falls "
            "back to %s. Add the key to .env.", kind, spec.name, spec.env_key, default,
        )
        return _fallback(kind, default, voice=voice, **extra)
    if spec.env_key is None and key is None:
        logger.error(
            "provider %r needs api_key_env to name an environment variable that is "
            "set; falling back to %s", spec.name, default,
        )
        return _fallback(kind, default, voice=voice, **extra)

    try:
        factory = FACTORIES[(spec.kind, spec.name)]
        return factory(model=model, voice=voice, base_url=base_url, api_key=key, **extra)
    except Exception as exc:
        logger.error("could not build %s provider %r (%s), falling back to %s",
                     kind, spec.name, exc, default)
        return _fallback(kind, default, voice=voice, **extra)


def _fallback(kind: str, default: str, **kwargs):
    return FACTORIES[(kind, default)](model=None, **kwargs)
