"""Every model provider the worker can build, as data.

This module is the *table*: what exists, which key each needs, which voices each
accepts. It deliberately imports no livekit. The factories that turn an entry
here into a live plugin live in `backend/worker/plugins.py`, keyed identically.

That split is why the API can import this module at all. The API serves
/_debug/providers, /_debug/voices and save-time validation from this table, and
never constructs a plugin. When the table and the factories lived together, the
API dragged every LiveKit plugin into its process to read a dict, and plugin
registration is only legal on the main thread — which is why every API import of
this module used to be a lazy in-function import.

LLM and speech are deliberately not symmetric, because they are not the same
problem. Every serious LLM host speaks the OpenAI protocol, so one
`openai-compatible` entry reaches all of them with a base_url and a key. STT
and TTS each have their own wire protocol (Deepgram websocket, Riva gRPC) and
need a real plugin, so that half stays a closed set. A table that pretended
otherwise would let someone select `whisper` and hear silence on a call.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger("service-desk.providers")

Kind = Literal["llm", "stt", "tts"]


@dataclass(frozen=True)
class Provider:
    name: str
    kind: Kind
    # The environment variable holding this provider's key. None means the
    # caller names one instead (only `openai-compatible` does this).
    env_key: str | None
    base_url: str | None = None
    # STT only. A non-streaming STT forces LiveKit into a VAD stream adapter,
    # which discards the semantic end-of-turn detection the latency budget
    # depends on. build_stt downgrades turn detection when this is False.
    streaming: bool = True
    # Heard on a real call. Never inferred — same honesty as the `heard` flag
    # that /_debug/voices already carries for voices.
    verified: bool = False
    # TTS only: the voices this provider accepts. A Deepgram voice name sent to
    # Riva is a guaranteed failure, so the picker is scoped per provider.
    voices: tuple[dict, ...] = field(default_factory=tuple)


# --- the table -------------------------------------------------------------

_AURA_VOICES = (
    {"id": "aura-2-thalia-en", "label": "Thalia", "character": "female, brisk", "heard": True},
    {"id": "aura-2-asteria-en", "label": "Asteria", "character": "female, warm", "heard": False},
    {"id": "aura-2-luna-en", "label": "Luna", "character": "female, soft", "heard": False},
    {"id": "aura-2-athena-en", "label": "Athena", "character": "female, measured", "heard": False},
    {"id": "aura-2-orion-en", "label": "Orion", "character": "male, even", "heard": False},
    {"id": "aura-2-arcas-en", "label": "Arcas", "character": "male, natural", "heard": False},
    {"id": "aura-2-apollo-en", "label": "Apollo", "character": "male, confident", "heard": False},
    {"id": "aura-2-zeus-en", "label": "Zeus", "character": "male, deep", "heard": False},
)

_MAGPIE_VOICES = (
    {"id": "Magpie-Multilingual.EN-US.Leo", "label": "Leo", "character": "male, even", "heard": False},
    {"id": "Magpie-Multilingual.EN-US.Mia", "label": "Mia", "character": "female, warm", "heard": False},
)

_CARTESIA_VOICES = ()  # Cartesia voices are account-scoped ids, not a fixed list.

_ALL = (
    Provider("google", "llm", "GOOGLE_API_KEY", verified=True),
    Provider("nvidia", "llm", "NVIDIA_API_KEY", verified=True),
    Provider("nebius", "llm", "NEBIUS_API_KEY"),
    Provider("openai", "llm", "OPENAI_API_KEY"),
    Provider("anthropic", "llm", "ANTHROPIC_API_KEY"),
    Provider("openai-compatible", "llm", None),
    Provider("deepgram", "stt", "DEEPGRAM_API_KEY", streaming=True, verified=True),
    Provider("nvidia", "stt", "NVIDIA_API_KEY", streaming=True),
    Provider("sarvam", "stt", "SARVAM_API_KEY", streaming=True),
    Provider("deepgram", "tts", "DEEPGRAM_API_KEY", verified=True, voices=_AURA_VOICES),
    Provider("cartesia", "tts", "CARTESIA_API_KEY", voices=_CARTESIA_VOICES),
    Provider("nvidia", "tts", "NVIDIA_API_KEY", voices=_MAGPIE_VOICES),
)

REGISTRY: dict[tuple[str, str], Provider] = {(p.kind, p.name): p for p in _ALL}


def get(kind: str, name: str | None) -> Provider | None:
    if not name:
        return None
    return REGISTRY.get((kind, name.strip().lower()))


def names(kind: str) -> list[str]:
    return [name for (k, name) in REGISTRY if k == kind]


def credential(spec: Provider, api_key_env: str | None) -> str | None:
    """This provider's key, or None when it is missing.

    `openai-compatible` has no fixed variable: the persona names one. Everything
    else carries its own on the spec.
    """
    var = spec.env_key or (api_key_env or "").strip()
    if not var:
        return None
    return os.getenv(var, "").strip() or None
