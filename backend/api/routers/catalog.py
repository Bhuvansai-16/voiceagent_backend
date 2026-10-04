"""What the picker may offer: models, voices, providers."""

import logging
import os

from fastapi import APIRouter

from backend.api.schemas import ModelListOut, ProviderListOut, VoiceListOut
from backend.core import providers
from backend.core.catalog import MODEL_NOTES, MODEL_PROVIDER

router = APIRouter(tags=["Catalog"])
logger = logging.getLogger("backend.api")


@router.get("/_debug/models", response_model=ModelListOut)
async def debug_models():
    """Models worth offering, with the caveats that are easy to forget.

    Merged from two sources on purpose: MODEL_PROVIDER says which registered
    provider runs a model, MODEL_NOTES carries measurements no API can report.
    """
    return {"models": [
        {"id": model_id, "provider": provider_name, **MODEL_NOTES.get(model_id, {})}
        for model_id, provider_name in MODEL_PROVIDER.items()
    ]}


@router.get("/_debug/voices", response_model=VoiceListOut)
async def debug_voices(provider: str = "deepgram"):
    """Voices the selected TTS provider accepts.

    Scoped per provider because they are not interchangeable: an aura-2 name
    sent to Riva fails the call rather than degrading it. `heard` says which
    have actually been heard, so the picker can be honest. The persona field
    takes any string, so this is a shortlist rather than a whitelist.
    """

    spec = providers.get("tts", provider)
    return {"provider": provider, "voices": list(spec.voices) if spec else []}


@router.get("/_debug/providers", response_model=ProviderListOut)
async def debug_providers():
    """Everything the worker can build, per kind.

    Reports whether each provider's environment variable is present, never its
    value. The console reads this so it cannot offer a provider that would fail
    at dispatch.
    """

    out: dict[str, list] = {"llm": [], "stt": [], "tts": []}
    for (kind, name), spec in providers.REGISTRY.items():
        out[kind].append({
            "name": name,
            "verified": spec.verified,
            "streaming": spec.streaming if kind == "stt" else None,
            "requires": spec.env_key,
            "key_present": bool(spec.env_key and os.getenv(spec.env_key, "").strip()),
            "voices": list(spec.voices),
        })
    for entries in out.values():
        entries.sort(key=lambda e: (not e["verified"], e["name"]))
    return {"providers": out}
