"""Agent gallery: read, save and delete agent configuration."""

import logging
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.api.schemas import AgentIn, AgentListOut, AgentSavedOut, Deleted
from backend.core import personas, providers
from backend.store import get_store

router = APIRouter(tags=["Agents"])
logger = logging.getLogger("backend.api")


@router.get("/_debug/agents", response_model=AgentListOut)
async def debug_agents():
    """The gallery. Served from the personas themselves so the page cannot drift
    out of sync with what the worker will actually run.

    `tools` is what configuration asked for; `granted_tools` is what the agent
    will really have. They differ when a config names an unknown tool or asks
    for the shell without ALLOW_SANDBOX_TOOL, and the UI shows both so the
    difference is visible rather than mysterious.
    """

    personas.apply_overrides(await get_store().agent_overrides())
    return {
        "agents": [
            {
                "id": p.id,
                "label": p.label,
                "blurb": p.blurb,
                "purpose": p.purpose,
                "tools": list(p.tools),
                "granted_tools": list(p.granted_tools()),
                "model": p.model,
                "provider": p.provider,
                "voice": p.voice,
                "toolkits": list(p.toolkits),
                "skills": list(p.skills),
                "enable_thinking": p.enable_thinking,
                "built_in": p.id in {b.id for b in personas.BUILT_IN},
            }
            for p in personas.ALL
        ],
        "known_tools": list(personas.KNOWN_TOOLS),
        "dangerous_tools": sorted(personas.DANGEROUS_TOOLS),
        "sandbox_allowed": os.getenv("ALLOW_SANDBOX_TOOL", "").strip().lower()
        in ("1", "true", "yes"),
    }


@router.put("/_debug/agents", response_model=AgentSavedOut)
async def debug_save_agent(body: AgentIn):
    """Write one agent to the store.

    Only the configurable fields are persisted, and the row merges rather than
    replaces, so a PUT carrying {id, provider} cannot wipe that agent's label,
    purpose, skills and tools - which is exactly what the file-based version
    did before it learned to merge. Capability filtering still happens at load: this endpoint can
    ask for a shell, and personas will still refuse without ALLOW_SANDBOX_TOOL.
    """

    # exclude_unset is what preserves the merge: a field the client never sent
    # is absent here, exactly as it was absent from the old raw dict, so it is
    # not written and the stored value survives. Dumping everything would post
    # a None over every field the editor did not touch.
    body = body.model_dump(exclude_unset=True)

    agent_id = (body.get("id") or "").strip()
    if not agent_id:
        return JSONResponse({"error": "id is required"}, status_code=400)


    # A provider whose key is absent is not a configuration, it is a call that
    # fails at dispatch. Refusing here is the difference between an error in the
    # panel and a caller listening to silence.
    for kind, field_name in (("llm", "provider"), ("stt", "stt_provider"), ("tts", "tts_provider")):
        name = (body.get(field_name) or "").strip().lower()
        if not name:
            continue
        spec = providers.get(kind, name)
        if spec is None:
            return JSONResponse(
                {"error": f"unknown {kind} provider {name!r}. "
                          f"Choose one of: {', '.join(sorted(providers.names(kind)))}."},
                status_code=400,
            )
        if spec.env_key is None:
            # openai-compatible names its own variable, so validate the pair.
            key_env = (body.get("api_key_env") or "").strip()
            if not key_env:
                return JSONResponse(
                    {"error": f"{name} needs api_key_env to name the environment "
                              "variable holding the key."},
                    status_code=400,
                )
            base = (body.get("base_url") or "").strip()
            if not base.startswith(("http://", "https://")):
                return JSONResponse(
                    {"error": f"{name} needs a base_url starting with http:// or https://."},
                    status_code=400,
                )
            if providers.credential(spec, key_env) is None:
                return JSONResponse(
                    {"error": f"{key_env} is empty. Add it to .env, restart the backend, then save."},
                    status_code=400,
                )
        elif providers.credential(spec, None) is None:
            return JSONResponse(
                {"error": f"{name} needs {spec.env_key} in .env. Add it, restart the backend, then save."},
                status_code=400,
            )

    # A voice belongs to one provider. aura-2-thalia-en sent to Riva is a call
    # that fails, so catch the mismatch here rather than on the phone.
    voice = (body.get("voice") or "").strip()
    tts_name = (body.get("tts_provider") or "deepgram").strip().lower()
    tts_spec = providers.get("tts", tts_name)
    if voice and tts_spec is not None and tts_spec.voices:
        if voice not in {v["id"] for v in tts_spec.voices}:
            return JSONResponse(
                {"error": f"voice {voice!r} is not a {tts_name} voice. "
                          f"Pick one of: {', '.join(v['id'] for v in tts_spec.voices)}."},
                status_code=400,
            )

    # One statement. The old path read agents.json, merged in Python, wrote the
    # whole file back and reloaded - with four other code paths doing the same
    # thing concurrently and no locking between them.
    store = get_store()
    fields = {
        field: body[field]
        for field in sorted(personas.CONFIGURABLE)
        if field in body and body[field] is not None
    }
    await store.upsert_agent(agent_id, fields)
    personas.apply_overrides(await store.agent_overrides())
    saved = personas.BY_ID.get(agent_id)
    return {
        "saved": agent_id,
        "granted_tools": list(saved.granted_tools()) if saved else [],
        # The worker reads config per call, but a session already in progress
        # keeps the agent it started with.
        "note": "applies to the next call, not one already running",
    }


@router.delete("/_debug/agents/{agent_id}", response_model=Deleted)
async def debug_delete_agent(agent_id: str):
    """Delete a custom agent, or reset a built-in to its defaults."""

    if agent_id in {b.id for b in personas.BUILT_IN}:
        # Built-ins are the floor; deleting the override row is what "delete"
        # can mean for them, and only if there is one.
        store = get_store()
        if not await store.delete_agent(agent_id):
            return JSONResponse(
                {"error": f"Cannot delete built-in agent '{agent_id}'"}, status_code=400
            )
        personas.apply_overrides(await store.agent_overrides())
        return {"deleted": agent_id}

    store = get_store()
    if not await store.delete_agent(agent_id):
        return JSONResponse(
            {"error": f"Agent '{agent_id}' not found"}, status_code=404
        )
    personas.apply_overrides(await store.agent_overrides())
    return {"deleted": agent_id}
