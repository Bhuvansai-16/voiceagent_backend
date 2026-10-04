"""Composio connector management endpoints under /_debug."""

import logging
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.api.schemas import (
    ComposioConfigIn,
    ComposioConfigOut,
    ConnectorIn,
    ConnectorListOut,
    ConnectorSavedOut,
    ConnectOut,
    DisconnectOut,
    OAuthUrlOut,
    Ok,
    ToggleAgentIn,
    ToggleAgentOut,
)
from backend.core import connectors as conn_logic
from backend.core import personas
from backend.core.composio import initiate_connection, verify_composio_key
from backend.store import get_store

router = APIRouter(tags=["Connectors"])
logger = logging.getLogger("backend.api")


async def _catalog() -> list[dict]:
    store = get_store()
    return conn_logic.merge(await store.connectors(), await store.agent_overrides())


@router.get("/_debug/connectors", response_model=ConnectorListOut)
async def debug_get_connectors():
    """All connectors with live connection status and active agent bindings."""
    status = conn_logic.get_composio_status()
    status["live_status"] = await verify_composio_key()
    return {"connectors": await _catalog(), "composio": status}


@router.post("/_debug/composio/config", response_model=ComposioConfigOut)
async def debug_update_composio_config(body: ComposioConfigIn):
    """Update the Composio API key or user id for this process.

    Deliberately does not write to .env. The previous version tried, resolved
    the path one directory too deep, and silently persisted nothing - so a key
    entered here worked until the next restart and then did not. Runtime only,
    and honest about it.
    """
    new_api_key = (body.api_key or "").strip()
    new_user_id = (body.user_id or "").strip()
    if new_api_key:
        os.environ["COMPOSIO_API_KEY"] = new_api_key
    if new_user_id:
        os.environ["COMPOSIO_USER_ID"] = new_user_id

    return {
        "success": True,
        "api_key_set": bool(os.getenv("COMPOSIO_API_KEY")),
        "user_id": os.getenv("COMPOSIO_USER_ID"),
        "verification": await verify_composio_key(new_api_key),
        "note": "Applies to this process. Add it to .env to survive a restart.",
    }


@router.post("/_debug/connectors", response_model=ConnectorSavedOut)
async def debug_save_connector(body: ConnectorIn):
    """Add or update a connector definition."""
    try:
        record = conn_logic.normalise(body.model_dump(exclude_unset=True))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return {"success": True, "connector": await get_store().upsert_connector(record)}


@router.post("/_debug/connectors/{connector_id}/toggle-agent", response_model=ToggleAgentOut)
async def debug_toggle_connector_agent(connector_id: str, body: ToggleAgentIn):
    """Toggle whether an agent includes this connector's toolkit."""
    agent_id = body.agent_id.strip()
    if not agent_id:
        return JSONResponse({"error": "agent_id is required"}, status_code=400)

    store = get_store()
    overrides = await store.agent_overrides()
    if agent_id not in overrides and agent_id not in {b.id for b in personas.BUILT_IN}:
        return JSONResponse({"error": f"Agent '{agent_id}' not found"}, status_code=404)

    current = overrides.get(agent_id, {}).get("toolkits", [])
    toolkits, enabled = conn_logic.toggle_toolkit(current, connector_id)

    await store.upsert_agent(agent_id, {"toolkits": toolkits})
    personas.apply_overrides(await store.agent_overrides())
    return {
        "success": True,
        "agent_id": agent_id,
        "connector_slug": connector_id.strip().lower(),
        "enabled": enabled,
        "toolkits": toolkits,
    }


async def _oauth_url(connector_id: str) -> tuple[dict | None, str | None]:
    """(connector, oauth_url). Falls back to the dashboard link on failure."""
    target = next(
        (c for c in await _catalog() if c["id"] == connector_id.strip().lower()), None
    )
    if target is None:
        return None, None

    slug = (target.get("slug") or connector_id).strip()
    toolkit = (target.get("toolkit") or target.get("id") or "").strip()
    try:
        return target, (await initiate_connection(slug)).get("redirect_url")
    except Exception as exc:
        logger.warning("Composio OAuth initiation failed for %s (slug=%s): %s",
                       connector_id, slug, exc)
    # An ac_ id can fail where the plain app name succeeds.
    if slug.startswith("ac_") and toolkit:
        try:
            return target, (await initiate_connection(toolkit)).get("redirect_url")
        except Exception:
            pass
    return target, None


@router.post("/_debug/connectors/{connector_id}/connect", response_model=ConnectOut)
async def debug_connect_connector(connector_id: str):
    """Begin a Composio OAuth flow and mark the connector connected."""
    target, oauth_url = await _oauth_url(connector_id)
    if target is None:
        return JSONResponse({"error": f"Connector '{connector_id}' not found"}, status_code=404)

    store = get_store()
    cid = connector_id.strip().lower()
    if not await store.set_connector_connected(cid, True):
        # Never stored before: insert the catalog entry, then flip the flag.
        await store.upsert_connector({k: v for k, v in target.items()
                                      if k in conn_logic.normalise(target)})
        await store.set_connector_connected(cid, True)

    return {
        "success": True,
        "id": cid,
        "slug": target.get("slug", cid),
        "connected": True,
        "auth_url": oauth_url or target.get("auth_url"),
        "message": f"Successfully connected {target.get('name', cid)}",
    }


@router.get("/_debug/connectors/{connector_id}/oauth-url", response_model=OAuthUrlOut)
async def debug_get_oauth_url(connector_id: str):
    """A fresh OAuth redirect URL, without marking anything connected."""
    target, oauth_url = await _oauth_url(connector_id)
    if target is None:
        return JSONResponse({"error": f"Connector '{connector_id}' not found"}, status_code=404)
    if oauth_url:
        return {"oauth_url": oauth_url}
    return {"oauth_url": target["auth_url"], "fallback": True}


@router.post("/_debug/connectors/{connector_id}/disconnect", response_model=DisconnectOut)
async def debug_disconnect_connector(connector_id: str):
    """Disconnect and unbind this connector from every agent."""
    store = get_store()
    cid = connector_id.strip().lower()
    await store.set_connector_connected(cid, False)

    overrides = await store.agent_overrides()
    for agent_id, entry in overrides.items():
        toolkits = entry.get("toolkits") or []
        kept = [t for t in toolkits if str(t).lower() not in (cid, )]
        if len(kept) != len(toolkits):
            await store.upsert_agent(agent_id, {"toolkits": kept})
    personas.apply_overrides(await store.agent_overrides())

    return {"success": True, "id": cid, "connected": False,
            "message": f"Successfully disconnected {cid}"}


@router.delete("/_debug/connectors/{connector_id}", response_model=Ok)
async def debug_delete_connector(connector_id: str):
    """Remove a connector's stored row. Catalog entries reappear as available."""
    if not await get_store().delete_connector(connector_id.strip().lower()):
        return JSONResponse({"error": "Connector not found"}, status_code=404)
    return {"success": True}
