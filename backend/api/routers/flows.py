"""Visual Flow Studio endpoints.

The graph compiler itself lives in core/flow_compiler.py; this file is storage
and HTTP. Deploying a flow writes an agent, which is why it touches the same
store the agents router does rather than a second file.
"""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.api.schemas import (
    FlowDeletedOut,
    FlowDeployedOut,
    FlowIn,
    FlowListOut,
    FlowSavedOut,
)
from backend.core import personas
from backend.core.flow_compiler import compile_flow
from backend.store import get_store

router = APIRouter(tags=["Flows"])
logger = logging.getLogger("backend.api")


@router.get("/_debug/flows", response_model=FlowListOut)
async def debug_get_flows():
    """Returns all saved visual voice agent flows and templates."""
    return {"flows": await get_store().flows()}


@router.post("/_debug/flows", response_model=FlowSavedOut)
async def debug_save_flow(flow: FlowIn):
    """Creates or updates a visual flow graph."""
    # exclude_unset so a graph saved without edges stores no edges, rather than
    # an empty list that overwrites the ones already there.
    flow = flow.model_dump(exclude_unset=True)

    flow_id = (flow.get("id") or "").strip()
    if not flow_id:
        flow_id = f"flow-{uuid.uuid4().hex[:8]}"
    flow["id"] = flow_id
    flow["updated_at"] = datetime.now(timezone.utc).isoformat()

    saved = await get_store().upsert_flow(flow)
    return {"success": True, "flow": saved}


@router.delete("/_debug/flows/{flow_id}", response_model=FlowDeletedOut)
async def debug_delete_flow(flow_id: str):
    """Deletes a visual flow graph by ID."""
    await get_store().delete_flow(flow_id)
    return {"success": True, "deleted": flow_id}


@router.post("/_debug/flows/{flow_id}/deploy", response_model=FlowDeployedOut)
async def debug_deploy_flow(flow_id: str, body: FlowIn | None = None):
    """Compiles a visual node flow graph into a live agent."""
    store = get_store()
    body = body.model_dump(exclude_unset=True) if body is not None else None

    target_flow = next((f for f in await store.flows() if f.get("id") == flow_id), None)
    if not target_flow:
        # A deploy may carry the flow directly, for a graph not yet saved.
        if body and body.get("nodes"):
            target_flow = {**body, "id": flow_id}
        else:
            return JSONResponse({"error": f"Flow '{flow_id}' not found"}, status_code=404)

    agent_entry = compile_flow(target_flow, flow_id)
    agent_id = agent_entry["id"]
    flow_name = target_flow.get("name") or "Visual Flow Agent"

    await store.upsert_agent(agent_id, {k: v for k, v in agent_entry.items() if k != "id"})
    personas.apply_overrides(await store.agent_overrides())

    target_flow["deployed_agent_id"] = agent_id
    target_flow["updated_at"] = datetime.now(timezone.utc).isoformat()
    await store.upsert_flow(target_flow)

    return {
        "success": True,
        "deployed_agent_id": agent_id,
        "agent": agent_entry,
        "message": (
            f"Successfully compiled flow '{flow_name}' into live Agent "
            f"'{agent_entry['label']}' ({agent_id})."
        ),
    }
