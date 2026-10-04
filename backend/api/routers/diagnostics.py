"""Integration health and raw service-desk state."""

import logging
import os

from fastapi import APIRouter

from backend.api.db import _iso, fetch_all
from backend.api.schemas import IntegrationListOut, StateOut
from backend.core import memory, sandbox, websearch
from backend.core.composio import composio_mcp_url

router = APIRouter(tags=["Diagnostics"])
logger = logging.getLogger("backend.api")


@router.get("/_debug/integrations", response_model=IntegrationListOut)
async def debug_integrations():
    """Which outside services the agent can reach, and what is missing when it cannot.

    Returns whether each key is present, never its value. The `missing` list is
    the whole point: a half-configured integration attaches nothing and the only
    symptom on a call is a tool that never fires.
    """


    # Assembled the same way the worker assembles it, so the panel cannot report
    # connected while the agent finds nothing, or the reverse.
    composio_url = composio_mcp_url()
    composio_key = os.getenv("COMPOSIO_API_KEY", "").strip()
    composio_user = os.getenv("COMPOSIO_USER_ID", "").strip()
    extra = [u.strip() for u in os.getenv("EXTRA_MCP_URLS", "").split(",") if u.strip()]

    return {
        "integrations": [
            {
                "id": "composio",
                "label": "Composio",
                "kind": "mcp",
                "connected": bool(composio_url and composio_key),
                "missing": [
                    name
                    for name, value in (
                        ("COMPOSIO_MCP_SERVER_ID", composio_url),
                        ("COMPOSIO_API_KEY", composio_key),
                    )
                    if not value
                ],
                # Callers connect their own Gmail or GitHub through Composio's
                # OAuth. We never hold their credentials.
                "detail": (
                    f"Per-caller OAuth as {composio_user}."
                    if composio_user
                    else "Per-caller OAuth. Set COMPOSIO_USER_ID to name the account holder."
                ),
            },
            {
                "id": "extra-mcp",
                "label": "Other MCP servers",
                "kind": "mcp",
                "connected": bool(extra),
                "missing": [] if extra else ["EXTRA_MCP_URLS"],
                "detail": f"{len(extra)} attached" if extra else "None attached.",
            },
            {
                "id": "tavily",
                "label": "Tavily",
                "kind": "tool",
                "connected": websearch.configured(),
                "missing": [] if websearch.configured() else ["TAVILY_API_KEY"],
                "detail": "Backs search_web. Adds roughly half a second to the turn.",
            },
            {
                "id": "supermemory",
                "label": "supermemory",
                "kind": "tool",
                # Key presence only. Every other integration here fails loudly on
                # a bad key, but memory swallows failures by design so a turn is
                # never delayed, which means an invalid key looks identical to a
                # working one from the outside. Say so rather than imply health.
                "connected": memory.configured(),
                "missing": [] if memory.configured() else ["SUPERMEMORY_API_KEY"],
                "detail": "Backs remember_about_caller. Key present; validity is not "
                          "checked here, and recall failures are silent by design.",
            },
            {
                "id": "daytona",
                "label": "Daytona",
                "kind": "tool",
                "connected": sandbox.configured(),
                "missing": [] if sandbox.configured() else ["DAYTONA_API_KEY"],
                # Two gates, not one: the key alone must not be enough to reach a
                # shell by voice.
                "detail": "Backs run_command, which also needs ALLOW_SANDBOX_TOOL=1.",
            },
        ]
    }


@router.get("/_debug/state", response_model=StateOut)
async def debug_state():
    employees = await fetch_all("SELECT * FROM employees ORDER BY id")
    tickets = await fetch_all(
        "SELECT id, employee_id, title, status, updated FROM tickets ORDER BY id"
    )
    resets = await fetch_all(
        "SELECT reset_id, employee_id, status, sent_to, created, idempotency_key"
        " FROM password_resets ORDER BY created DESC"
    )
    return {
        "employees": employees,
        "tickets": [{**r, "updated": _iso(r["updated"])} for r in tickets],
        "resets": [{**r, "created": _iso(r["created"])} for r in resets],
    }
