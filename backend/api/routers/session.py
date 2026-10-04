"""Live session endpoints: browser tokens, outbound calls, telephony status."""

import logging
import os
import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.api.events import publish
from backend.api.schemas import CallPlacedOut, TelephonyOut, TokenOut
from backend.core import personas

router = APIRouter(tags=["Session"])
logger = logging.getLogger("backend.api")


@router.get("/_debug/livekit-token", response_model=TokenOut)
async def livekit_token(
    identity: str = "browser-tester",
    room: str = None,
    agent: str = None,
):
    """Mints WebRTC room credentials for the browser workbench."""
    from livekit import api

    key = os.getenv("LIVEKIT_API_KEY", "").strip()
    secret = os.getenv("LIVEKIT_API_SECRET", "").strip()
    url = os.getenv("LIVEKIT_URL", "").strip()
    if not (key and secret and url):
        missing = [
            name
            for name, value in (
                ("LIVEKIT_URL", url),
                ("LIVEKIT_API_KEY", key),
                ("LIVEKIT_API_SECRET", secret),
            )
            if not value
        ]
        return JSONResponse(
            {"error": f"missing in .env: {', '.join(missing)}"}, status_code=503
        )

    # Room name picks the persona: the worker reads it back with
    # personas.from_room, so no dispatch config and no second process is needed
    # to run three different agents.

    persona = personas.BY_ID.get(agent, personas.DEFAULT)
    room = room or f"{persona.id}-{uuid.uuid4().hex[:8]}"
    token = (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_name("Browser tester")
        .with_grants(
            api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True)
        )
        .to_jwt()
    )
    return {"url": url, "token": token, "room": room, "agent": persona.id}


class CallRequest(BaseModel):
    phone: str
    agent: str = "it-support"


@router.post("/_debug/call", response_model=CallPlacedOut)
async def debug_call(body: CallRequest):
    """Ring a phone and put the agent on the line.

    This spends money on every request and dials a real person, so it validates
    the number strictly and refuses rather than guessing. A dev convenience under
    /_debug: it will dial anything asked of it and must not survive into
    anything reachable by strangers.
    """
    from backend.worker import telephony

    if not telephony.configured():
        return JSONResponse(
            {"error": f"missing in .env: {', '.join(telephony.missing())}"},
            status_code=503,
        )
    if not telephony.valid_number(body.phone):
        return JSONResponse(
            {"error": f"{body.phone!r} is not E.164. Include the country code, e.g. +919876543210"},
            status_code=400,
        )

    persona = personas.BY_ID.get(body.agent, personas.DEFAULT)
    room = f"{persona.id}-{uuid.uuid4().hex[:8]}"
    try:
        result = await telephony.place_call(body.phone, room)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)

    publish({"kind": "call", "detail": f"dialling {body.phone} as {persona.id}"})
    return {"agent": persona.id, **result}


@router.get("/_debug/telephony", response_model=TelephonyOut)
async def debug_telephony():
    from backend.worker import telephony

    return {
        "configured": telephony.configured(),
        "missing": telephony.missing(),
        "max_call_seconds": telephony.MAX_CALL_SECONDS,
    }
