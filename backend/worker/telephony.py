"""Outbound phone calls: Twilio dials, LiveKit answers over SIP.

This is the one part of the project that spends real money on every use, so the
cost caps are set in three independent places and none of them depend on the
agent behaving.

**Why this shape.** LiveKit's documented outbound path uses a Twilio Elastic SIP
Trunk, which needs a Termination URI and a SIP credential list created by hand in
the Twilio console. The credentials already in `.env` are REST ones
(`ACCOUNT_SID`, `AUTH_TOKEN`, `TWILIO_FROM_NUMBER`), so this takes the other
route: Twilio's REST API places the call, and TwiML bridges the answered call
into LiveKit over SIP. Same result, using the credentials that exist.

The flow:

    POST /_debug/call  ->  twilio.calls.create(to=<caller>, twiml=<Dial><Sip>)
                                    |
                        caller's phone rings, they answer
                                    |
                       Twilio dials sip:<room>@<LIVEKIT_SIP_HOST>
                                    |
                 LiveKit inbound trunk + callee dispatch rule -> room
                                    |
                          agent is already in that room

The room name carries the persona, exactly as it does for browser sessions, so
`SIPDispatchRuleCallee` with `randomize=False` puts the call in the room the
grid asked for.
"""

import logging
import os
import re
from xml.sax.saxutils import escape, quoteattr

from livekit import api

logger = logging.getLogger("service-desk.telephony")

TRUNK_NAME = os.getenv("SIP_TRUNK_NAME", "voice-agent-inbound")
RULE_NAME = os.getenv("SIP_RULE_NAME", "voice-agent-callee")

# A call nobody ends is the failure that costs money rather than time. Capped on
# the Twilio leg, on the LiveKit trunk, and by ringing timeout, so no single
# component failing removes the ceiling.
MAX_CALL_SECONDS = int(os.getenv("MAX_CALL_SECONDS", "300"))
RINGING_TIMEOUT_SECONDS = int(os.getenv("RINGING_TIMEOUT_SECONDS", "30"))

# Deliberately strict. This value is dialled, and it arrives from a browser
# field, so "looks roughly like a number" is not good enough.
E164 = re.compile(r"^\+[1-9]\d{7,14}$")

REQUIRED = (
    "ACCOUNT_SID",
    "AUTH_TOKEN",
    "TWILIO_FROM_NUMBER",
    "LIVEKIT_SIP_HOST",
    "SIP_AUTH_USERNAME",
    "SIP_AUTH_PASSWORD",
)


def configured() -> bool:
    return not missing()


def missing() -> list[str]:
    return [name for name in REQUIRED if not os.getenv(name, "").strip()]


def valid_number(phone: str) -> bool:
    """E.164 only: a leading + and 8 to 15 digits."""
    return bool(E164.match(phone.strip()))


async def ensure_inbound_route(lk: api.LiveKitAPI) -> tuple[str, str]:
    """The inbound trunk and callee dispatch rule, created once if absent.

    Both persist on the LiveKit project, so this looks them up by name rather
    than creating duplicates on every call.
    """
    trunk_id = ""
    for trunk in (await lk.sip.list_sip_inbound_trunk(api.ListSIPInboundTrunkRequest())).items:
        if trunk.name == TRUNK_NAME:
            trunk_id = trunk.sip_trunk_id
            break

    if not trunk_id:
        created = await lk.sip.create_sip_inbound_trunk(
            api.CreateSIPInboundTrunkRequest(
                trunk=api.SIPInboundTrunkInfo(
                    name=TRUNK_NAME,
                    # Digest auth, so an inbound INVITE from anywhere else is
                    # rejected. Without this the trunk answers whoever finds it.
                    auth_username=os.environ["SIP_AUTH_USERNAME"],
                    auth_password=os.environ["SIP_AUTH_PASSWORD"],
                    # Second cost ceiling, enforced by LiveKit rather than Twilio.
                    max_call_duration=_duration(MAX_CALL_SECONDS),
                    ringing_timeout=_duration(RINGING_TIMEOUT_SECONDS),
                    krisp_enabled=True,
                )
            )
        )
        trunk_id = created.sip_trunk_id
        logger.info("created inbound trunk %s", trunk_id)

    for rule in (await lk.sip.list_sip_dispatch_rule(api.ListSIPDispatchRuleRequest())).items:
        if rule.name == RULE_NAME:
            return trunk_id, rule.sip_dispatch_rule_id

    # Callee dispatch with no randomness and no prefix: the room is exactly the
    # user part of the SIP URI Twilio dialled, which is how the persona choice
    # survives the trip through the phone network.
    created_rule = await lk.sip.create_sip_dispatch_rule(
        api.CreateSIPDispatchRuleRequest(
            name=RULE_NAME,
            trunk_ids=[trunk_id],
            rule=api.SIPDispatchRule(
                dispatch_rule_callee=api.SIPDispatchRuleCallee(
                    room_prefix="", randomize=False
                )
            ),
        )
    )
    logger.info("created dispatch rule %s", created_rule.sip_dispatch_rule_id)
    return trunk_id, created_rule.sip_dispatch_rule_id


def _duration(seconds: int):
    from google.protobuf.duration_pb2 import Duration

    return Duration(seconds=seconds)


def sip_host() -> str:
    """The bare host, with no scheme.

    The LiveKit console shows this value as `sip:xyz.sip.livekit.cloud`, so it
    gets pasted into .env with the scheme attached. Building `sip:{room}@{host}`
    from that produced `sip:room@sip:xyz...`, which does not route. Accept it
    written either way rather than relying on everyone trimming it by hand.
    """
    host = os.environ["LIVEKIT_SIP_HOST"].strip()
    for prefix in ("sips:", "sip:"):
        if host.lower().startswith(prefix):
            host = host[len(prefix):]
            break
    return host.strip().rstrip("/")


def twiml_for(room: str) -> str:
    """TwiML that bridges the answered call into the agent's room.

    `answerOnBridge` matters: without it Twilio answers immediately and the
    caller hears silence while SIP connects. With it, their phone keeps ringing
    until the agent is actually on the line.

    Everything interpolated is XML-escaped. A password containing `&` produced
    Twilio error 12100, "Document parse failure: the reference to entity 'tN1'
    must end with the ';' delimiter" — the caller's phone rang, Twilio read the
    trial preamble, then hung up with "an application error has occurred".
    Random passwords contain `&` roughly half the time, so this was a coin flip,
    not an edge case.
    """
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Dial answerOnBridge="true" timeLimit="{MAX_CALL_SECONDS}">'
        f"<Sip username={quoteattr(os.environ['SIP_AUTH_USERNAME'])}"
        f" password={quoteattr(os.environ['SIP_AUTH_PASSWORD'])}>"
        f"{escape(f'sip:{room}@{sip_host()}')}"
        "</Sip>"
        "</Dial>"
        "</Response>"
    )


async def place_call(phone: str, room_name: str) -> dict:
    """Ring `phone` and bridge them to `room_name`. Returns while it rings."""
    if not valid_number(phone):
        raise ValueError(f"{phone!r} is not an E.164 number, e.g. +919876543210")
    if gaps := missing():
        raise RuntimeError(f"telephony not configured, missing: {', '.join(gaps)}")

    lk = api.LiveKitAPI()
    try:
        await ensure_inbound_route(lk)
    finally:
        await lk.aclose()

    from twilio.rest import Client

    client = Client(os.environ["ACCOUNT_SID"], os.environ["AUTH_TOKEN"])
    call = client.calls.create(
        to=phone,
        from_=os.environ["TWILIO_FROM_NUMBER"],
        twiml=twiml_for(room_name),
        # Third ceiling. Twilio drops the call at this point regardless of what
        # LiveKit or the agent are doing.
        time_limit=MAX_CALL_SECONDS,
        timeout=RINGING_TIMEOUT_SECONDS,
    )
    logger.info("dialling %s into %s (twilio %s)", phone, room_name, call.sid)
    return {
        "room": room_name,
        "call_sid": call.sid,
        "status": call.status,
        "max_call_seconds": MAX_CALL_SECONDS,
    }


async def hang_up(room_name: str) -> None:
    """End the call by deleting the room, which drops the SIP leg with it."""
    lk = api.LiveKitAPI()
    try:
        await lk.room.delete_room(api.DeleteRoomRequest(room=room_name))
        logger.info("hung up %s", room_name)
    finally:
        await lk.aclose()
