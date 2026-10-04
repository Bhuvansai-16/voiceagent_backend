"""LiveKit entrypoint and session wiring.

Run it three ways:
    uv run python -m backend.worker.main console     # mic + speakers in the terminal, no LiveKit
    uv run python -m backend.worker.main dev         # joins a LiveKit room, use the Playground
    uv run python -m backend.worker.main start       # production worker

Every provider is swappable by env var so a wrong model identifier is a
one-line fix in .env rather than a code change.
"""

import asyncio
import json
import logging
import os


from backend.eventloop import use_selector_loop_on_windows

use_selector_loop_on_windows()

from dotenv import load_dotenv
from livekit.agents import (
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    mcp,
)
# Importing a plugin registers it, and registration is only legal on the main
# thread — a lazy import inside a build function runs in the job thread instead
# and dies with "Plugins must be registered on the main thread" on the first
# call. Every model plugin now lives in backend/worker/providers.py, which this module
# imports at top level, so they are all still registered on the main thread.
# silero stays here because the VAD is session wiring, not a provider choice.
from livekit.plugins import silero

from backend.core import personas
from backend.core import providers
from backend.worker import plugins, telephony, tools, tracing
from backend.core.composio import COMPOSIO_MCP_BASE, composio_mcp_url
from backend.worker.brain import ServiceDeskAgent

load_dotenv()

logger = logging.getLogger("service-desk")

# asyncio does not hold a strong reference to running tasks, so a fire-and-forget
# task can be garbage collected mid-flight. Park them here until they finish.
_background: set[asyncio.Task] = set()


def _fire(coro) -> None:
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)

AI_DISCLOSURE = (
    "Greet the caller in one sentence. State plainly that you are an automated"
    " assistant, then ask how you can help. Do not list your capabilities."
)


def build_stt(persona=None, sample_rate: int | None = None):
    """The agent's speech recognition, config first then environment.

    sample_rate is passed only when feeding audio from a source whose rate is
    already known — a wrong value does not error, it returns an empty transcript.
    """
    name = getattr(persona, "stt_provider", None) or os.getenv("STT_PROVIDER", "deepgram")
    return plugins.build(
        "stt", name,
        default="deepgram",
        model=getattr(persona, "stt_model", None) or None,
        sample_rate=sample_rate,
    )


def turn_detection_for(persona=None) -> str:
    """"stt" when the chosen recogniser streams, "vad" when it does not.

    Flux and Riva both do semantic end-of-turn detection themselves, which is
    worth roughly 260ms against VAD silence timeouts. A non-streaming recogniser
    has no end-of-turn to report, and leaving turn_detection on "stt" would not
    error — it would just get slower, which is the hardest kind of regression to
    notice. No shipped provider is non-streaming; the guard exists because the
    openai-compatible escape hatch makes one inevitable.
    """
    name = getattr(persona, "stt_provider", None) or os.getenv("STT_PROVIDER", "deepgram")
    spec = providers.get("stt", name)
    if spec is not None and not spec.streaming:
        logger.warning(
            "STT provider %r does not stream, so end-of-turn detection falls back "
            "to VAD and replies will be slower.", spec.name,
        )
        return "vad"
    return "stt"


def build_tts(persona=None):
    """The agent's speech synthesis. The persona wins over the environment,
    same precedence as the LLM."""
    name = getattr(persona, "tts_provider", None) or os.getenv("TTS_PROVIDER", "deepgram")
    return plugins.build(
        "tts", name,
        default="deepgram",
        voice=getattr(persona, "voice", None) or None,
    )


def build_llm(persona=None):
    """The agent's model, falling back to the environment.

    Per-agent models are the point of the gallery, and they also multiply the
    free tier: Gemini's 20 requests a day is counted per model, so three agents
    on three models get three separate budgets rather than sharing one.

    Which providers exist, which key each needs, and what happens when that key
    is missing all live in backend/worker/providers.py. This function only decides *which*
    name to look up.
    """
    name = (
        persona.resolved_provider()
        if persona is not None
        else os.getenv("LLM_PROVIDER", "google").strip().lower()
    )
    return plugins.build(
        "llm", name,
        default="google",
        model=getattr(persona, "model", None) or os.getenv("LLM_MODEL") or None,
        base_url=getattr(persona, "base_url", None) or None,
        api_key_env=getattr(persona, "api_key_env", None) or None,
    )


from backend.core.composio import COMPOSIO_MCP_BASE, composio_mcp_url, get_allowed_tools_for_toolkits


def build_mcp_servers(persona=None) -> list[mcp.MCPServer]:
    """Composio, plus any extra HTTP MCP servers listed in EXTRA_MCP_URLS.

    Servers are initialised in parallel by the session, so adding one does not
    multiply startup time.
    """
    servers: list[mcp.MCPServer] = []

    composio_url = composio_mcp_url()
    composio_key = os.getenv("COMPOSIO_API_KEY", "").strip()

    # Half-configured Composio used to attach nothing and say nothing, so the
    # agent simply had no Composio tools and gave no clue why. Both halves are
    # required: the key authenticates, the URL says which server and which user.
    if composio_key and not composio_url:
        logger.warning(
            "COMPOSIO_API_KEY is set but no MCP server is configured, so no Composio "
            "tools are attached. Create an MCP server at platform.composio.dev, then "
            "set either COMPOSIO_MCP_SERVER_ID=<server_id> (the user id is taken from "
            "COMPOSIO_USER_ID) or COMPOSIO_MCP_URL=%s/<server_id>?user_id=<user_id>",
            COMPOSIO_MCP_BASE,
        )
    elif composio_url and not composio_key:
        raise RuntimeError("COMPOSIO_MCP_URL is set but COMPOSIO_API_KEY is empty")

    if composio_url and composio_key:
        toolkits = getattr(persona, "toolkits", ()) if persona else ()
        allowed = get_allowed_tools_for_toolkits(toolkits)
        servers.append(
            mcp.MCPServerHTTP(
                composio_url,
                headers={"x-api-key": composio_key},
                allowed_tools=allowed,
            )
        )

    for url in (u.strip() for u in os.getenv("EXTRA_MCP_URLS", "").split(",")):
        if url:
            servers.append(mcp.MCPServerHTTP(url))

    logger.info("attached %d MCP server(s) (allowed tools: %s)", len(servers), allowed if composio_url and composio_key else "all")
    return servers


async def _load_personas() -> None:
    """Refresh personas from the store, tolerating a database that is down.

    A failure here must not fail the call: the built-in agents are the floor and
    a caller reaching a default agent is better than a caller reaching nobody.
    """
    from backend.store import get_store
    from backend.store.pool import configured, open_pool

    if not configured():
        logger.warning("NEON_CONNECTION_URI is not set; using built-in agents only")
        return
    try:
        await open_pool()
        personas.apply_overrides(await get_store().agent_overrides())

        from backend.core import docstore
        from backend.store.pool import get_pool

        docstore.bind(lambda: get_pool().connection())
    except Exception as exc:
        logger.error("could not read agents from the store, using built-ins: %s", exc)


async def entrypoint(ctx: JobContext) -> None:
    trace_file = tracing.setup(ctx.job.id)
    if trace_file:
        logger.info("tracing this call to %s", trace_file)

    # A Postgres sink buffers on OpenTelemetry's thread and needs a drainer.
    # Flushing every couple of seconds trades a little crash durability for not
    # putting a network write on the call path.
    stop_flushing = asyncio.Event()
    if isinstance(trace_file, tracing.BufferedTraceSink):
        from backend.store.traces import TraceStore

        await TraceStore().begin_call(ctx.job.id)
        _fire(trace_file.run(stop_flushing))

    # Resolved from the room name before the session is built, because the model
    # is per-agent and the session needs it at construction. The room name is
    # known at dispatch, so this does not wait for anyone to join.
    # Read the gallery from the shared store at job start. This is what makes a
    # console edit reach a call: two processes reading one database agree, where
    # two processes reading their own copy of a JSON file did not.
    await _load_personas()
    persona = personas.from_room(ctx.job.room.name if ctx.job.room else ctx.room.name)
    provider_name = persona.resolved_provider()
    model_name = persona.model or os.getenv("LLM_MODEL") or "provider default"
    logger.info("agent %s on %s/%s", persona.id, provider_name, model_name)

    # Stamped before the session starts, so every span in this file can be
    # attributed to the model that produced it.
    if isinstance(trace_file, tracing.BufferedTraceSink):
        from backend.store.traces import TraceStore

        await TraceStore().begin_call(
            ctx.job.id, agent=persona.id, model=model_name, provider=provider_name,
            voice=persona.voice or os.getenv("DEEPGRAM_TTS_MODEL") or "aura-2-thalia-en",
            room=ctx.job.room.name if ctx.job.room else ctx.room.name,
        )
    tracing.write_meta(
        trace_file,
        agent=persona.id,
        model=model_name,
        provider=provider_name,
        voice=persona.voice or os.getenv("DEEPGRAM_TTS_MODEL") or "aura-2-thalia-en",
        room=ctx.job.room.name if ctx.job.room else ctx.room.name,
    )

    session = AgentSession(
        stt=build_stt(persona),
        llm=build_llm(persona),
        tts=build_tts(persona),
        # Flux does semantic end-of-turn detection itself; VAD stays for
        # responsive interruption handling, which Flux alone does not cover.
        vad=silero.VAD.load(),
        turn_handling={
            "turn_detection": turn_detection_for(persona),
            # Calls are traced per stage. Preemptive generation is on by
            # default; preemptive_tts is not, and it starts speech
            # synthesis before end-of-turn is confirmed. It costs wasted
            # generations when a caller resumes talking, which on a free tier is
            # quota rather than money.
            "preemptive_generation": {"enabled": True, "preemptive_tts": True},
            # Plan section 154 wants a barge-in to cut audio within ~200ms.
            # LiveKit's default is 0.5s of speech before it counts as an
            # interruption, and dropping that also makes coughs, "mm-hm", and
            # background talk cut the agent off. It is a phone line, so that
            # happens. Exposed as a knob because the right value can only be
            # found by interrupting a real call and listening, not by reasoning.
            "interruption": {
                "min_duration": float(os.getenv("INTERRUPT_MIN_DURATION", "0.5")),
            },
        },
        mcp_servers=build_mcp_servers(persona),
    )

    @session.on("conversation_item_added")
    def _on_item(event) -> None:
        # item is ChatMessage | AgentHandoff — only the former carries text.
        text = getattr(event.item, "text_content", None) or ""
        role = str(getattr(event.item, "role", "?"))
        if text:
            _fire(
                tools.log_event(
                    "transcript", role=role, text=text[:400]
                )
            )
            # Capture agent responses for quality analytics (groundedness metric)
            if role == "assistant":
                tracing.write_event(
                    trace_file, "turn.agent",
                    text=text[:500],
                )

    await ctx.connect()

    # Memory is scoped to this identity, so it decides whose memories the agent
    # can see. The browser console mints "browser-tester"; a phone call would
    # carry the caller's number. Falling back to the room name keeps sessions
    # separate rather than pooling strangers into one shared memory.
    participant = await ctx.wait_for_participant()
    caller_id = getattr(participant, "identity", None) or ctx.room.name

    logger.info("caller %s connected to %s", caller_id, persona.id)

    # Deleting the room ends the session whichever way the caller arrived, so
    # the hangup is always wired rather than gated on detecting a phone.
    # The previous version tested `kind` and a "phone:" identity prefix, and the
    # prefix was left over from an earlier dialling path that set the identity
    # itself. On the current path LiveKit assigns it, so that test would have
    # quietly disabled end_call on exactly the calls it exists for.
    async def hang_up() -> None:
        await telephony.hang_up(ctx.room.name)

    await session.start(
        agent=ServiceDeskAgent(
            caller_id=caller_id, persona=persona, hang_up=hang_up,
            trace_path=trace_file,
        ),
        room=ctx.room,
    )

    # Listen for text turns sent from the browser chat input
    @ctx.room.on("data_received")
    def _on_data(data_packet) -> None:
        try:
            raw = data_packet.data.decode("utf-8")
            payload = json.loads(raw)
            text = payload.get("text") if isinstance(payload, dict) else raw
            if text and isinstance(text, str):
                logger.info("received text chat turn: %s", text)
                _fire(session.generate_reply(user_input=text.strip()))
        except Exception as exc:
            logger.debug("ignoring non-chat data packet: %s", exc)

    try:
        await session.generate_reply(instructions=AI_DISCLOSURE)
    finally:
        # Whatever happened, get the trace out before this job goes away.
        if isinstance(trace_file, tracing.BufferedTraceSink):
            stop_flushing.set()
            await trace_file.flush()
            from backend.store.traces import TraceStore

            await TraceStore().end_call(ctx.job.id)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
