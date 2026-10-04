"""Per-call latency spans, written to JSONL (plan section 164).

LiveKit already instruments itself with OpenTelemetry, and its spans already
carry the numbers this project needs: `lk.response.ttft`, `lk.response.ttfb`,
and the end-of-utterance delays. So nothing here re-instruments the pipeline. It
attaches one span processor, renames what LiveKit emits into the stage names the
plan fixed in section 166, and appends a line of JSON per stage.

Those stage names are a contract. Plan section 171 says Week 2's eval harness
will assert against them, so treat `STAGES` below as public API: add to it
freely, rename nothing.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanProcessor

logger = logging.getLogger("service-desk.tracing")

# The contract. LiveKit span name -> the stage names plan section 166 requires.
# A span can yield more than one stage: an LLM span carries both its
# time-to-first-token attribute and its own total duration.
STAGES: dict[str, tuple[str, ...]] = {
    "eou_detection": ("stt.finalize",),
    # `llm_node` is the logical call. `llm_request_run` is one attempt inside
    # its retry loop, so mapping both to llm.complete counted every call twice,
    # and would have counted a call that retried four times five times over.
    # Attempts are kept under their own name because they are how a one-second
    # call becomes a twenty-three second one.
    "llm_node": ("llm.first_token", "llm.complete"),
    "llm_request_run": ("llm.attempt",),
    "tts_node": ("tts.first_byte",),
    "tts_request_run": ("tts.attempt",),
    "function_tool": ("tool",),          # suffixed with the tool's own name
    # user_turn carries the STT timings as attributes as well as its own
    # duration. It has to, because `eou_detection` only exists when a separate
    # turn detector runs, and it does not here: turn_detection is "stt", so Flux
    # decides end-of-turn itself and LiveKit never opens that span. Reading
    # stt.finalize only from `eou_detection` left it permanently empty, which a
    # report cannot distinguish from a stage that was genuinely instant.
    # NOT turn.total. LiveKit's `user_turn` span covers the caller speaking, so
    # its duration is how long they talked, which has nothing to do with how
    # long the agent took to answer. Reporting it as turn.total made a 13s pause
    # look like a 13s latency. turn.total is derived in the report instead, by
    # pairing the end of a user turn with the first audio that followed it.
    "user_turn": ("user.speech", "stt.finalize", "stt.transcript"),
}

# Attributes that already hold a millisecond-ish number, so the stage reports
# what the provider measured rather than a duration we inferred around it.
FIRST_TOKEN_ATTR = "lk.response.ttft"
FIRST_BYTE_ATTR = "lk.response.ttfb"
TOOL_NAME_ATTR = "lk.function_tool.name"
END_OF_TURN_ATTR = "lk.end_of_turn_delay"
TRANSCRIPTION_ATTR = "lk.transcription_delay"

# Stages read straight off an attribute rather than from a span's duration.
FROM_ATTR = {
    "llm.first_token": FIRST_TOKEN_ATTR,
    "tts.first_byte": FIRST_BYTE_ATTR,
    "stt.finalize": END_OF_TURN_ATTR,
    "stt.transcript": TRANSCRIPTION_ATTR,
}


def _hex_id(ctx) -> str | None:
    """Span id as hex, or None. Identity is a nice-to-have for correlation, so a
    span shaped differently than expected must not cost us the whole trace."""
    span_id = getattr(ctx, "span_id", None)
    return f"{span_id:016x}" if isinstance(span_id, int) else None


def _ms(seconds_or_none) -> float | None:
    if seconds_or_none is None:
        return None
    try:
        return round(float(seconds_or_none) * 1000, 2)
    except (TypeError, ValueError):
        return None


def rows_for_span(span: ReadableSpan) -> list[dict]:
    """The stage rows one span yields, or an empty list.

    Pure, and separated from where the rows go. The mapping is the contract
    (plan section 171); the destination is a deployment detail — a local file
    when developing, Postgres in production, where the API can actually read it.
    """
    stages = STAGES.get(span.name)
    if not stages:
        return []

    attrs = dict(span.attributes or {})
    duration_ms = round((span.end_time - span.start_time) / 1_000_000, 2)
    rows = []

    for stage in stages:
        if stage in FROM_ATTR:
            ms = _ms(attrs.get(FROM_ATTR[stage]))
            # A missing attribute costs only its own stage. Realtime models
            # report no time-to-first-token, and that must not also lose the
            # completion timing sitting on the same span.
            if ms is None:
                continue
        elif stage == "tool":
            name = attrs.get(TOOL_NAME_ATTR) or attrs.get("gen_ai.tool.name") or "unknown"
            stage, ms = f"tool.{name}", duration_ms
        else:
            ms = duration_ms

        rows.append({
            "stage": stage,
            "ms": ms,
            "span": span.name,
            # Both ends, in epoch seconds. The report needs them to pair a
            # user turn with the reply that followed it; LiveKit does not put
            # lk.speech_id on these spans, so time order is the only link.
            "start": round(span.start_time / 1e9, 4),
            "end": round(span.end_time / 1e9, 4),
            # Identity, so a future report can correlate by the span tree
            # instead of by wall-clock order. Turns overlap, which makes
            # time ordering an unreliable way to decide which reply belongs
            # to which turn.
            "span_id": _hex_id(getattr(span, "context", None)),
            "parent_id": _hex_id(getattr(span, "parent", None)),
        })
    return rows


class StageSpanProcessor(SpanProcessor):
    """Maps finished spans to stage rows and hands them to a sink.

    `on_end` is called synchronously from OpenTelemetry's own thread, so the
    sink must not block and cannot await. The Postgres sink buffers; the file
    sink writes straight through.
    """

    def __init__(self, sink) -> None:
        self._sink = sink

    def on_end(self, span: ReadableSpan) -> None:
        rows = rows_for_span(span)
        if rows:
            self._sink(rows)

    def shutdown(self) -> None: ...
    def force_flush(self, timeout_millis: int = 30_000) -> bool: return True


class JsonlSpanProcessor(StageSpanProcessor):
    """Writes stage rows to a local JSONL file.

    Kept for local development and for the tests that pin the stage mapping.
    Unbatched on purpose: a call that crashes is exactly the call worth reading
    afterwards.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(self._write)

    def _write(self, rows: list[dict]) -> None:
        with self._lock, self._path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")


class BufferedTraceSink:
    """Collects rows on OpenTelemetry's thread; an async task drains them.

    This gives up the crash-durability the file processor had: a hard kill loses
    whatever has not flushed, roughly the last `interval` seconds. Writing each
    span to Neon synchronously on the call path would be far worse, and a local
    write-ahead file is the thing this migration exists to remove. The tradeoff
    is deliberate, not an oversight.
    """

    def __init__(self, call_id: str, store=None, interval: float = 2.0) -> None:
        self.call_id = call_id
        self._store = store
        self._interval = interval
        self._pending: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, rows: list[dict]) -> None:
        with self._lock:
            self._pending.extend(rows)

    def add(self, row: dict) -> None:
        self([row])

    def _take(self) -> list[dict]:
        with self._lock:
            rows, self._pending = self._pending, []
        return rows

    async def flush(self) -> int:
        rows = self._take()
        if not rows or self._store is None:
            return 0
        try:
            return await self._store.add_events(self.call_id, rows)
        except Exception as exc:
            # Telemetry must never take a call down.
            logger.warning("could not flush %d trace rows: %s", len(rows), exc)
            return 0

    async def run(self, stop) -> None:
        """Flush every `interval` seconds until `stop` is set, then once more."""
        import asyncio

        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass
            await self.flush()


def setup(job_id: str) -> "Path | BufferedTraceSink | None":
    """Start recording spans for this call. Returns the handle, or None if off.

    Postgres when NEON_CONNECTION_URI is set, because the API runs in a
    different process and cannot read the worker's disk. A local JSONL file
    otherwise, which keeps offline development working.

    Set TRACE_DIR empty to turn tracing off entirely.
    """
    if not os.getenv("TRACE_DIR", "calls") and not _postgres_tracing():
        return None

    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        # Nobody installed an SDK provider, so the global one is a no-op and
        # LiveKit's spans are being discarded. Install one.
        provider = TracerProvider()
        trace.set_tracer_provider(provider)

    if _postgres_tracing():
        from backend.store.traces import TraceStore

        sink = BufferedTraceSink(job_id, store=TraceStore())
        provider.add_span_processor(StageSpanProcessor(sink))
        return sink

    directory = os.getenv("TRACE_DIR", "calls")
    path = Path(directory) / f"{time.strftime('%Y%m%d-%H%M%S')}-{job_id}.jsonl"
    provider.add_span_processor(JsonlSpanProcessor(path))
    return path


def _postgres_tracing() -> bool:
    if os.getenv("TRACE_BACKEND", "").strip().lower() == "file":
        return False
    return bool(os.getenv("NEON_CONNECTION_URI", "").strip())


def write_meta(handle, **fields) -> None:
    """Record what this call was, as the first line of its trace.

    Spans carry timings and nothing else, so a file full of them cannot say
    which model produced them. Without this line the traces are unattributable
    and comparing models across calls is guesswork. Written as a `call.meta`
    stage so existing readers, which filter on known stage names, ignore it.
    """
    if not path:
        return
    row = {"stage": "call.meta", **fields, "at": time.time()}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError as exc:  # telemetry must never take a call down
        logger.warning("could not write call metadata: %s", exc)


# ---------------------------------------------------------------------------
# Quality analytics events
#
# These are structured events for computing the 8 LLM quality metrics.
# They sit alongside the latency spans in the same JSONL file. Existing
# readers that filter on known stage names will ignore them, and the
# quality analytics endpoint reads only these.
#
# Stage names:
#   turn.user          — user transcript
#   turn.agent         — agent response text
#   turn.tool_call     — tool invocation with args, result, success/fail
#   turn.intent        — detected intent from user speech
#   turn.error_recovery — tool failure followed by continued conversation
#   turn.resolution    — call resolved (end_call or last tool success)
# ---------------------------------------------------------------------------

_write_lock = threading.Lock()


def write_event(handle, stage: str, **fields) -> None:
    """Append a structured quality event to the trace file.

    Same contract as write_meta: fire-and-forget, telemetry must never
    break a live call. Fields are merged into a dict with `stage` and `at`.
    """
    if not handle:
        return
    row = {"stage": stage, "at": time.time(), **fields}
    if isinstance(handle, BufferedTraceSink):
        handle.add(row)
        return
    try:
        with _write_lock, handle.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
    except OSError as exc:
        logger.warning("could not write quality event %s: %s", stage, exc)
