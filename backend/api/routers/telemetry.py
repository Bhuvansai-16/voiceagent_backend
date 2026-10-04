"""Operational telemetry: the log feed, the event stream, and trace analytics."""

import asyncio
import json
import logging
import math
import os
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from backend.api.events import _subscribers, last_seq, publish, recent
from backend.api.schemas import (
    AnalyticsOut,
    CallDetailOut,
    LogEvent,
    LogsOut,
    QualityAnalyticsOut,
    TraceListOut,
)

router = APIRouter(tags=["Telemetry"])
logger = logging.getLogger("backend.api")


@router.get("/_debug/logs", response_model=LogsOut)
async def debug_logs(since: int = 0, limit: int = 300):
    """Recent agent events, newest last, for the live log view.

    Polled rather than streamed on purpose. /_debug/events exists and works, but
    an open SSE connection wedges `uvicorn --reload` (see that endpoint), and
    putting one in the main UI would bring that back on every code change. The
    seq numbers make polling cheap: the client asks for what it has not seen.
    """
    fresh = recent(since)
    return {
        "events": fresh[-limit:],
        "seq": last_seq(),
        "dropped": max(0, len(fresh) - limit),
    }


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    # Nearest-rank. With a handful of calls per model, interpolating between two
    # samples invents a number that was never measured.
    idx = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return round(ordered[idx], 1)


@router.get("/_debug/analytics", response_model=AnalyticsOut)
async def debug_analytics():
    """Per-model timings, aggregated in SQL.

    This used to glob TRACE_DIR and JSON-parse every trace file on every
    request, which is both an O(all calls) scan per poll and, once the worker
    runs in its own container, a read of a directory that is always empty.

    /_debug/models carries numbers typed in by hand from a bench; this carries
    what actually happened on real calls. Reported separately on purpose: a
    hand-entered figure and a measured distribution are different claims.
    """
    from backend.analytics.latency import reply_latencies
    from backend.store.traces import TraceStore

    store = TraceStore()
    stats = await store.per_model_stats()
    total = await store.total_calls()

    # Call counts come from the calls table, not from the per-stage aggregate:
    # grouping by stage would count one call once per stage it produced.
    by_model: dict[str, dict] = {
        row["model"]: {
            "model": row["model"], "provider": row["provider"],
            "agents": sorted(row["agents"] or []), "calls": row["calls"],
            "tools": 0, "stats": {},
        }
        for row in await store.per_model_calls()
    }

    for row in stats:
        bucket = by_model.setdefault(row["model"], {
            "model": row["model"], "provider": row["provider"], "agents": [],
            "calls": 0, "tools": 0, "stats": {},
        })
        stage = row["stage"]
        entry = {"n": row["n"],
                 "p50": round(row["p50"], 1) if row["p50"] is not None else None,
                 "p95": round(row["p95"], 1) if row["p95"] is not None else None}
        if stage.startswith("tool."):
            bucket["tools"] += row["n"]
            tool = bucket["stats"].setdefault(
                "tool", {"n": 0, "p50": None, "p95": None, "by_name": {}})
            tool["n"] += row["n"]
            tool["by_name"][stage.split(".", 1)[1]] = row["n"]
            if tool["p50"] is None:
                tool["p50"], tool["p95"] = entry["p50"], entry["p95"]
        else:
            bucket["stats"][stage] = entry

    # reply latency is the pairing algorithm, not a percentile: match each user
    # turn to the first audio after it, and skip turns where a second turn ended
    # in between, because ownership there is genuinely ambiguous.
    model_of = {c["id"]: c["model"] for c in await store.calls_summary()}
    for call_id in await store.call_ids():
        derived = reply_latencies(await store.rows_for(call_id))
        if not derived:
            continue
        bucket = by_model.get(model_of.get(call_id) or "unattributed")
        if bucket is not None:
            bucket.setdefault("_reply", []).extend(derived)

    out = []
    for bucket in by_model.values():
        replies = sorted(bucket.pop("_reply", []))
        bucket["stats"].setdefault("reply", {
            "n": len(replies),
            "p50": _percentile(replies, 50),
            "p95": _percentile(replies, 95),
        })
        # The console reads underscore names; the stage names are dotted.
        for flat, dotted in (("first_token", "llm.first_token"),
                             ("tts_first_byte", "tts.first_byte"),
                             ("stt_finalize", "stt.finalize")):
            bucket["stats"][flat] = bucket["stats"].get(
                dotted, {"n": 0, "p50": None, "p95": None})
        bucket["stats"]["pipeline"] = {
            "stt": (bucket["stats"].get("stt.finalize") or {}).get("p50") or 0,
            "llm": (bucket["stats"].get("llm.first_token") or {}).get("p50") or 0,
            "tts": (bucket["stats"].get("tts.first_byte") or {}).get("p50") or 0,
            "tool": (bucket["stats"].get("tool") or {}).get("p50") or 0,
        }
        out.append(bucket)

    out.sort(key=lambda b: (b["stats"]["reply"]["p50"] is None,
                            b["stats"]["reply"]["p50"] or 0))
    return {"models": out, "files": total, "budget_ms": 800,
            "note": "Measured from call traces in Postgres. Turns where the agent "
                    "never spoke are skipped, and turns whose reply is ambiguous "
                    "are dropped rather than guessed."}


@router.post("/_debug/log", status_code=204)
async def debug_log(body: LogEvent):
    """The agent posts its own turn/latency events here so the UI can show them."""
    publish({"kind": body.kind, **body.detail})


@router.get("/_debug/events")
async def debug_events(request: Request):
    # Known, unfixed: with this feed open, `uvicorn --reload` wedges on
    # "Waiting for connections to close" and the old code keeps serving. Three
    # things were tried and measured, none worked. request.is_disconnected()
    # never fires, because on reload the client has not gone away, the server
    # wants to stop. --timeout-graceful-shutdown 2 changed nothing. Ending the
    # stream on a deadline does not help either: EventSource reconnects into
    # the still-draining server and resets the wait.
    #
    # The deadline bounds any one connection and allows clients to reconnect cleanly.
    since = 0
    try:
        since = int(request.headers.get("Last-Event-ID", 0))
    except ValueError:
        pass

    async def stream():
        # MAX_STREAM_SECONDS is read off backend.api.main at call time so tests can
        # monkeypatch it there; the module-level default lives in backend.api.events.
        from backend.api import main as app_module

        q: asyncio.Queue = asyncio.Queue()
        _subscribers.add(q)
        deadline = time.monotonic() + app_module.MAX_STREAM_SECONDS
        try:
            # Reconnect fast, and replay only what this client has not seen.
            yield "retry: 800\n\n"
            for event in recent()[-50:]:
                if event.get("seq", 0) > since:
                    yield f"id: {event.get('seq', 0)}\ndata: {json.dumps(event)}\n\n"
            while time.monotonic() < deadline:
                if await request.is_disconnected():
                    return
                try:
                    # Never wait past the deadline, or the connection outlives
                    # it by up to a whole keepalive interval.
                    wait = min(5, deadline - time.monotonic())
                    event = await asyncio.wait_for(q.get(), timeout=wait)
                    yield f"id: {event.get('seq', 0)}\ndata: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            raise
        finally:
            _subscribers.discard(q)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/_debug/traces", response_model=TraceListOut)
async def list_traces(limit: int = 50):
    """Historical calls with summary statistics."""
    from backend.store.traces import TraceStore

    rows = await TraceStore().calls_summary()
    return {"traces": [{
        "filename": r["id"],
        "agent": r["agent"] or "unknown",
        "model": r["model"] or "unattributed",
        "turns": r["turns"],
        "total_seconds": None,
        "avg_first_token_ms": round(r["avg_first_token"], 1) if r["avg_first_token"] else None,
        "created_at": r["started"].isoformat() if r["started"] else "",
    } for r in rows[:limit]]}


@router.get("/_debug/calls/{call_id}", response_model=CallDetailOut)
async def call_detail(call_id: str):
    """Everything known about one call.

    The console had four separate views of call data, each fetching its own
    aggregate; none of them could answer "what happened on this particular
    call". The rows were already in Postgres, just never exposed per call.

    Returns the conversation in order, the timings, and the quality metrics
    computed from the same events - so the drill-down and the aggregates cannot
    disagree about a call.
    """
    from backend.analytics.latency import reply_latencies
    from backend.analytics.quality import compute_all
    from backend.store.traces import TraceStore

    store = TraceStore()
    meta = next((c for c in await store.calls_summary() if c["id"] == call_id), None)
    if meta is None:
        return JSONResponse({"error": f"unknown call {call_id}"}, status_code=404)

    rows = await store.rows_for(call_id)
    quality_events = [r for r in rows if str(r.get("stage", "")).startswith("turn.")]

    # The spoken conversation, in the order it happened.
    transcript = [
        {
            "turn": r.get("turn"),
            "who": "caller" if r["stage"] == "turn.user" else "agent",
            "text": r.get("text", ""),
            "intent": r.get("intent"),
        }
        for r in rows
        if r.get("stage") in ("turn.user", "turn.agent") and r.get("text")
    ]

    tools = [
        {
            "turn": r.get("turn"),
            "tool": r.get("tool"),
            "success": r.get("success"),
            "args": r.get("args"),
            "result": r.get("result"),
            "recovered": r.get("recovered_from_error"),
        }
        for r in rows if r.get("stage") == "turn.tool_call"
    ]

    stage_ms: dict[str, list[float]] = {}
    for r in rows:
        if r.get("ms") is not None:
            stage_ms.setdefault(r["stage"], []).append(float(r["ms"]))

    replies = reply_latencies(rows)
    return {
        "id": call_id,
        "agent": meta["agent"],
        "model": meta["model"],
        "provider": meta["provider"],
        "started": meta["started"].isoformat() if meta["started"] else None,
        "transcript": transcript,
        "tools": tools,
        "timings": {
            stage: {"n": len(v), "p50": _percentile(v, 50), "max": round(max(v), 1)}
            for stage, v in sorted(stage_ms.items())
        },
        "reply_ms": {
            "n": len(replies),
            "p50": _percentile(replies, 50),
            "slowest": round(max(replies), 1) if replies else None,
            "over_budget": len([r for r in replies if r > 800]),
        },
        "quality": compute_all(quality_events) if quality_events else {},
        "event_count": len(rows),
    }


@router.get("/_debug/quality-analytics", response_model=QualityAnalyticsOut)
async def debug_quality_analytics():
    """Per-model and per-call LLM quality metrics, read from Postgres."""
    from backend.analytics.quality import compute_all, compute_per_call, compute_per_model
    from backend.store.traces import TraceStore

    store = TraceStore()
    # per-model grouping needs each call's model, which lives on the calls row
    # rather than in the events.
    meta_of = {c["id"]: {"model": c["model"], "agent": c["agent"]}
               for c in await store.calls_summary()}
    calls = []
    for call_id in await store.call_ids():
        rows = await store.rows_for(call_id)
        events = [r for r in rows if str(r.get("stage", "")).startswith("turn.")]
        if events:
            calls.append({"filename": call_id, "meta": meta_of.get(call_id, {}),
                          "events": events})

    total = await store.total_calls()
    if not calls:
        return {"aggregate": {}, "per_model": {}, "per_call": [],
                "calls_with_quality_data": 0, "total_files": total,
                "note": "No quality events yet. They are recorded from new calls "
                        "only, so a call has to happen first."}

    all_events = [e for c in calls for e in c["events"]]
    return {
        "aggregate": compute_all(all_events),
        "per_model": compute_per_model(calls),
        "per_call": compute_per_call(calls),
        "calls_with_quality_data": len(calls),
        "total_files": total,
        "metric_definitions": {
            "task_success_rate": {
                "label": "Task Success Rate",
                "priority": 3,
                "description": "Percentage of tool calls that returned a success response.",
            },
            "intent_accuracy": {
                "label": "Intent Accuracy",
                "priority": 3,
                "description": "Percentage of user turns where detected intent matched the tool called.",
            },
            "tool_selection_accuracy": {
                "label": "Tool Selection Accuracy",
                "priority": 3,
                "description": "Percentage of tool calls that used a known/valid tool name.",
            },
            "tool_argument_accuracy": {
                "label": "Tool Argument Accuracy",
                "priority": 3,
                "description": "Percentage of tool calls where arguments passed format validation.",
            },
            "groundedness": {
                "label": "Groundedness",
                "priority": 3,
                "description": "Percentage of agent responses grounded in tool data or conversation context.",
            },
            "recovery_success_rate": {
                "label": "Recovery Success Rate",
                "priority": 2,
                "description": "Percentage of tool failures followed by successful recovery.",
            },
            "context_retention": {
                "label": "Context Retention",
                "priority": 2,
                "description": "Percentage of entity references persisted across turns without re-prompting.",
            },
            "avg_turns_to_resolution": {
                "label": "Avg Turns to Resolution",
                "priority": 2,
                "description": "Average number of user turns before the call was resolved.",
                "unit": "turns",
            },
        },
    }
