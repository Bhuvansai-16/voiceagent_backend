"""LLM quality metrics computed from call trace JSONL files.

Eight metrics derived from the structured events that brain.py emits alongside
the latency spans. Every metric returns a float between 0 and 1 (except
avg_turns_to_resolution, which is an absolute count).

Usage:
    uv run python scripts/quality_metrics.py                # all calls
    uv run python scripts/quality_metrics.py calls/*.jsonl   # specific files
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Known tools that the agent can legitimately call.
KNOWN_TOOLS = frozenset({
    "get_tickets", "create_ticket", "reset_password",
    "search_web", "search_documents", "remember_about_caller",
    "run_command", "end_call",
})


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_events(paths: list[Path]) -> tuple[dict, list[dict]]:
    """Read JSONL trace files and separate meta from quality events.

    Returns (meta, events) where meta is the call.meta row and events is
    a list of quality event rows (turn.user, turn.agent, turn.tool_call, etc.).
    """
    all_events: list[dict] = []
    meta: dict = {}
    for path in paths:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stage = row.get("stage", "")
                if stage == "call.meta":
                    meta = row
                elif stage.startswith("turn."):
                    all_events.append(row)
        except OSError:
            continue
    return meta, all_events


def read_all_calls(directory: Path) -> list[dict]:
    """Read all call trace files and group events per call.

    Returns a list of dicts, each with:
        - meta: the call.meta row
        - events: list of quality events
        - filename: the trace file name
    """
    calls = []
    files = sorted(directory.glob("*.jsonl")) if directory.is_dir() else []
    for path in files:
        meta, events = read_events([path])
        if events:  # only include calls that have quality events
            calls.append({
                "meta": meta,
                "events": events,
                "filename": path.name,
            })
    return calls


# ---------------------------------------------------------------------------
# Individual metric computations
# ---------------------------------------------------------------------------

def compute_task_success_rate(events: list[dict]) -> dict:
    """Metric 1: % of tool calls that returned a success response.

    A tool call is "successful" when success=True in the turn.tool_call event.
    """
    tool_calls = [e for e in events if e.get("stage") == "turn.tool_call"]
    if not tool_calls:
        return {"value": None, "n": 0, "successes": 0}
    successes = sum(1 for e in tool_calls if e.get("success"))
    return {
        "value": round(successes / len(tool_calls), 4),
        "n": len(tool_calls),
        "successes": successes,
    }


def compute_intent_accuracy(events: list[dict]) -> dict:
    """Metric 2: % of turns where the heuristic intent matched the tool called.

    For each turn.user with an intent, check if the next turn.tool_call in the
    same turn used the expected tool.
    """
    user_turns = [e for e in events if e.get("stage") == "turn.user" and e.get("intent")]
    tool_calls = [e for e in events if e.get("stage") == "turn.tool_call"]

    if not user_turns:
        return {"value": None, "n": 0, "matches": 0}

    # Build a map: turn number -> list of tools called
    tools_by_turn: dict[int, list[str]] = defaultdict(list)
    for tc in tool_calls:
        turn = tc.get("turn", 0)
        tools_by_turn[turn].append(tc.get("tool", ""))

    matches = 0
    evaluated = 0
    for ut in user_turns:
        expected_tool = ut["intent"]
        turn = ut.get("turn", 0)
        # Check this turn and the next (tool may fire on the agent's response turn)
        called = tools_by_turn.get(turn, []) + tools_by_turn.get(turn + 1, [])
        if called:
            evaluated += 1
            if expected_tool in called:
                matches += 1

    return {
        "value": round(matches / evaluated, 4) if evaluated else None,
        "n": evaluated,
        "matches": matches,
    }


def compute_tool_selection_accuracy(events: list[dict]) -> dict:
    """Metric 3: % of tool calls that used a known/valid tool name.

    Any call to a tool not in KNOWN_TOOLS is counted as a selection error.
    """
    tool_calls = [e for e in events if e.get("stage") == "turn.tool_call"]
    if not tool_calls:
        return {"value": None, "n": 0, "valid": 0}
    valid = sum(1 for e in tool_calls if e.get("tool") in KNOWN_TOOLS)
    return {
        "value": round(valid / len(tool_calls), 4),
        "n": len(tool_calls),
        "valid": valid,
    }


def compute_tool_argument_accuracy(events: list[dict]) -> dict:
    """Metric 4: % of tool calls where arguments passed validation.

    Validation is done at call time and recorded in args_valid.
    """
    tool_calls = [e for e in events if e.get("stage") == "turn.tool_call"]
    if not tool_calls:
        return {"value": None, "n": 0, "valid": 0}
    valid = sum(1 for e in tool_calls if e.get("args_valid", True))
    return {
        "value": round(valid / len(tool_calls), 4),
        "n": len(tool_calls),
        "valid": valid,
    }


def compute_groundedness(events: list[dict]) -> dict:
    """Metric 5: % of agent responses that appear grounded in tool/prompt data.

    Heuristic: an agent response is "grounded" if it came after a tool call
    in the same or immediately preceding turn. Responses without a preceding
    tool call are still grounded if they are conversational (greetings,
    confirmations, etc.) — we check for short responses or common patterns.
    """
    agent_msgs = [e for e in events if e.get("stage") == "turn.agent"]
    tool_calls = [e for e in events if e.get("stage") == "turn.tool_call"]

    if not agent_msgs:
        return {"value": None, "n": 0, "grounded": 0}

    tool_timestamps = sorted(tc.get("at", 0) for tc in tool_calls)

    grounded = 0
    for msg in agent_msgs:
        text = msg.get("text", "")
        msg_time = msg.get("at", 0)

        # Short conversational responses are inherently grounded (greetings, etc.)
        if len(text) < 100:
            grounded += 1
            continue

        # Check if a tool call happened within 30s before this response
        has_recent_tool = any(
            0 < (msg_time - tt) < 30 for tt in tool_timestamps
        )
        if has_recent_tool:
            grounded += 1
            continue

        # Check for specific data patterns that should come from tools
        # (employee IDs, ticket IDs, email addresses)
        has_specific_data = bool(
            re.search(r"(E\d{4}|INC\d{4}|PR-\w+|@\w+\.\w+)", text)
        )
        if not has_specific_data:
            # No specific data referenced = still grounded (general knowledge)
            grounded += 1

        # If it references specific data without a recent tool call, it's
        # potentially hallucinated

    return {
        "value": round(grounded / len(agent_msgs), 4),
        "n": len(agent_msgs),
        "grounded": grounded,
    }


def compute_recovery_success_rate(events: list[dict]) -> dict:
    """Metric 6: % of tool failures that were followed by successful recovery.

    A recovery is logged as turn.error_recovery when a failed tool is followed
    by a successful one.
    """
    failures = [e for e in events
                if e.get("stage") == "turn.tool_call" and not e.get("success")]
    recoveries = [e for e in events if e.get("stage") == "turn.error_recovery"]

    if not failures:
        return {"value": None, "n": 0, "recovered": 0}

    return {
        "value": round(len(recoveries) / len(failures), 4),
        "n": len(failures),
        "recovered": len(recoveries),
    }


def compute_context_retention(events: list[dict]) -> dict:
    """Metric 7: % of calls where entity IDs persist across turns.

    If the user mentions an employee ID in turn N and the agent or a tool
    uses it in a later turn without the user repeating it, context is retained.
    We check via the resolution event's entities_seen set.
    """
    resolutions = [e for e in events if e.get("stage") == "turn.resolution"]
    user_turns = [e for e in events if e.get("stage") == "turn.user"]

    if not resolutions:
        return {"value": None, "n": 0, "retained": 0}

    # Find entity mentions across user turns
    entity_turns: dict[str, list[int]] = defaultdict(list)
    for ut in user_turns:
        text = ut.get("text", "")
        turn = ut.get("turn", 0)
        for word in text.split():
            if re.match(r"^E\d{4}$", word):
                entity_turns[word].append(turn)

    # If any entity was mentioned in only one turn but used across multiple
    # tool calls, context was retained
    tool_calls = [e for e in events if e.get("stage") == "turn.tool_call"]
    entity_in_tools: dict[str, set[int]] = defaultdict(set)
    for tc in tool_calls:
        args = tc.get("args", {})
        for val in args.values():
            if isinstance(val, str) and re.match(r"^E\d{4}$", val):
                entity_in_tools[val].add(tc.get("turn", 0))

    retained = 0
    total = 0
    for entity, mention_turns in entity_turns.items():
        tool_turns = entity_in_tools.get(entity, set())
        if tool_turns:
            total += 1
            # Entity used in more tool turns than it was mentioned by user
            if len(tool_turns) >= len(mention_turns):
                retained += 1

    return {
        "value": round(retained / total, 4) if total else None,
        "n": total,
        "retained": retained,
    }


def compute_avg_turns_to_resolution(events: list[dict]) -> dict:
    """Metric 8: Average number of user turns before call resolution.

    Read from turn.resolution events' total_turns field.
    """
    resolutions = [e for e in events if e.get("stage") == "turn.resolution"]
    if not resolutions:
        return {"value": None, "n": 0}

    total_turns = [r.get("total_turns", 0) for r in resolutions]
    return {
        "value": round(sum(total_turns) / len(total_turns), 2),
        "n": len(resolutions),
        "min": min(total_turns),
        "max": max(total_turns),
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_all(events: list[dict]) -> dict:
    """Compute all 8 metrics from a list of quality events."""
    return {
        "task_success_rate": compute_task_success_rate(events),
        "intent_accuracy": compute_intent_accuracy(events),
        "tool_selection_accuracy": compute_tool_selection_accuracy(events),
        "tool_argument_accuracy": compute_tool_argument_accuracy(events),
        "groundedness": compute_groundedness(events),
        "recovery_success_rate": compute_recovery_success_rate(events),
        "context_retention": compute_context_retention(events),
        "avg_turns_to_resolution": compute_avg_turns_to_resolution(events),
    }


def compute_per_call(calls: list[dict]) -> list[dict]:
    """Compute metrics per call with metadata."""
    results = []
    for call in calls:
        metrics = compute_all(call["events"])
        results.append({
            "filename": call["filename"],
            "model": call["meta"].get("model", "unattributed"),
            "provider": call["meta"].get("provider"),
            "agent": call["meta"].get("agent", "unknown"),
            "metrics": metrics,
        })
    return results


def compute_per_model(calls: list[dict]) -> dict[str, dict]:
    """Aggregate metrics per model across calls."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for call in calls:
        model = call["meta"].get("model", "unattributed")
        by_model[model].extend(call["events"])

    result = {}
    for model, events in by_model.items():
        result[model] = compute_all(events)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "task_success_rate": ("Task Success Rate", "P1 CRITICAL"),
    "intent_accuracy": ("Intent Accuracy", "P1 CRITICAL"),
    "tool_selection_accuracy": ("Tool Selection Accuracy", "P1 CRITICAL"),
    "tool_argument_accuracy": ("Tool Argument Accuracy", "P1 CRITICAL"),
    "groundedness": ("Groundedness", "P1 CRITICAL"),
    "recovery_success_rate": ("Recovery Success Rate", "P2 STANDARD"),
    "context_retention": ("Context Retention", "P2 STANDARD"),
    "avg_turns_to_resolution": ("Avg Turns to Resolution", "P2 STANDARD"),
}


def main(argv: list[str]) -> int:
    if argv:
        paths = [Path(a) for a in argv]
    else:
        paths = sorted(Path("calls").glob("*.jsonl"))

    if not paths:
        print("No call files. Make a call first, or pass a path.", file=sys.stderr)
        return 1

    calls = []
    for path in paths:
        if path.is_file():
            meta, events = read_events([path])
            if events:
                calls.append({"meta": meta, "events": events, "filename": path.name})

    if not calls:
        print(f"No quality events in {len(paths)} file(s). "
              "Quality events are only recorded from new calls.", file=sys.stderr)
        return 1

    all_events = []
    for call in calls:
        all_events.extend(call["events"])

    metrics = compute_all(all_events)

    print(f"\n{len(calls)} call(s) with quality data\n")
    print(f"{'Metric':<30} {'Priority':<10} {'Value':>10} {'Detail'}")
    print("-" * 75)

    for key, (label, priority) in METRIC_LABELS.items():
        m = metrics[key]
        val = m.get("value")
        if val is None:
            val_str = "\u2014"
        elif key == "avg_turns_to_resolution":
            val_str = f"{val:.1f}"
        else:
            val_str = f"{val:.0%}"

        detail_parts = []
        for dk in ("n", "successes", "matches", "valid", "grounded",
                    "recovered", "retained", "min", "max"):
            if dk in m:
                detail_parts.append(f"{dk}={m[dk]}")
        detail = ", ".join(detail_parts)

        print(f"{label:<30} {priority:<10} {val_str:>10} {detail}")

    print()
    return 0
