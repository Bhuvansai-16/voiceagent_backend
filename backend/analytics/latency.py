"""Per-stage p50/p95 from a traced call (plan section 168).

    uv run python scripts/latency_report.py calls/20260727-094752-AJ_xxx.jsonl
    uv run python scripts/latency_report.py calls/*.jsonl
    uv run python scripts/latency_report.py            # every call in calls/

Stage names come from backend/worker/tracing.py and are the contract plan
section 171 fixes for Week 2's eval harness.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

BUDGET_MS = 800          # plan section 9, for the whole turn
TURN_STAGE = "turn.total"

# Printed in this order when present, so a report reads in pipeline order rather
# than alphabetically. Anything unrecognised is appended afterwards.
ORDER = ["stt.transcript", "stt.finalize", "llm.first_token", "llm.complete",
         "llm.attempt", "tts.first_byte", "tts.attempt", TURN_STAGE, "user.speech"]


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank. On the handful of turns one call produces, interpolating
    invents precision that is not there."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(pct / 100 * len(ordered))))
    return ordered[rank - 1]


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  skipped {path.name}:{line_no}, not JSON", file=sys.stderr)
    return rows


def reply_latencies(rows: list[dict]) -> list[float]:
    """How long the caller waited, per turn.

    Measured from the moment their turn was committed to the first byte of audio
    coming back. There is no shared id linking a user turn to the reply it
    caused, so this pairs each `user.speech` with the first `tts.first_byte`
    that started after it. `ttfb` is measured from the start of the TTS request,
    so first audio lands at that span's start plus ttfb.

    Turns where the agent never spoke are skipped rather than counted as zero.
    """
    speech = sorted((r for r in rows if r["stage"] == "user.speech" and r.get("end")),
                    key=lambda r: r["end"])
    audio = sorted((r for r in rows
                    if r["stage"] == "tts.first_byte" and r.get("start") and r.get("ms")),
                   key=lambda r: r["start"])

    ends = [t["end"] for t in speech]
    out, used = [], 0

    for turn in speech:
        while used < len(audio) and audio[used]["start"] < turn["end"]:
            used += 1                      # belongs to an earlier turn
        if used >= len(audio):
            break
        reply = audio[used]

        # If another turn ended between this one and the audio, ownership is
        # genuinely ambiguous: the caller spoke again before the agent answered,
        # so the reply may be to either turn. Guessing here produced a measured
        # 12-second "wait" for a turn whose own reply arrived in 806ms. Skip the
        # turn without consuming the audio, and let the later turn claim it.
        if any(turn["end"] < end <= reply["start"] for end in ends):
            continue

        first_audio_at = reply["start"] + reply["ms"] / 1000
        out.append(round((first_audio_at - turn["end"]) * 1000, 1))
        used += 1

    return out


def collect(rows: list[dict]) -> dict[str, list[float]]:
    stages: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("ms") is not None:
            stages[row["stage"]].append(float(row["ms"]))

    # Traces written before turn.total was derived recorded it straight from
    # LiveKit's user_turn span, so the value is how long the caller talked. It
    # has no start/end and no user.speech alongside it. Printing it under a
    # heading that promises reply latency would be worse than printing nothing.
    stale = TURN_STAGE in stages and not any(r["stage"] == "user.speech" for r in rows)
    if stale:
        print(f"  ignoring {len(stages[TURN_STAGE])} {TURN_STAGE} rows from an older"
              " trace format, where the value was speech duration, not latency.",
              file=sys.stderr)
        del stages[TURN_STAGE]

    derived = reply_latencies(rows)
    if derived:
        stages[TURN_STAGE] = derived
    return stages


def resolve(args: list[str]) -> list[Path]:
    if args:
        paths = [Path(a) for a in args]
    else:
        paths = sorted(Path("calls").glob("*.jsonl"))
    return [p for p in paths if p.is_file()]


def main(argv: list[str]) -> int:
    paths = resolve(argv)
    if not paths:
        print("No call files. Make a call first, or pass a path.", file=sys.stderr)
        print("Traces land in calls/ unless TRACE_DIR says otherwise.", file=sys.stderr)
        return 1

    stages = collect(read_rows(paths))
    if not stages:
        print(f"No timed spans in {len(paths)} file(s).", file=sys.stderr)
        return 1

    known = [s for s in ORDER if s in stages]
    rest = sorted(s for s in stages if s not in ORDER)
    rows = known + rest

    width = max(len(s) for s in rows)
    print(f"\n{len(paths)} call(s), {sum(len(v) for v in stages.values())} spans\n")
    print(f"{'stage'.ljust(width)}    n     p50      p95      max")
    print("-" * (width + 34))
    for stage in rows:
        v = stages[stage]
        print(f"{stage.ljust(width)}  {len(v):>3}  {percentile(v,50):>7.0f}  "
              f"{percentile(v,95):>7.0f}  {max(v):>7.0f}")

    turns = stages.get(TURN_STAGE, [])
    print()
    print(f"{TURN_STAGE} is the wait between a caller finishing and hearing audio.")
    print("user.speech is how long they talked, which is not a latency.")
    print()
    if turns:
        over = [t for t in turns if t > BUDGET_MS]
        print(f"turns over {BUDGET_MS}ms: {len(over)}/{len(turns)}"
              f"  ({100 * len(over) / len(turns):.0f}%)")
    else:
        # Silence here would read as "no turns were slow", which is not the same
        # as "no turn was measured".
        print(f"No {TURN_STAGE} could be derived, so the budget was not evaluated.")
    return 0
