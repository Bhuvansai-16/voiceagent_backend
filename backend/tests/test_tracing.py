"""Span mapping and the latency report.

The stage names asserted here are the contract from plan section 171. If a test
in this file has to change, Week 2's eval harness breaks too.
"""

import json
import types
from pathlib import Path

import pytest

from backend.worker import tracing
from backend.analytics import latency as latency_report


_next_id = iter(range(1, 10_000))


def span(name, *, ms=100.0, parent=None, **attrs):
    start = 1_000_000_000
    return types.SimpleNamespace(
        name=name,
        attributes=attrs,
        start_time=start,
        end_time=start + int(ms * 1_000_000),
        context=types.SimpleNamespace(span_id=next(_next_id)),
        parent=types.SimpleNamespace(span_id=parent) if parent else None,
    )


def test_span_identity_is_recorded_for_correlation(written):
    rows = written(span("user_turn", ms=100, parent=0xABC))
    assert rows[0]["span_id"] is not None
    assert rows[0]["parent_id"] == f"{0xABC:016x}"


def test_a_span_without_context_still_produces_a_row(written):
    """Identity is a nice-to-have. A span shaped differently than expected must
    not cost the whole trace."""
    bare = types.SimpleNamespace(
        name="user_turn", attributes={}, start_time=0, end_time=100_000_000
    )
    rows = written(bare)
    assert rows and rows[0]["stage"] == "user.speech"
    assert rows[0]["span_id"] is None


@pytest.fixture
def written(tmp_path):
    path = tmp_path / "call.jsonl"
    proc = tracing.JsonlSpanProcessor(path)

    def emit(*spans):
        for s in spans:
            proc.on_end(s)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    return emit


def test_user_turn_is_not_reported_as_a_latency(written):
    """LiveKit's user_turn span covers the caller speaking. Calling that
    turn.total made a 13s pause look like a 13s wait."""
    rows = written(span("user_turn", ms=13498))
    assert [(r["stage"], r["ms"]) for r in rows] == [("user.speech", 13498.0)]


def test_user_turn_also_carries_the_stt_timings(written):
    """`eou_detection` never opens when turn_detection is "stt", because Flux
    decides end-of-turn itself. Reading stt.finalize only from that span left it
    permanently empty, which a report cannot tell apart from an instant stage."""
    rows = written(span(
        "user_turn", ms=1450,
        **{"lk.end_of_turn_delay": 0.48, "lk.transcription_delay": 0.31},
    ))
    assert {r["stage"]: r["ms"] for r in rows} == {
        "user.speech": 1450.0,
        "stt.finalize": 480.0,
        "stt.transcript": 310.0,
    }


def test_user_turn_without_stt_attributes_still_reports_the_turn(written):
    rows = written(span("user_turn", ms=1450))
    assert [r["stage"] for r in rows] == ["user.speech"]


# ---------- derived reply latency ----------

def turn_rows(*, speech_end, tts_start, ttfb_ms):
    return [
        {"stage": "user.speech", "ms": 1000, "start": speech_end - 1, "end": speech_end},
        {"stage": "tts.first_byte", "ms": ttfb_ms, "start": tts_start, "end": tts_start + 2},
    ]


def test_reply_latency_spans_the_wait_plus_time_to_first_byte():
    # caller stops at t=10, TTS request starts at t=10.9, first byte 300ms later
    rows = turn_rows(speech_end=10.0, tts_start=10.9, ttfb_ms=300)
    assert latency_report.reply_latencies(rows) == [1200.0]


def test_audio_from_before_a_turn_is_not_credited_to_it():
    """The agent's previous reply must not be paired with the turn that
    interrupted it, which would report a negative or absurdly small wait."""
    rows = [
        {"stage": "tts.first_byte", "ms": 300, "start": 5.0, "end": 7.0},
        *turn_rows(speech_end=10.0, tts_start=10.9, ttfb_ms=300),
    ]
    assert latency_report.reply_latencies(rows) == [1200.0]


def test_a_turn_the_agent_never_answered_is_skipped_not_zeroed():
    rows = [{"stage": "user.speech", "ms": 900, "start": 9.0, "end": 10.0}]
    assert latency_report.reply_latencies(rows) == []


def test_overlapping_turns_are_skipped_not_guessed():
    """Real trace: turn A ended at 18.85s, turn B at 20.68s, and the first audio
    started at 21.48s. Attributing it to A reported a 12-second wait for a turn
    whose reply actually arrived in 806ms. When a second turn ends before the
    audio, ownership is ambiguous and the later turn should claim it."""
    rows = [
        {"stage": "user.speech", "ms": 900, "start": 17.0, "end": 18.85},
        {"stage": "user.speech", "ms": 900, "start": 19.5, "end": 20.68},
        {"stage": "tts.first_byte", "ms": 294, "start": 21.48, "end": 23.0},
    ]
    # One measurement, belonging to the turn that actually preceded the audio.
    assert latency_report.reply_latencies(rows) == [1094.0]


def test_each_reply_is_used_once():
    rows = [
        *turn_rows(speech_end=10.0, tts_start=10.5, ttfb_ms=200),
        *turn_rows(speech_end=20.0, tts_start=21.0, ttfb_ms=200),
    ]
    assert latency_report.reply_latencies(rows) == [700.0, 1200.0]


def test_llm_span_yields_both_first_token_and_complete(written):
    rows = written(span("llm_node", ms=900, **{"lk.response.ttft": 0.695}))
    assert {r["stage"]: r["ms"] for r in rows} == {
        "llm.first_token": 695.0,   # from the attribute, in seconds
        "llm.complete": 900.0,      # from the span's own duration
    }


def test_retry_attempts_are_not_counted_as_completions(written):
    """`llm_request_run` is one attempt inside llm_node's retry loop. Mapping it
    to llm.complete counted every call twice, and a four-retry call five times."""
    rows = written(
        span("llm_node", ms=900, **{"lk.response.ttft": 0.695}),
        span("llm_request_run", ms=300),
        span("llm_request_run", ms=280),
    )
    stages = [r["stage"] for r in rows]
    assert stages.count("llm.complete") == 1
    assert stages.count("llm.attempt") == 2


def test_llm_span_without_ttft_still_reports_completion(written):
    """Realtime models report no time-to-first-token. That must not cost us the
    completion timing as well."""
    rows = written(span("llm_node", ms=900))
    assert [r["stage"] for r in rows] == ["llm.complete"]


def test_tool_span_is_named_after_the_tool(written):
    rows = written(span("function_tool", ms=42, **{"lk.function_tool.name": "reset_password"}))
    assert rows[0]["stage"] == "tool.reset_password"


def test_unknown_spans_are_ignored(written):
    assert written(span("drain_agent_activity", ms=10)) == []


def test_tts_first_byte_comes_from_the_attribute_not_the_duration(written):
    rows = written(span("tts_node", ms=4000, **{"lk.response.ttfb": 0.36}))
    assert [(r["stage"], r["ms"]) for r in rows] == [("tts.first_byte", 360.0)]


# ---------- report ----------

def write_call(tmp_path, rows) -> Path:
    path = tmp_path / "call.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_percentile_uses_nearest_rank():
    assert latency_report.percentile([10, 20, 30, 40], 50) == 20
    assert latency_report.percentile([10, 20, 30, 40], 95) == 40
    assert latency_report.percentile([5], 95) == 5


def test_report_counts_turns_over_budget(tmp_path, capsys):
    # Three replies: 500ms, 1400ms, 1600ms after the caller stopped.
    rows = [{"stage": "llm.first_token", "ms": 695}]
    for i, wait_ms in enumerate((500, 1400, 1600)):
        end = 100.0 + i * 60
        rows += [
            {"stage": "user.speech", "ms": 900, "start": end - 0.9, "end": end},
            {"stage": "tts.first_byte", "ms": 200,
             "start": end + (wait_ms - 200) / 1000, "end": end + 3},
        ]
    assert latency_report.main([str(write_call(tmp_path, rows))]) == 0
    out = capsys.readouterr().out
    assert "turns over 800ms: 2/3" in out
    assert "llm.first_token" in out


def test_report_says_so_when_no_turn_was_measured(tmp_path, capsys):
    """Printing nothing would read as "no turn was slow", which is a different
    claim from "no turn was measured"."""
    path = write_call(tmp_path, [{"stage": "llm.first_token", "ms": 695}])
    assert latency_report.main([str(path)]) == 0
    assert "budget was not evaluated" in capsys.readouterr().out


def test_report_survives_a_corrupt_line(tmp_path, capsys):
    path = tmp_path / "call.jsonl"
    path.write_text(
        '{"stage":"user.speech","ms":900,"start":9.1,"end":10.0}\n'
        "not json\n"
        '{"stage":"tts.first_byte","ms":200,"start":10.7,"end":13.0}\n',
        encoding="utf-8",
    )
    assert latency_report.main([str(path)]) == 0
    assert "turns over 800ms: 1/1" in capsys.readouterr().out


def test_report_fails_loudly_when_there_is_nothing_to_read(tmp_path, capsys):
    assert latency_report.main([str(tmp_path / "missing.jsonl")]) == 1


def test_old_format_turn_totals_are_dropped_not_relabelled(tmp_path, capsys):
    """Traces written before turn.total was derived hold speech duration under
    that name. Printing them beneath a heading promising reply latency is worse
    than printing nothing."""
    path = write_call(tmp_path, [
        {"stage": "turn.total", "ms": 13498},
        {"stage": "llm.first_token", "ms": 695},
    ])
    assert latency_report.main([str(path)]) == 0
    captured = capsys.readouterr()
    assert "older trace format" in captured.err
    assert "13498" not in captured.out
    assert "budget was not evaluated" in captured.out
