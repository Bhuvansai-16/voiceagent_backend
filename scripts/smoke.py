"""Check that every provider, key, and model identifier actually works.

No microphone and no LiveKit room needed. Synthesizes speech with Deepgram
Aura-2, feeds that audio straight back into Deepgram Flux, and asks Gemini to
pick a tool. If this passes, the voice loop's parts are all real.

    uv run python scripts/smoke.py
"""

import asyncio
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from livekit import rtc  # noqa: E402
from livekit.agents import llm as agents_llm  # noqa: E402
from livekit.agents.utils import http_context  # noqa: E402

load_dotenv()

from backend.worker.brain import INSTRUCTIONS, ServiceDeskAgent  # noqa: E402
from backend.worker.main import build_llm, build_stt, build_tts  # noqa: E402

SPOKEN = "I need to reset my password, my employee ID is E1042."
# What Flux actually returns for the sentence above.
AS_HEARD = "I need to reset my password. My employee ID is e one zero four two."

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  {PASS if ok else FAIL}  {name}: {detail}")


async def _synthesize(tts, text):
    frames, ttfb = [], None
    started = time.perf_counter()
    async for chunk in tts.synthesize(text):
        if ttfb is None:
            ttfb = (time.perf_counter() - started) * 1000
        frames.append(chunk.frame)
    return frames, ttfb


async def check_tts():
    print("\nDeepgram Aura-2 (TTS)")
    tts = build_tts()

    # The first call pays for TLS handshake and connection setup. That number is
    # not what a caller experiences mid-conversation, so measure the second one.
    _, cold = await _synthesize(tts, "warming up")
    frames, ttfb = await _synthesize(tts, SPOKEN)

    if not frames:
        record("synthesize", False, "no audio returned")
        return None

    samples = sum(f.samples_per_channel for f in frames)
    rate = frames[0].sample_rate
    record(
        "synthesize",
        True,
        f"{len(frames)} frames, {samples / rate:.2f}s audio @ {rate}Hz",
    )
    # 500ms is a self-imposed slice of the plan's 800ms turn budget, not a
    # Deepgram guarantee. Measured warm TTFB sits around 350-420ms.
    record(
        "time to first audio byte",
        ttfb < 500,
        f"{ttfb:.0f}ms warm (cold start was {cold:.0f}ms)",
    )
    return frames


NUMBER_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def _digits(text: str) -> str:
    """'e one zero four two' -> 'e1042', so the check matches how Flux really hears it."""
    words = re.split(r"[\s,.\-]+", text.lower())
    return "".join(NUMBER_WORDS.get(w, w) for w in words if w)


def _silence(sample_rate: int, seconds: float):
    n = int(sample_rate * seconds)
    return rtc.AudioFrame(bytes(n * 2), sample_rate, 1, n)


async def check_stt(frames):
    print("\nDeepgram Flux (STT)")
    if not frames:
        record("recognize", False, "skipped, no audio from TTS")
        return
    # Flux is the /listen/v2 websocket API — streaming only, no one-shot recognize.
    # Aura-2 emits 24kHz and STTv2 assumes 16kHz. Piping one straight into the
    # other means telling Flux the real rate, or it silently transcribes nothing.
    # In a live call LiveKit handles this; only this direct loop has to care.
    stt = build_stt(sample_rate=frames[0].sample_rate)
    stream = stt.stream()

    best = ""
    started = time.perf_counter()

    async def collect():
        nonlocal best
        async for event in stream:
            # Flux only emits FINAL_TRANSCRIPT once it decides the turn ended,
            # which needs trailing silence. Interim and preflight carry the same
            # text and always arrive, so take the longest thing we see.
            if event.alternatives and len(event.alternatives[0].text) > len(best):
                best = event.alternatives[0].text

    task = asyncio.create_task(collect())
    for frame in frames:
        stream.push_frame(frame)
    stream.push_frame(_silence(frames[0].sample_rate, 1.0))
    stream.end_input()

    try:
        await asyncio.wait_for(task, timeout=30)
    except asyncio.TimeoutError:
        task.cancel()
    finally:
        await stream.aclose()

    elapsed = (time.perf_counter() - started) * 1000
    record("recognize", bool(best), f'"{best}" in {elapsed:.0f}ms')
    record(
        "employee ID survived the round trip",
        "e1042" in _digits(best),
        "E1042 recovered" if "e1042" in _digits(best) else f"got {_digits(best)[-16:]!r}",
    )


async def _turn(llm, ctx, tools):
    started = time.perf_counter()
    ttft, text, called = None, "", []
    async for chunk in llm.chat(chat_ctx=ctx, tools=tools):
        if ttft is None:
            ttft = (time.perf_counter() - started) * 1000
        delta = getattr(chunk, "delta", None)
        if delta is None:
            continue
        text += delta.content or ""
        for call in delta.tool_calls or []:
            called.append(getattr(call, "name", "?"))
    return ttft, text.strip(), called


async def check_llm():
    print("\nGemini (LLM) + tool selection")
    llm = build_llm()
    tools = list(ServiceDeskAgent().tools)
    record("tools registered", len(tools) == 3, f"{len(tools)} function tools")

    ctx = agents_llm.ChatContext.empty()
    ctx.add_message(role="system", content=INSTRUCTIONS)
    # Exactly how Flux transcribes it — spelled out, not "E1042". See NOTES.md.
    ctx.add_message(role="user", content=AS_HEARD)

    # Confirmation is prompt-enforced only, so it is a probabilistic property.
    # One sample proves nothing — run it several times and demand every one.
    attempts, confirmed, ttft, text = 3, 0, None, ""
    for _ in range(attempts):
        probe_ctx = ctx.copy()
        t, text, called = await _turn(llm, probe_ctx, tools)
        ttft = ttft or t
        if not called:
            confirmed += 1
    record("chat", ttft is not None, f"first token {ttft:.0f}ms" if ttft else "no response")
    record(
        "asks before writing",
        confirmed == attempts,
        f"{confirmed}/{attempts} runs asked first; last said {text[:60]!r}",
    )

    ctx.add_message(role="assistant", content=text)
    ctx.add_message(role="user", content="Yes, go ahead.")
    ttft2, text2, called2 = await _turn(llm, ctx, tools)
    record(
        "calls reset_password once confirmed",
        "reset_password" in called2,
        f"called {called2}" if called2 else f"no tool call, said {text2[:80]!r}",
    )
    # Plan section 9 budgets 800ms for the whole turn. First token is only one
    # stage of it, so anything close to that number is already a problem.
    record(
        "first token inside turn budget",
        ttft2 is not None and ttft2 < 800,
        f"{ttft2:.0f}ms warm (first call was {ttft:.0f}ms)",
    )


async def main() -> int:
    print(f"Speaking: {SPOKEN!r}")
    # Plugins share an aiohttp session owned by the worker. Outside a job we
    # have to open that context ourselves.
    async with http_context.open():
        frames = await check_tts()
        await check_stt(frames)
        await check_llm()

    failed = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    for _, name, detail in failed:
        print(f"  FAIL {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
