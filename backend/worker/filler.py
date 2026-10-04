"""What the agent says while a tool is still running.

Plan section 148 asks for a short acknowledgment the moment a tool starts, spoken
in parallel with the work rather than after it, varied so it does not sound
robotic.

LiveKit does the hard part: `RunContext.with_filler` plays audio directly to the
caller without touching the chat context, so the LLM never sees these lines and
never starts quoting them back. All this module owns is the wording.
"""

import os
import random

# Deliberately short. These are spoken over a gap of well under a second, and a
# long filler is worse than the silence it replaces.
PHRASES = (
    "Let me pull that up.",
    "One moment.",
    "Checking that now.",
    "Just a sec.",
    "Looking that up now.",
)

# Longer forms, for the rare case where the backend is genuinely slow and the
# first filler has already played.
FOLLOW_UPS = (
    "Still working on it.",
    "Almost there.",
    "Won't be a moment.",
)

# How long a tool may run before the caller hears anything. Below this, staying
# quiet sounds better than filling. Plan section 154 wants a filler when the
# backend takes 800ms, so this has to sit comfortably under that.
DELAY = float(os.getenv("FILLER_DELAY", "0.4"))
INTERVAL = float(os.getenv("FILLER_INTERVAL", "6"))


def phrases(rng: random.Random | None = None):
    """A step -> phrase function, starting at a random point in the pool.

    Back-to-back tool calls in one conversation therefore do not open with the
    same words, which is the thing that makes an agent sound canned.
    """
    rand = rng or random
    start = rand.randrange(len(PHRASES))

    def pick(step: int) -> str:
        if step == 0:
            return PHRASES[start]
        return FOLLOW_UPS[(step - 1) % len(FOLLOW_UPS)]

    return pick
