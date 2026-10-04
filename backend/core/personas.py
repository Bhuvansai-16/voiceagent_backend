"""The agents a caller can choose between.

Adding web search and a shell to the service desk was a mistake: its prompt says
it does exactly three things, and handing it a terminal made that untrue. Separate
personas fix that, and they contain blast radius. Only `general` gets a shell, so
a misheard command during a password reset has nowhere to go.

A persona is instructions plus a tool allow-list. The allow-list is enforced with
`Agent.update_tools`, not by asking the prompt nicely, for the same reason the
write gate is code: a prompt is a suggestion.
"""

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from backend.core.skills import SKILLS_DIR, load_skill  # noqa: F401  (re-exported)

logger = logging.getLogger("service-desk.personas")

PROJECT_ROOT = Path(__file__).parents[2]


# Shared by every persona. How to speak, not what to do.
VOICE_RULES = """\
You are speaking on a phone call, not writing. Keep replies to one or two short
sentences. No lists, no markdown, no bullet points. Read identifiers as words
where you can: "ticket forty-one" rather than "I N C zero zero four one", unless
the caller needs the exact characters.

Speech recognition writes spoken digits as words. "E1042" reaches you as "e one
zero four two", or "E ten forty two", or "e 1 0 4 2". Join those back together
yourself before using them. "oh" means zero. Callers often drop the letter and
say only "one zero four two"; every employee ID starts with E, so add it back.

If what you heard does not resolve cleanly, read it back and ask them to confirm.

If you are given remembered context about this caller, use it naturally. Do not
announce that you remembered something or read the memory back to them.

When the caller says goodbye, says they need nothing else, or the reason for the
call is clearly finished and they are happy, say a short goodbye and then end the
call. Do not leave the line open waiting for them to hang up, and do not end it
while they might still be talking.
"""

CONFIRM_WRITES = """\
Before opening a ticket, resetting a password, or running a command, say what
you are about to do and wait for the caller to confirm. Never do any of them
without a clear yes.

Confirming is done by calling the tool, not by asking in words first. As soon as
you have what a write needs, call the tool. It will not perform the action: it
returns the sentence for you to say and waits. Say that sentence, and when the
caller agrees, call the same tool again with the same arguments to carry it out.

Asking "shall I go ahead?" without calling the tool costs the caller an extra
round of saying yes, because their first yes then only reaches the step you
should already have taken.
"""

EXTERNAL_WRITE_RULES = """\
When sending an email, posting a channel message, or creating a record in an external workspace app (such as Gmail or Slack), summarize the recipient, subject, and body to the caller in one short sentence and ask for their verbal confirmation before triggering the send action. Once they say yes, execute the tool. For read actions (searching emails, fetching threads, listing events), execute immediately and summarize the results clearly in 1-2 spoken sentences.
"""

TICKET_TOOLS = ("get_tickets", "create_ticket", "reset_password")
WEB_TOOLS = ("search_web",)
MEMORY_TOOLS = ("remember_about_caller",)
SANDBOX_TOOLS = ("run_command",)
RAG_TOOLS = ("search_documents",)
# Every persona can hang up. On a browser session there is no call to end and
# the tool says so; on a phone call, an agent that cannot hang up leaves the
# line open and billing until the caller works out that they should.
CALL_TOOLS = ("end_call",)


# Every tool the code actually implements. Configuration may choose from these
# and nothing else: a config file that could name a tool into existence would
# just fail at call time, and one that could grant any tool would make the
# per-agent allow-list decorative.
KNOWN_TOOLS = TICKET_TOOLS + WEB_TOOLS + MEMORY_TOOLS + SANDBOX_TOOLS + RAG_TOOLS + CALL_TOOLS


# run_command is arbitrary code execution reached through speech recognition,
# which mishears. It stays off unless the environment says otherwise, so editing
# a config file cannot hand a shell to an agent by itself.
DANGEROUS_TOOLS = frozenset(SANDBOX_TOOLS)


@dataclass(frozen=True)
class Persona:
    id: str
    label: str
    blurb: str                    # one line, shown on the grid
    purpose: str                  # goes into the prompt
    tools: tuple[str, ...] = field(default_factory=tuple)
    # None means "whatever LLM_MODEL says". Per-agent models are the point of the
    # gallery, and they also multiply the free tier: Gemini's 20 requests a day
    # is per model, so three agents on three models is three separate budgets.
    model: str | None = None
    provider: str | None = None
    # Aura-2 voice. None means "whatever DEEPGRAM_TTS_MODEL says". Per-agent so a
    # service desk and a research assistant do not have to sound identical.
    voice: str | None = None
    # Speech providers, resolved the same way as the LLM: config first, then
    # environment. None means "whatever STT_PROVIDER / TTS_PROVIDER says".
    # There is no tts_model on purpose: nvidia.TTS has no model parameter and
    # Deepgram's model *is* its voice. Only Cartesia separates the two and it
    # reads CARTESIA_MODEL from the environment. Add one when a second TTS
    # provider needs a model distinct from its voice.
    stt_provider: str | None = None
    stt_model: str | None = None
    tts_provider: str | None = None
    # For the `openai-compatible` provider only. api_key_env names an
    # environment variable; it never holds a key. The store is not a secrets
    # manager and agent rows are readable by anyone with database access.
    base_url: str | None = None
    api_key_env: str | None = None
    # Composio toolkits this agent wants. Each caller connects their own
    # accounts, so this says which apps to offer, not whose account to use.
    toolkits: tuple[str, ...] = field(default_factory=tuple)
    # Extra instruction files loaded from skills/<name>.md at startup.
    skills: tuple[str, ...] = field(default_factory=tuple)
    # Per-agent thinking toggle for models that support it (e.g. NVIDIA Nemotron).
    # False by default to save multi-second latency on voice calls.
    enable_thinking: bool = False

    def resolved_provider(self) -> str:
        """The provider actually used, config first then environment.

        Both build_llm and the prompt need this answer and they must not be able
        to disagree: a prompt built for one provider against another model is a
        silent behaviour change, not an error.
        """
        return (self.provider or os.getenv("LLM_PROVIDER", "google")).strip().lower()

    @property
    def instructions(self) -> str:
        parts = []
        # NVIDIA's Nemotron models support internal thinking/reasoning. For real-time
        # voice calls, /no_think is prepended by default to save multi-second latency.
        # The per-agent enable_thinking toggle controls this; the global env var
        # ENABLE_NVIDIA_THINKING is checked as a fallback so existing deployments
        # that set it keep working. Nebius hosts the same Nemotron weights.
        if self.resolved_provider() in ("nvidia", "nebius"):
            global_on = os.getenv("ENABLE_NVIDIA_THINKING", "").strip().lower() in ("1", "true", "yes")
            if not self.enable_thinking and not global_on:
                parts.append("/no_think")
        parts.append(self.purpose)
        for skill in self.skills:
            if body := load_skill(skill):
                parts.append(body)
        if self.toolkits:
            # Tell the model what connected toolkits it can use
            tk_names = [
                "Gmail" if "gmail" in tk.lower() or "dfsagmjexo9u" in tk.lower()
                else tk.replace("ac_", "").capitalize()
                for tk in self.toolkits
                if tk
            ]
            parts.append(
                f"You have active access to external workspace integrations: {', '.join(tk_names)}. "
                "Use your provided function-calling tools to look up information, send messages, or perform actions in these services when requested."
            )
            parts.append(EXTERNAL_WRITE_RULES)
        parts.append(VOICE_RULES)
        if set(self.tools) & set(TICKET_TOOLS + SANDBOX_TOOLS):
            parts.append(CONFIRM_WRITES)
        return "\n".join(parts)

    def granted_tools(self) -> tuple[str, ...]:
        """Tools after safety filtering. This, not `tools`, is what is wired up."""
        allow_shell = os.getenv("ALLOW_SANDBOX_TOOL", "").strip().lower() in ("1", "true", "yes")
        return tuple(
            t for t in self.tools
            if t in KNOWN_TOOLS and (allow_shell or t not in DANGEROUS_TOOLS)
        )


IT_SUPPORT = Persona(
    id="it-support",
    label="IT Support",
    blurb="Tickets and password resets for this company.",
    tools=TICKET_TOOLS + CALL_TOOLS,
    purpose="""\
You are the voice of an IT service desk.

You can do exactly three things:
1. Look up someone's support tickets.
2. Open a new support ticket.
3. Trigger a password reset.

If asked for anything else, say briefly that you only handle those three and
offer to open a ticket about it. You cannot search the web and you cannot run
commands. Do not offer to.

Every tool needs an employee ID, formatted like E1042. If you do not have one,
ask for it before doing anything else.
""",
)

GENERAL = Persona(
    id="general",
    label="General purpose",
    blurb="Tickets, web search, memory, and a sandboxed terminal.",
    tools=TICKET_TOOLS + WEB_TOOLS + MEMORY_TOOLS + SANDBOX_TOOLS + CALL_TOOLS,
    purpose="""\
You are a general assistant speaking on a phone call. You can look up and open
support tickets, trigger password resets, search the web, remember things about
the caller, and run shell commands in an isolated sandbox.

Do not search the web for this company's tickets, employees or passwords. Those
live in your tools, and the web does not know them.

Only run a command when the caller has asked for one. Never run a command to
answer something you could answer yourself, and never invent a command they did
not ask for. The sandbox has no network access and nothing in it survives the
call.

Ticket and password tools need an employee ID formatted like E1042.
""",
)

RESEARCH = Persona(
    id="research",
    label="Research",
    blurb="Looks things up on the web and remembers what you care about.",
    tools=WEB_TOOLS + MEMORY_TOOLS + CALL_TOOLS,
    purpose="""\
You are a research assistant speaking on a phone call. You search the web and
answer from what you find.

Say where an answer came from when it matters, and say plainly when the search
did not settle the question rather than filling the gap yourself. If a claim
looked contested across sources, say so.

You have no access to this company's tickets, employees or passwords, and you
cannot run commands. If asked for those, say so and suggest the IT Support
agent.
""",
)

RAG = Persona(
    id="rag",
    label="Document RAG",
    blurb="Answers questions from your uploaded documents and images.",
    tools=RAG_TOOLS + MEMORY_TOOLS + CALL_TOOLS,
    purpose="""\
You are a knowledge assistant speaking on a phone call. You search uploaded documents
and OCR-extracted images using your document search tool to answer the caller's questions.

Base your answers strictly on what the document search tool returns.
If the search returns no relevant information, say plainly that the uploaded documents
do not contain the answer.
""",
)

BUILT_IN = (IT_SUPPORT, GENERAL, RESEARCH, RAG)

# Fields a config file may change. `tools` is here but still passes through
# granted_tools, so configuration selects from what exists rather than inventing
# capabilities. `id` is deliberately absent: it is the key.
CONFIGURABLE = frozenset(
    {"label", "blurb", "purpose", "tools", "model", "provider", "toolkits", "skills",
     "voice", "enable_thinking", "stt_provider", "stt_model", "tts_provider",
     "base_url", "api_key_env"}
)

_TUPLE_FIELDS = ("tools", "toolkits", "skills")


def _apply(base: Persona, overrides: dict) -> Persona:
    clean = {}
    for key, value in overrides.items():
        if key not in CONFIGURABLE:
            logger.warning("ignoring unknown or protected field %r for agent %r", key, base.id)
            continue
        clean[key] = tuple(value) if key in _TUPLE_FIELDS and value is not None else value
    return replace(base, **clean)


def resolve(overrides: dict[str, dict]) -> tuple[Persona, ...]:
    """The built-in agents with stored overrides layered on top.

    Pure: takes the override records, returns personas. No I/O, which is what
    keeps this module free of any dependency on where those records live. The
    caller fetches them from the store and passes them in.

    The built-ins are the floor rather than a default that configuration
    replaces, so a store that is empty, unreachable, or half-written still
    yields a working gallery instead of none.
    """
    agents = {p.id: p for p in BUILT_IN}

    for agent_id, entry in (overrides or {}).items():
        if not agent_id:
            logger.warning("skipping stored agent with no id")
            continue
        base = agents.get(agent_id)
        if base is None:
            # A wholly new agent still needs the fields a Persona cannot infer.
            missing = [f for f in ("label", "blurb", "purpose") if not entry.get(f)]
            if missing:
                logger.warning("stored agent %r missing %s, skipping", agent_id, missing)
                continue
            base = Persona(
                id=agent_id,
                label=entry["label"],
                blurb=entry["blurb"],
                purpose=entry["purpose"],
                tools=CALL_TOOLS,
            )
        agents[agent_id] = _apply(base, {k: v for k, v in entry.items() if k != "id"})

    return tuple(agents.values())


# The floor, available the moment this module is imported. No file is read and
# no database is touched at import time; apply_overrides replaces these once the
# caller has fetched the stored records.
ALL: tuple[Persona, ...] = BUILT_IN
BY_ID: dict[str, Persona] = {p.id: p for p in ALL}
DEFAULT: Persona = BY_ID["it-support"]


def apply_overrides(overrides: dict[str, dict]) -> tuple[Persona, ...]:
    """Install a resolved set of personas as the process-wide view.

    The API calls this per request and the worker once per job, which is how a
    console edit reaches a call without a restart. Two processes reading the
    same database agree; two processes reading their own copy of a JSON file,
    as this did before, do not.
    """
    global ALL, BY_ID, DEFAULT
    ALL = resolve(overrides)
    BY_ID = {p.id: p for p in ALL}
    DEFAULT = BY_ID.get("it-support", ALL[0])
    return ALL


def from_room(room_name: str) -> Persona:
    """Rooms are named `<persona-id>-<random>`, which is how the grid choice
    reaches the worker without any dispatch configuration.

    Longest id first, so `it-support` is not shadowed by a shorter id that
    happens to be a prefix of it.
    """
    for persona in sorted(ALL, key=lambda p: -len(p.id)):
        if room_name.startswith(f"{persona.id}-"):
            return persona
    return DEFAULT
