"""The agent: every tool it could have, narrowed by the persona it is running as.

All tools are defined here and registered automatically, so `backend/worker/personas.py`
takes them away rather than adding them. A persona whose prompt says "you cannot
run commands" is a suggestion; a persona without the tool cannot call it.

No LangGraph. LiveKit's AgentSession already runs exactly the shape the plan
allows in section 25 â€” one supervisor turn plus tool calls â€” so wrapping it in a
graph would add a hop and latency for no capability.
"""

import asyncio
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from livekit.agents import Agent, RunContext, function_tool

from backend.core import docstore, memory, personas, rag, sandbox, websearch
from backend.worker import filler, tools, tracing

logger = logging.getLogger("service-desk")

# Keyword → expected tool mapping for intent accuracy heuristic.
# When a user says something matching these patterns, we know which tool
# _should_ be called. This is a best-effort signal, not ground truth.
_INTENT_MAP = {
    "ticket": "get_tickets",
    "tickets": "get_tickets",
    "open a ticket": "create_ticket",
    "create a ticket": "create_ticket",
    "new ticket": "create_ticket",
    "file a ticket": "create_ticket",
    "raise a ticket": "create_ticket",
    "password": "reset_password",
    "reset": "reset_password",
    "reset password": "reset_password",
    "password reset": "reset_password",
    "search": "search_web",
    "look up": "search_web",
    "google": "search_web",
    "find online": "search_web",
    "document": "search_documents",
    "knowledge base": "search_documents",
    "run command": "run_command",
    "execute": "run_command",
    "terminal": "run_command",
    "shell": "run_command",
    "remember": "remember_about_caller",
    "goodbye": "end_call",
    "bye": "end_call",
    "hang up": "end_call",
    "that's all": "end_call",
    "done": "end_call",
}

_EMPLOYEE_ID_RE = re.compile(r"^E\d{4}$")


def _detect_intent(text: str) -> str | None:
    """Best-effort intent detection from user speech via keyword matching."""
    lower = text.lower()
    # Try longest matches first for better accuracy
    for phrase in sorted(_INTENT_MAP, key=len, reverse=True):
        if phrase in lower:
            return _INTENT_MAP[phrase]
    return None


def _validate_employee_id(employee_id: str) -> bool:
    """Check if an employee_id argument matches the expected E#### format."""
    return bool(_EMPLOYEE_ID_RE.match(employee_id))


@asynccontextmanager
async def _speak_while_working(context: RunContext):
    """Cover the gap while a tool runs, per plan section 148.

    LiveKit plays this straight to the caller and keeps it out of the chat
    context, so the model never sees the filler and never starts repeating it.
    Nothing is spoken at all if the tool returns inside `filler.DELAY`, which is
    the normal case against the local service â€” a filler on a fast call would sound
    worse than the silence it replaced.
    """
    async with context.with_filler(
        filler.phrases(),
        delay=filler.DELAY,
        interval=filler.INTERVAL,
        max_steps=4,
    ):
        yield

# Kept as a name because scripts/smoke.py imports it.
# The prompt itself now lives with the persona that owns it.
INSTRUCTIONS = personas.IT_SUPPORT.instructions

# ponytail: a 60s window per employee dedupes a retry or a barge-in-triggered
# re-call onto one idempotency key, which is what keeps a double reset from
# happening. Proper per-turn scoping lands with the Day 5 barge-in work.
_RESET_KEY_TTL = 60.0


CONFIRMATION_REQUIRED_TOOLS = frozenset({"create_ticket", "reset_password", "run_command"})


def _compress_for_voice(raw_text: str, max_chars: int = 450) -> str:
    """Condenses verbose search/doc text into speech-friendly summary chunks."""
    if not raw_text:
        return ""
    text = raw_text.strip()
    # Remove markdown link formatting [text](url) -> text
    import re
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Collapse multiple newlines/bullets into single spaces
    text = re.sub(r'\s+', ' ', text)
    if len(text) <= max_chars:
        return text
    # Truncate on sentence boundary
    truncated = text[:max_chars]
    last_period = max(truncated.rfind('. '), truncated.rfind('? '), truncated.rfind('! '))
    if last_period > 100:
        return truncated[:last_period + 1]
    return truncated + "..."


class ServiceDeskAgent(Agent):
    def __init__(
        self,
        caller_id: str = "anonymous",
        persona: personas.Persona | None = None,
        hang_up=None,
        trace_path: Path | None = None,
    ) -> None:
        self.persona = persona or personas.DEFAULT
        super().__init__(instructions=self.persona.instructions)

        allowed = set(self.persona.granted_tools())
        self._tools = [t for t in self._tools if t.info.name in allowed]
        self._chat_ctx = self._chat_ctx.copy(tools=self._tools)

        self._reset_keys: dict[str, tuple[str, float]] = {}
        self._turn = 0
        self._pending: dict[str, int] = {}
        self._caller_id = caller_id
        self._hang_up = hang_up
        self._trace_path = trace_path
        # Quality analytics tracking
        self._entities_seen: set[str] = set()   # entity IDs seen across turns
        self._last_tool_failed = False           # for recovery tracking
        self._tool_calls_this_call = 0
        self._tool_successes_this_call = 0
        self._sandbox = (
            sandbox.Sandbox()
            if sandbox.configured() and "run_command" in allowed
            else None
        )

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        """Counts real user turns, and pulls anything this caller said before.

        Recall runs on every turn and is capped at 200ms in `memory.recall`,
        returning empty on timeout or failure. A caller waiting on a memory
        lookup is worse off than one whose agent forgot something.
        """
        self._turn += 1

        spoken = getattr(new_message, "text_content", None) if new_message else None
        if not spoken:
            return

        # --- Quality analytics: user turn + intent ---
        intent = _detect_intent(spoken)
        tracing.write_event(
            self._trace_path, "turn.user",
            turn=self._turn, text=spoken[:500], intent=intent,
        )
        # Track entity references for context retention metric
        for word in spoken.split():
            if _EMPLOYEE_ID_RE.match(word):
                self._entities_seen.add(word)

        if not memory.configured():
            return

        remembered = await memory.recall(self._caller_id, spoken)
        if remembered:
            turn_ctx.add_message(
                role="system",
                content="You know this about the caller already: " + " ".join(remembered),
            )

    def _gate(self, action: str, spoken_summary: str) -> str | None:
        """Approval gate for write actions. None means go ahead.

        The prompt already asks the model to confirm before writing, and
        gemini-2.5-flash does. gemini-2.5-flash-lite ignores it every time (see
        so the prompt is not the guard — this is. A write runs only if
        the same action was proposed on the *previous* turn, which means the
        caller has since spoken. Two calls inside one turn cannot get through,
        and an approval does not keep for later: it is good for exactly the next
        turn and then goes stale.
        """
        proposed_on = self._pending.get(action)
        if proposed_on is not None and proposed_on == self._turn - 1:
            del self._pending[action]
            return None

        self._pending[action] = self._turn
        return (
            "NOT DONE YET. Nothing has been created or changed. Say this to the"
            f" caller, making clear it has not happened yet: {spoken_summary}"
            " Then stop and wait for their answer. Do not imply the work is done."
            " Call this tool again only after they agree. If they decline, say so"
            " and do not call it again."
        )

    def _guard_action(self, tool_name: str, action_key: str, spoken_summary: str) -> str | None:
        """Declarative confirmation policy interceptor."""
        if tool_name not in CONFIRMATION_REQUIRED_TOOLS:
            return None
        return self._gate(action_key, spoken_summary)

    def _idempotency_key(self, employee_id: str) -> str:
        key, issued = self._reset_keys.get(employee_id, ("", 0.0))
        if key and time.monotonic() - issued < _RESET_KEY_TTL:
            return key
        key = str(uuid.uuid4())
        self._reset_keys[employee_id] = (key, time.monotonic())
        return key

    def _emit_tool_event(self, tool_name: str, args: dict, result: str,
                         success: bool, args_valid: bool = True) -> None:
        """Emit a turn.tool_call quality event and track error recovery."""
        self._tool_calls_this_call += 1
        recovered_from_error = self._last_tool_failed and success
        if success:
            self._tool_successes_this_call += 1
        if recovered_from_error:
            tracing.write_event(
                self._trace_path, "turn.error_recovery",
                turn=self._turn, tool=tool_name,
            )
        self._last_tool_failed = not success
        tracing.write_event(
            self._trace_path, "turn.tool_call",
            turn=self._turn, tool=tool_name,
            args=args, result=result[:300],
            success=success, args_valid=args_valid,
            recovered_from_error=recovered_from_error,
        )

    @function_tool
    async def get_tickets(
        self,
        context: RunContext,
        employee_id: str,
        include_closed: bool = False,
    ) -> str:
        """Look up the support tickets belonging to an employee.

        Args:
            employee_id: The employee ID, format E followed by four digits, e.g. E1042.
            include_closed: True to include closed tickets, False for open ones only.
        """
        valid_id = _validate_employee_id(employee_id)
        async with _speak_while_working(context):
            employee = await tools.timed("tool.get_employee", tools.get_employee(employee_id))
            if employee is None:
                result = f"No employee found with ID {employee_id}. Ask them to repeat it."
                self._emit_tool_event("get_tickets",
                    {"employee_id": employee_id, "include_closed": include_closed},
                    result, success=False, args_valid=valid_id)
                return result

            tickets = await tools.timed(
                "tool.get_tickets",
                tools.list_tickets(employee_id, "all" if include_closed else "open"),
            )
        if not tickets:
            result = f"{employee['name']} has no {'' if include_closed else 'open '}tickets."
        else:
            listed = "; ".join(f"{t['id']} {t['title']} ({t['status']})" for t in tickets)
            result = f"{employee['name']} has {len(tickets)} ticket(s): {listed}"
        self._emit_tool_event("get_tickets",
            {"employee_id": employee_id, "include_closed": include_closed},
            result, success=True, args_valid=valid_id)
        return result

    @function_tool
    async def create_ticket(
        self,
        context: RunContext,
        employee_id: str,
        title: str,
        description: str = "",
    ) -> str:
        """Open a new support ticket. Only call this after the caller has confirmed.

        Args:
            employee_id: The employee ID, format E followed by four digits.
            title: A short summary of the problem, under ten words.
            description: What the caller said about the problem, in their words.
        """
        blocked = self._gate(
            f"create_ticket:{employee_id}:{title.strip().lower()}",
            f'"I\'ll open a ticket for {employee_id} about {title}. Shall I go ahead?"',
        )
        if blocked:
            return blocked

        valid_id = _validate_employee_id(employee_id)
        args = {"employee_id": employee_id, "title": title, "description": description}
        async with _speak_while_working(context):
            created = await tools.timed(
                "tool.create_ticket", tools.create_ticket(employee_id, title, description)
            )
        if created is None:
            result = f"No employee found with ID {employee_id}. Nothing was created."
            self._emit_tool_event("create_ticket", args, result, success=False, args_valid=valid_id)
            return result
        result = f"Opened ticket {created['id']} for {employee_id}, status open."
        self._emit_tool_event("create_ticket", args, result, success=True, args_valid=valid_id)
        return result

    @function_tool
    async def reset_password(self, context: RunContext, employee_id: str) -> str:
        """Trigger a password reset email. Only call this after the caller has confirmed.

        Args:
            employee_id: The employee ID, format E followed by four digits.
        """
        blocked = self._gate(
            f"reset_password:{employee_id}",
            f'"I\'m about to reset the password for {employee_id}. Is that right?"',
        )
        if blocked:
            return blocked

        valid_id = _validate_employee_id(employee_id)
        args = {"employee_id": employee_id}
        async with _speak_while_working(context):
            api_result = await tools.timed(
                "tool.reset_password",
                tools.reset_password(employee_id, self._idempotency_key(employee_id)),
            )
        if api_result is None:
            result = f"No employee found with ID {employee_id}. No reset was sent."
            self._emit_tool_event("reset_password", args, result, success=False, args_valid=valid_id)
            return result
        result = (
            f"Password reset {api_result['status']}, sent to {api_result['sent_to']}."
            " Tell them to check that inbox."
        )
        self._emit_tool_event("reset_password", args, result, success=True, args_valid=valid_id)
        return result

    @function_tool
    async def search_web(self, context: RunContext, query: str) -> str:
        """Search the web for current information. Not for company data.

        Args:
            query: What to search for, as a short phrase.
        """
        args = {"query": query}
        if not websearch.configured():
            result = "Web search is not configured. Say you cannot look that up."
            self._emit_tool_event("search_web", args, result, success=False)
            return result
        async with _speak_while_working(context):
            try:
                raw = await tools.timed("tool.search_web", websearch.search(query))
                result = _compress_for_voice(raw)
                self._emit_tool_event("search_web", args, result, success=True)
                return result
            except Exception as exc:
                result = f"The search failed: {exc}. Say you could not look it up."
                self._emit_tool_event("search_web", args, result, success=False)
                return result

    @function_tool
    async def search_documents(self, context: RunContext, query: str) -> str:
        """Search uploaded knowledge documents and user files for relevant information.

        Args:
            query: What to search for in the uploaded documents.
        """
        args = {"query": query}
        async with _speak_while_working(context):
            try:
                fallback = await docstore.search(query, caller_id=self._caller_id)
                raw = await tools.timed(
                    "tool.search_documents",
                    asyncio.to_thread(
                        rag.query_vector_store, query,
                        caller_id=self._caller_id, fallback_rows=fallback,
                    ),
                )
                result = _compress_for_voice(raw)
                self._emit_tool_event("search_documents", args, result, success=True)
                return result
            except Exception as exc:
                result = f"Document search error: {exc}"
                self._emit_tool_event("search_documents", args, result, success=False)
                return result

    @function_tool
    async def remember_about_caller(self, context: RunContext, fact: str) -> str:
        """Store one durable fact about this caller for future calls.

        Use for stable things: their name, their employee ID, their usual
        machine. Not for what they want right now.

        Args:
            fact: One short sentence, written about the caller in the third person.
        """
        args = {"fact": fact}
        if not memory.configured():
            result = "Memory is not configured. Do not tell the caller anything was saved."
            self._emit_tool_event("remember_about_caller", args, result, success=False)
            return result
        stored = await memory.remember(self._caller_id, fact)
        result = "Noted." if stored else "That could not be saved. Do not claim it was."
        self._emit_tool_event("remember_about_caller", args, result, success=stored)
        return result

    @function_tool
    async def run_command(self, context: RunContext, command: str) -> str:
        """Run a shell command in an isolated sandbox with no network access.

        Only when the caller explicitly asked for a command to be run.

        Args:
            command: The exact shell command the caller asked for.
        """
        args = {"command": command}
        if self._sandbox is None:
            result = "The sandbox is not configured. Say you cannot run commands."
            self._emit_tool_event("run_command", args, result, success=False)
            return result

        # Same code gate as a password reset, for a stronger reason. This is
        # arbitrary code execution reached through speech recognition, which
        # mishears. A prompt saying "only when asked" is a suggestion; the gate
        # means a misheard command cannot run without the caller hearing it read
        # back and agreeing.
        blocked = self._gate(
            f"run_command:{command.strip()}",
            f'"I\'m about to run: {command.strip()}. Shall I?"',
        )
        if blocked:
            return blocked

        async with _speak_while_working(context):
            result = await tools.timed("tool.run_command", self._sandbox.run(command))
            self._emit_tool_event("run_command", args, result, success=True)
            return result

    @function_tool
    async def end_call(self, context: RunContext) -> str:
        """Hang up. Only after saying goodbye, and only when the caller is done.

        Use when the caller says goodbye, says they need nothing else, or the
        reason for the call is finished and they have confirmed they are happy.
        """
        if self._hang_up is None:
            return "There is no call to end. Say goodbye but do not claim to have hung up."

        logger.info("agent ending the call")

        # Emit resolution event before hanging up
        tracing.write_event(
            self._trace_path, "turn.resolution",
            turn=self._turn,
            total_turns=self._turn,
            tool_calls=self._tool_calls_this_call,
            tool_successes=self._tool_successes_this_call,
            entities_seen=sorted(self._entities_seen),
        )

        # The goodbye has to finish playing first. Deleting the room mid-sentence
        # cuts the caller off, which reads as a dropped call rather than a
        # completed one.
        await context.wait_for_playout()
        await self._hang_up()
        return "Call ended."

    async def on_exit(self) -> None:
        # Emit resolution if end_call was never invoked (caller disconnected)
        if self._turn > 0 and self._tool_calls_this_call > 0:
            tracing.write_event(
                self._trace_path, "turn.resolution",
                turn=self._turn,
                total_turns=self._turn,
                tool_calls=self._tool_calls_this_call,
                tool_successes=self._tool_successes_this_call,
                entities_seen=sorted(self._entities_seen),
                graceful=False,
            )
        if self._sandbox is not None:
            await self._sandbox.close()
