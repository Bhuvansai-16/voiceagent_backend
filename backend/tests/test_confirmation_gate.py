"""The write gate must make an unconfirmed reset impossible, not unlikely.

gemini-2.5-flash-lite ignored the prompt's confirmation instruction in 3 runs out
of 3 and reset a password unasked. These tests assert the property the prompt
could not: no backend call happens without a user turn in between.
"""

from contextlib import asynccontextmanager

import pytest

from backend.worker import brain
from backend.worker.brain import ServiceDeskAgent


class FakeContext:
    """Stands in for RunContext, and counts how often filler was armed."""

    def __init__(self) -> None:
        self.filler_armed = 0

    @asynccontextmanager
    async def with_filler(self, source, *, delay=0, interval=None, max_steps=None):
        self.filler_armed += 1
        yield


@pytest.fixture
def agent(monkeypatch):
    """An agent whose backend calls are recorded instead of made."""
    calls: list[tuple] = []

    async def fake_reset(employee_id, idempotency_key):
        calls.append(("reset_password", employee_id))
        return {"status": "sent", "sent_to": "p***@corp.com"}

    async def fake_create(employee_id, title, description=""):
        calls.append(("create_ticket", employee_id, title))
        return {"id": "INC9999", "status": "open"}

    async def passthrough(kind, coro):
        return await coro

    monkeypatch.setattr(brain.tools, "reset_password", fake_reset)
    monkeypatch.setattr(brain.tools, "create_ticket", fake_create)
    monkeypatch.setattr(brain.tools, "timed", passthrough)

    a = ServiceDeskAgent()
    a.calls = calls
    a.ctx = FakeContext()
    return a


async def reset(agent, employee_id="E1042"):
    # FunctionTool is callable and already bound, so this is the same path the
    # framework takes when the model asks for the tool.
    return await agent.reset_password(agent.ctx, employee_id)


async def make_ticket(agent, employee_id="E1042", title="Broken laptop"):
    return await agent.create_ticket(agent.ctx, employee_id, title)


async def user_speaks(agent):
    await agent.on_user_turn_completed(None, None)


@pytest.mark.asyncio
async def test_first_call_does_not_reset_anything(agent):
    await user_speaks(agent)
    out = await reset(agent)
    assert agent.calls == []
    assert "NOT DONE YET" in out


@pytest.mark.asyncio
async def test_calling_twice_in_one_turn_still_does_not_reset(agent):
    """The bypass a prompt cannot stop: model calls the tool, ignores the
    refusal, and immediately calls it again without the caller saying anything."""
    await user_speaks(agent)
    await reset(agent)
    await reset(agent)
    await reset(agent)
    assert agent.calls == []


@pytest.mark.asyncio
async def test_reset_runs_after_the_caller_speaks(agent):
    await user_speaks(agent)
    await reset(agent)
    await user_speaks(agent)  # "yes, go ahead"
    out = await reset(agent)
    assert agent.calls == [("reset_password", "E1042")]
    assert "sent" in out


@pytest.mark.asyncio
async def test_approval_is_spent_and_does_not_authorise_a_second_reset(agent):
    await user_speaks(agent)
    await reset(agent)
    await user_speaks(agent)
    await reset(agent)
    assert len(agent.calls) == 1

    await reset(agent)  # same turn, already spent
    assert len(agent.calls) == 1


@pytest.mark.asyncio
async def test_stale_approval_expires_instead_of_lingering(agent):
    """Caller declines, conversation moves on, model retries later."""
    await user_speaks(agent)
    await reset(agent)
    await user_speaks(agent)  # "no, actually don't"
    await user_speaks(agent)  # talks about something else
    await user_speaks(agent)

    assert await reset(agent) and agent.calls == []


@pytest.mark.asyncio
async def test_approving_one_employee_does_not_approve_another(agent):
    await user_speaks(agent)
    await reset(agent, "E1042")
    await user_speaks(agent)
    await reset(agent, "E1043")
    assert agent.calls == []


@pytest.mark.asyncio
async def test_approving_a_reset_does_not_approve_a_ticket(agent):
    await user_speaks(agent)
    await reset(agent)
    await user_speaks(agent)
    await make_ticket(agent)
    assert agent.calls == []


@pytest.mark.asyncio
async def test_refusing_a_write_does_not_arm_filler(agent):
    """The gate returns before any work starts, so the caller must not hear
    "let me pull that up" immediately followed by a request to confirm."""
    await user_speaks(agent)
    await reset(agent)
    assert agent.ctx.filler_armed == 0


@pytest.mark.asyncio
async def test_an_approved_write_arms_filler(agent):
    await user_speaks(agent)
    await reset(agent)
    await user_speaks(agent)
    await reset(agent)
    assert agent.ctx.filler_armed == 1


@pytest.mark.asyncio
async def test_ticket_creation_is_gated_the_same_way(agent):
    await user_speaks(agent)
    await make_ticket(agent)
    assert agent.calls == []

    await user_speaks(agent)
    await make_ticket(agent)
    assert agent.calls == [("create_ticket", "E1042", "Broken laptop")]
