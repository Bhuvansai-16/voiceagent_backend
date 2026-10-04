"""Barge-in arriving while a write is in flight (plan section 152).

The dangerous case: the caller interrupts mid-`reset_password`, LiveKit cancels
the tool task, and the reset may or may not have reached the backend. It must not
end up applied twice. The idempotency key from Day 1 is the mechanism.
"""

import asyncio
from contextlib import asynccontextmanager

import pytest

from backend.worker import brain
from backend.worker.brain import ServiceDeskAgent


class FakeContext:
    @asynccontextmanager
    async def with_filler(self, source, *, delay=0, interval=None, max_steps=None):
        yield


@pytest.fixture
def agent(monkeypatch):
    """Backend calls record their idempotency key and are slow enough to interrupt."""
    keys: list[str] = []

    async def slow_reset(employee_id, idempotency_key):
        keys.append(idempotency_key)
        await asyncio.sleep(0.3)  # long enough for a barge-in to land mid-call
        return {"status": "sent", "sent_to": "p***@corp.com"}

    async def passthrough(kind, coro):
        return await coro

    monkeypatch.setattr(brain.tools, "reset_password", slow_reset)
    monkeypatch.setattr(brain.tools, "timed", passthrough)

    a = ServiceDeskAgent()
    a.keys = keys
    a.ctx = FakeContext()
    return a


async def approve_and_start_reset(agent, employee_id="E1042"):
    """Get past the write gate, then return the still-running reset task."""
    await agent.on_user_turn_completed(None, None)
    await agent.reset_password(agent.ctx, employee_id)  # proposes, writes nothing
    await agent.on_user_turn_completed(None, None)  # caller says yes
    return asyncio.create_task(agent.reset_password(agent.ctx, employee_id))


@pytest.mark.asyncio
async def test_barge_in_mid_reset_does_not_produce_a_second_reset(agent):
    task = await approve_and_start_reset(agent)
    await asyncio.sleep(0.05)  # reset is in flight

    task.cancel()  # the caller interrupts; LiveKit cancels the tool
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(agent.keys) == 1, "the interrupted call had already reached the backend"

    # The interrupt spent the approval, so the agent re-proposes and the caller
    # confirms again before anything is retried.
    await agent.on_user_turn_completed(None, None)
    await agent.reset_password(agent.ctx, "E1042")  # re-proposes, writes nothing
    assert len(agent.keys) == 1

    await agent.on_user_turn_completed(None, None)  # "yes, do it"
    await agent.reset_password(agent.ctx, "E1042")

    assert len(agent.keys) == 2, "expected a retry to reach the backend"
    assert agent.keys[0] == agent.keys[1], (
        "retry used a fresh idempotency key, so the backend would create a second "
        "reset for one request"
    )


@pytest.mark.asyncio
async def test_interrupted_write_cannot_be_retried_within_the_same_turn(agent):
    """The gate's approval is spent by the first attempt. A model that retries
    immediately after a barge-in gets refused rather than firing again."""
    task = await approve_and_start_reset(agent)
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    out = await agent.reset_password(agent.ctx, "E1042")

    assert len(agent.keys) == 1, "a same-turn retry reached the backend"
    assert "NOT DONE YET" in out


@pytest.mark.asyncio
async def test_a_different_employee_gets_its_own_key(agent):
    """Key reuse must be scoped to the employee, or one caller's interrupted
    reset would suppress a genuine reset for someone else."""
    task = await approve_and_start_reset(agent, "E1042")
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await agent.on_user_turn_completed(None, None)
    await agent.reset_password(agent.ctx, "E1043")  # proposes
    await agent.on_user_turn_completed(None, None)
    await agent.reset_password(agent.ctx, "E1043")  # confirmed, runs

    assert agent.keys[0] != agent.keys[-1]
