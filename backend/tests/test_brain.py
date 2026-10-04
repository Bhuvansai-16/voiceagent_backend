"""The idempotency-key window is the one piece of non-trivial logic in brain.py.

If it breaks, a retried or barge-in-interrupted reset creates a second reset for
a real person. Worth a test even though the rest of Day 5 is not built yet.
"""

from backend.worker import brain
from backend.worker.brain import ServiceDeskAgent


def test_same_employee_reuses_key_inside_window():
    agent = ServiceDeskAgent()
    assert agent._idempotency_key("E1042") == agent._idempotency_key("E1042")


def test_different_employees_get_different_keys():
    agent = ServiceDeskAgent()
    assert agent._idempotency_key("E1042") != agent._idempotency_key("E1043")


def test_key_rotates_once_the_window_expires():
    agent = ServiceDeskAgent()
    first = agent._idempotency_key("E1042")
    # Age the stored entry rather than patching the clock globally.
    key, issued = agent._reset_keys["E1042"]
    agent._reset_keys["E1042"] = (key, issued - brain._RESET_KEY_TTL - 1)
    assert agent._idempotency_key("E1042") != first
