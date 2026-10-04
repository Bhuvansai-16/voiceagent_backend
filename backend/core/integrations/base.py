"""Provider abstraction for the service desk backend.

The voice agent never talks to a concrete system directly: it talks to a
ServiceDeskProvider. Which provider is used is decided once, at import time,
from the SERVICEDESK_PROVIDER env var (see backend/worker/integrations/__init__.py).
"""

from abc import ABC, abstractmethod


class ServiceDeskProvider(ABC):
    """The four operations the agent's tools need. All async, all dict-shaped."""

    @abstractmethod
    async def get_employee(self, employee_id: str) -> dict | None:
        """Return {id, name, email, locked} or None if unknown."""

    @abstractmethod
    async def list_tickets(self, employee_id: str, status: str = "open") -> list[dict]:
        """Return [{id, title, status, updated}, ...]."""

    @abstractmethod
    async def create_ticket(self, employee_id: str, title: str, description: str = "") -> dict | None:
        """Return {id, status} or None if the employee is unknown."""

    @abstractmethod
    async def reset_password(self, employee_id: str, idempotency_key: str) -> dict | None:
        """Return {reset_id, status, sent_to} or None if the employee is unknown.

        Implementations MUST honour idempotency_key: replaying the same key
        returns the original result instead of triggering a second reset. That
        is what makes a barge-in during a reset safe (see tests/test_barge_in.py).
        """

    async def aclose(self) -> None:
        """Release any held resources (HTTP clients, connections)."""
