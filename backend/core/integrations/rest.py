"""Env-configurable REST adapter for any service desk exposing a JSON API.

Nothing about Jira, ServiceNow or anything else is hardcoded. Point it at a
system via env vars and, if its payloads differ from the agent's internal
shape, map the fields:

    SERVICEDESK_BASE_URL          required, e.g. https://desk.example.com/api
    SERVICEDESK_API_KEY           sent as a bearer token by default
    SERVICEDESK_AUTH_HEADER       default Authorization
    SERVICEDESK_AUTH_SCHEME       default Bearer; set empty for a raw key
    SERVICEDESK_EMPLOYEE_PATH     default /employees/{id}
    SERVICEDESK_TICKETS_PATH      default /tickets
    SERVICEDESK_CREATE_TICKET_PATH  default /tickets
    SERVICEDESK_RESET_PATH        default /password-reset
    SERVICEDESK_TICKETS_KEY       default tickets (list wrapper key)
    SERVICEDESK_TIMEOUT_SECONDS   default 10

Response field names are assumed to match the internal shape ({id, name, email,
locked} for employees, {id, title, status, updated} for tickets, {reset_id,
status, sent_to} for resets). Systems that differ get a thin subclass of this
class overriding `_map_employee` / `_map_ticket` / `_map_reset` rather than a
new provider.
"""

import os

import httpx

from backend.core.integrations.base import ServiceDeskProvider


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


class GenericRestProvider(ServiceDeskProvider):
    def __init__(self) -> None:
        base_url = os.getenv("SERVICEDESK_BASE_URL", "http://localhost:8002")
        headers: dict[str, str] = {}
        api_key = os.getenv("SERVICEDESK_API_KEY", "")
        if api_key:
            header = _env("SERVICEDESK_AUTH_HEADER", "Authorization")
            scheme = _env("SERVICEDESK_AUTH_SCHEME", "Bearer")
            headers[header] = f"{scheme} {api_key}".strip()
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=float(_env("SERVICEDESK_TIMEOUT_SECONDS", "10")),
        )
        self._employee_path = _env("SERVICEDESK_EMPLOYEE_PATH", "/employees/{id}")
        self._tickets_path = _env("SERVICEDESK_TICKETS_PATH", "/tickets")
        self._create_ticket_path = _env("SERVICEDESK_CREATE_TICKET_PATH", "/tickets")
        self._reset_path = _env("SERVICEDESK_RESET_PATH", "/password-reset")
        self._tickets_key = _env("SERVICEDESK_TICKETS_KEY", "tickets")

    # --- field mapping hooks -------------------------------------------------

    def _map_employee(self, body: dict) -> dict:
        return {
            "id": body.get("id") or body.get("employee_id"),
            "name": body.get("name") or body.get("full_name"),
            "email": body.get("email"),
            "locked": bool(body.get("locked", body.get("account_locked", False))),
        }

    def _map_ticket(self, body: dict) -> dict:
        return {
            "id": body.get("id") or body.get("key") or body.get("ticket_id"),
            "title": body.get("title") or body.get("summary") or body.get("subject"),
            "status": body.get("status") or body.get("state"),
            "updated": body.get("updated") or body.get("updated_at"),
        }

    def _map_reset(self, body: dict) -> dict:
        return {
            "reset_id": body.get("reset_id") or body.get("id"),
            "status": body.get("status"),
            "sent_to": body.get("sent_to") or body.get("destination"),
        }

    # --- ServiceDeskProvider --------------------------------------------------

    async def get_employee(self, employee_id: str) -> dict | None:
        r = await self._client.get(self._employee_path.format(id=employee_id))
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return self._map_employee(r.json())

    async def list_tickets(self, employee_id: str, status: str = "open") -> list[dict]:
        r = await self._client.get(
            self._tickets_path, params={"employee_id": employee_id, "status": status}
        )
        r.raise_for_status()
        body = r.json()
        rows = body[self._tickets_key] if isinstance(body, dict) else body
        return [self._map_ticket(t) for t in rows]

    async def create_ticket(self, employee_id: str, title: str, description: str = "") -> dict | None:
        r = await self._client.post(
            self._create_ticket_path,
            json={"employee_id": employee_id, "title": title, "description": description},
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        body = r.json()
        return {"id": body.get("id") or body.get("key"), "status": body.get("status", "open")}

    async def reset_password(self, employee_id: str, idempotency_key: str) -> dict | None:
        r = await self._client.post(
            self._reset_path,
            json={"employee_id": employee_id, "idempotency_key": idempotency_key},
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return self._map_reset(r.json())

    async def aclose(self) -> None:
        await self._client.aclose()
