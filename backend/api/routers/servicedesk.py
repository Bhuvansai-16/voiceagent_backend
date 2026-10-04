"""Agent-facing service desk contract endpoints."""

import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.api.db import _iso, _mask, execute, fetch_all, fetch_one
from backend.api.schemas import (
    EmployeeOut,
    ResetOut,
    TicketCreatedOut,
    TicketListOut,
)

router = APIRouter(tags=["Service desk"])


@router.get("/employees/{employee_id}", response_model=EmployeeOut)
async def get_employee(employee_id: str):
    row = await fetch_one(
        "SELECT id, name, email, locked FROM employees WHERE id = %s", (employee_id,)
    )
    if row is None:
        raise HTTPException(404, f"unknown employee {employee_id}")
    return {"id": row["id"], "name": row["name"], "email": row["email"],
            "locked": bool(row["locked"])}


@router.get("/tickets", response_model=TicketListOut)
async def list_tickets(
    employee_id: str,
    status: Literal["open", "all"] = "open",
):
    sql = "SELECT id, title, status, updated FROM tickets WHERE employee_id = %s"
    if status == "open":
        sql += " AND status = 'open'"
    sql += " ORDER BY updated DESC"
    rows = await fetch_all(sql, (employee_id,))
    return {"tickets": [{**r, "updated": _iso(r["updated"])} for r in rows]}


class NewTicket(BaseModel):
    employee_id: str
    title: str
    description: str = ""


@router.post("/tickets", status_code=201, response_model=TicketCreatedOut)
async def create_ticket(body: NewTicket):
    from backend.store.pool import get_pool

    async with get_pool().connection() as conn:
        exists = await (await conn.execute(
            "SELECT 1 FROM employees WHERE id = %s", (body.employee_id,)
        )).fetchone()
        if exists is None:
            raise HTTPException(404, f"unknown employee {body.employee_id}")

        # One statement, so two concurrent creates cannot pick the same number.
        # The old SELECT MAX(...) then INSERT was a race that SQLite's single
        # writer hid; Postgres has no such accident to rely on.
        row = await (await conn.execute(
            """
            INSERT INTO tickets (id, employee_id, title, description, status, updated)
            SELECT 'INC' || LPAD((COALESCE(MAX(CAST(SUBSTRING(id FROM 4) AS INTEGER)), 0) + 1)::text, 4, '0'),
                   %s, %s, %s, 'open', now()
              FROM tickets
            RETURNING id, status
            """,
            (body.employee_id, body.title, body.description),
        )).fetchone()
    return {"id": row["id"], "status": row["status"]}


class ResetRequest(BaseModel):
    employee_id: str
    idempotency_key: str = Field(min_length=1)


@router.post("/password-reset", response_model=ResetOut)
async def reset_password(body: ResetRequest):
    prior = await fetch_one(
        "SELECT reset_id, status, sent_to FROM password_resets WHERE idempotency_key = %s",
        (body.idempotency_key,),
    )
    if prior is not None:
        # Replay: same reset_id, 200 instead of 202. No second reset is created.
        # This is what makes a barge-in mid-reset safe.
        return {"reset_id": prior["reset_id"], "status": prior["status"],
                "sent_to": prior["sent_to"]}

    employee = await fetch_one(
        "SELECT email FROM employees WHERE id = %s", (body.employee_id,)
    )
    if employee is None:
        raise HTTPException(404, f"unknown employee {body.employee_id}")

    reset_id = f"PR-{uuid.uuid4().hex[:12]}"
    sent_to = _mask(employee["email"])
    await execute(
        "INSERT INTO password_resets"
        " (idempotency_key, reset_id, employee_id, status, sent_to, created)"
        " VALUES (%s, %s, %s, 'sent', %s, now())"
        " ON CONFLICT (idempotency_key) DO NOTHING",
        (body.idempotency_key, reset_id, body.employee_id, sent_to),
    )

    # The insert may have lost a race with an identical concurrent request. Read
    # back rather than reporting a reset_id that was never stored.
    stored = await fetch_one(
        "SELECT reset_id, status, sent_to FROM password_resets WHERE idempotency_key = %s",
        (body.idempotency_key,),
    )
    return JSONResponse(
        {"reset_id": stored["reset_id"], "status": stored["status"],
         "sent_to": stored["sent_to"]},
        status_code=202,
    )
