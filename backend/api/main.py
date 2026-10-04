"""Voice service desk API application assembly.

The routes live in backend/api/routers/*, split by concern:
    servicedesk.py  agent-facing contract (employees, tickets, password reset)
    documents.py    RAG document upload/query
    connectors.py   Composio connector management
    flows.py        Visual Flow Studio, delegating to core.flow_compiler
    session.py      browser tokens, outbound calls, telephony status
    agents.py       agent gallery read/save/delete
    catalog.py      models, voices, providers the picker may offer
    telemetry.py    log feed, event stream, trace analytics
    diagnostics.py  integration health and raw service-desk state

The `/_debug/*` routes support operational diagnostics and workbench telemetry.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from backend.eventloop import use_selector_loop_on_windows

use_selector_loop_on_windows()

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.api.events import _recent, publish, MAX_STREAM_SECONDS  # noqa: F401  (re-exported for tests)
from backend.api.routers import (
    agents, catalog, connectors, diagnostics, documents,
    flows, servicedesk, session, telemetry,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The schema is applied by scripts/migrate.py over the direct endpoint, not
    # here: DDL through PgBouncer's transaction pooling is not reliable, and a
    # web process racing to create tables on every boot is its own problem.
    from backend.store.pool import close_pool, configured, open_pool

    if configured():
        await open_pool()
        # core/ must not import backend.store, so the connection source is
        # injected here rather than imported there.
        from backend.core import docstore
        from backend.store.pool import get_pool

        docstore.bind(lambda: get_pool().connection())
    else:
        logger.warning(
            "NEON_CONNECTION_URI is not set, so every database-backed endpoint "
            "will fail. Set it in .env."
        )
    try:
        yield
    finally:
        await close_pool()


logger = logging.getLogger("backend.api")

app = FastAPI(title="Voice Service Desk API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_calls(request: Request, call_next):
    # Only agent-facing calls belong in the operational event feed.
    if request.url.path.startswith(("/_debug", "/favicon", "/docs", "/openapi")) or request.url.path == "/":
        return await call_next(request)
    started = time.perf_counter()
    response = await call_next(request)
    publish(
        {
            "kind": "tool_call",
            "method": request.method,
            "path": request.url.path,
            "query": str(request.url.query),
            "status": response.status_code,
            "ms": round((time.perf_counter() - started) * 1000, 1),
        }
    )
    return response


app.include_router(servicedesk.router)
app.include_router(documents.router)
app.include_router(connectors.router)
app.include_router(flows.router)
app.include_router(session.router)
app.include_router(agents.router)
app.include_router(catalog.router)
app.include_router(telemetry.router)
app.include_router(diagnostics.router)
