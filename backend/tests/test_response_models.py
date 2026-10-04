"""`response_model` must not silently remove fields from a response.

This is the one failure mode the rest of the suite cannot see. FastAPI filters
every response through its declared model and drops any key the model does not
declare - no error, no warning, no failing assertion. A response model that
forgets a field simply deletes it from the API, and every existing test still
passes because they assert on the fields they know about, never on the absence
of ones they do not.

So this compares each handler's own return value against what the HTTP layer
actually serialized. A key in the first and not the second is a field the model
ate.

Covers fifteen no-argument GET endpoints. Eleven need nothing but the process
itself, so this stays part of the suite that runs with no network and no
credentials; the four that read Postgres skip when no pool is open.
"""

import asyncio
import inspect

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.store import set_store
from backend.store.memory import MemoryStore


def _api_routes(routes):
    """Flatten the router tree.

    `app.routes` does not contain APIRoutes directly in this FastAPI version -
    each `include_router` leaves an opaque `_IncludedRouter` that keeps the real
    thing on `.original_router`.
    """
    for route in routes:
        if isinstance(route, APIRoute):
            yield route
        elif hasattr(route, "original_router"):
            yield from _api_routes(route.original_router.routes)


def _key_paths(value, prefix=""):
    """Every key in a nested structure, so a drop inside a list is caught too."""
    found = set()
    if isinstance(value, dict):
        for key, inner in value.items():
            found.add(prefix + key)
            found |= _key_paths(inner, f"{prefix}{key}.")
    elif isinstance(value, list):
        for inner in value[:3]:
            found |= _key_paths(inner, f"{prefix}[].")
    return found


def _no_argument_get_routes():
    for route in _api_routes(app.routes):
        if "GET" not in route.methods:
            continue
        parameters = inspect.signature(route.endpoint).parameters.values()
        if any(p.default is inspect.Parameter.empty for p in parameters):
            continue  # needs a path parameter we would have to invent
        yield route


@pytest.fixture(scope="module")
def offline_client():
    """A client that deliberately does not run the application lifespan.

    Entering TestClient as a context manager would open a second connection
    pool, and psycopg pools bind to the event loop that opened them - so its
    shutdown collides with the session-scoped `client` fixture's pool and the
    module errors out with "attached to a different loop". Skipping lifespan
    means this module never opens one of its own. The four database-backed
    handlers then run only if some other fixture already opened a pool, and
    skip cleanly when none has.
    """
    set_store(MemoryStore())
    try:
        yield TestClient(app)
    finally:
        set_store(None)


@pytest.mark.parametrize(
    "route", list(_no_argument_get_routes()), ids=lambda r: r.path
)
def test_response_model_keeps_every_field_the_handler_returns(route, offline_client):
    try:
        returned = asyncio.run(route.endpoint())
    except Exception as exc:  # needs a database; covered against live Postgres
        pytest.skip(f"{route.path} needs a backing service: {type(exc).__name__}")

    if not isinstance(returned, dict):
        pytest.skip(f"{route.path} does not return a plain dict")

    response = offline_client.get(route.path)
    assert response.status_code == 200, response.text

    dropped = _key_paths(returned) - _key_paths(response.json())
    assert not dropped, (
        f"response_model on {route.path} removed {sorted(dropped)} from the "
        f"response. Declare the field on the model, or give the model "
        f'ConfigDict(extra="allow") if the shape is genuinely open.'
    )
