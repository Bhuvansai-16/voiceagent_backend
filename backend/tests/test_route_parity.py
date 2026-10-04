"""Every route that existed before the router split must still be served.

`frontend/src/api.js` pins every one of these paths by hand. A path that moves
or disappears during a refactor is not a test failure over there — it is a
silent 404 in the console with no server-side error.

The baseline is read from the OpenAPI schema rather than `app.routes`, because
this FastAPI version nests included routers instead of flattening them:
`app.routes` reports four docs routes plus five opaque router objects, so
walking it would make this test pass while checking nothing.
"""

import json
from pathlib import Path

from backend.api.main import app

BASELINE = Path(__file__).parent / "_routes_baseline.json"


def _current() -> set[tuple[str, str]]:
    spec = app.openapi()
    return {(p, m.upper()) for p, ops in spec["paths"].items() for m in ops}


def test_no_route_was_lost_in_the_split():
    before = {tuple(pair) for pair in json.loads(BASELINE.read_text(encoding="utf-8"))}
    missing = before - _current()
    assert not missing, f"routes no longer served: {sorted(missing)}"


def test_the_baseline_is_not_empty():
    """A baseline that captured nothing would make the test above vacuous."""
    before = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert len(before) >= 38, f"baseline only has {len(before)} entries"
