"""The layering rule, enforced rather than documented.

`backend/core` must not import livekit, fastapi, or any sibling layer. That rule
is the whole point of the split: without it the API drags every LiveKit plugin
into its process just to read a provider data table, and plugin registration is
only legal on the main thread. That is how this codebase accumulated roughly
fifteen lazy in-function imports, each one a workaround for a dependency that
should never have existed.

If this test fails, do not add another lazy import. Move the offending code out
of core/, or move what it needs into core/.
"""

import ast
from pathlib import Path

import pytest

CORE = Path(__file__).parents[1] / "core"
FORBIDDEN = ("livekit", "fastapi", "backend.api", "backend.worker", "backend.store")


def _imported_modules(path: Path) -> set[str]:
    """Every module name this file imports, including inside functions."""
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # node.level > 0 is a relative import, which cannot escape core/.
            if node.module and not node.level:
                found.add(node.module)
    return found


CORE_FILES = sorted(CORE.rglob("*.py"))


def test_core_package_is_not_empty():
    """A glob that matched nothing would make the parametrised test vacuous."""
    assert len(CORE_FILES) >= 10, f"only found {len(CORE_FILES)} files under core/"


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: p.name)
def test_core_imports_no_forbidden_layer(path):
    offenders = sorted(
        name
        for name in _imported_modules(path)
        for bad in FORBIDDEN
        if name == bad or name.startswith(bad + ".")
    )
    assert not offenders, f"core/{path.name} imports {offenders}"


def test_production_api_and_worker_entry_points_import():
    from backend.api.main import app
    from backend.worker import main

    assert app.title
    assert callable(main.entrypoint)
