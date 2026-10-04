"""Picking a store, the same way integrations picks a service desk provider.

STORE_BACKEND=postgres (default) talks to Neon. STORE_BACKEND=memory is for
tests and for running the API with no database at all.
"""

import os

from backend.store.base import Store
from backend.store.memory import MemoryStore
from backend.store.postgres import PostgresStore

_BACKENDS: dict[str, type] = {
    "postgres": PostgresStore,
    "memory": MemoryStore,
}

_store: Store | None = None


def get_store() -> Store:
    """The process-wide store. One instance, created on first use."""
    global _store
    if _store is None:
        name = os.getenv("STORE_BACKEND", "postgres").strip().lower()
        if name not in _BACKENDS:
            raise ValueError(
                f"Unknown STORE_BACKEND {name!r}. Available: {sorted(_BACKENDS)}"
            )
        _store = _BACKENDS[name]()
    return _store


def set_store(store: Store | None) -> None:
    """Replace the process-wide store. For tests and for app startup."""
    global _store
    _store = store


__all__ = ["Store", "MemoryStore", "PostgresStore", "get_store", "set_store"]
