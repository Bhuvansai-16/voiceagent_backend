"""What a store must be able to do.

Two implementations satisfy this, both real: PostgresStore against Neon, and
MemoryStore for tests that should not need a network. The Protocol is not an
abstraction over a single case, and a shared contract suite runs against both —
a test double that quietly diverges from Postgres semantics is worse than none.

Async throughout, because Postgres is. A synchronous store would mean a blocking
network call inside an `async def` request handler, which is the class of bug
this restructure exists to remove.
"""

from typing import Protocol


class Store(Protocol):
    # --- agents -------------------------------------------------------------
    async def agent_overrides(self) -> dict[str, dict]:
        """Every configured agent, as {agent_id: fields}.

        These layer on top of core.personas.BUILT_IN, which is the floor. A
        store returning nothing still leaves a working gallery.
        """

    async def upsert_agent(self, agent_id: str, fields: dict) -> dict:
        """Merge `fields` into the agent's stored fields and return the result.

        Merge, not replace: a PUT carrying only {id, provider} must not wipe
        that agent's label, purpose, skills and tools.
        """

    async def delete_agent(self, agent_id: str) -> bool:
        """True if a row was removed, False if there was nothing to remove."""

    # --- flows --------------------------------------------------------------
    async def flows(self) -> list[dict]: ...

    async def upsert_flow(self, flow: dict) -> dict: ...

    async def delete_flow(self, flow_id: str) -> bool: ...

    # --- connectors ---------------------------------------------------------
    async def connectors(self) -> list[dict]:
        """Stored connector state, keyed by id, as {**definition, connected}."""

    async def upsert_connector(self, connector: dict) -> dict: ...

    async def set_connector_connected(self, connector_id: str, connected: bool) -> bool: ...

    async def delete_connector(self, connector_id: str) -> bool: ...
