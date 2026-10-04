"""An in-process store, for tests that should not need a network.

Deliberately mirrors PostgresStore's *semantics*, not just its signatures: the
agent and flow upserts merge rather than replace, because `jsonb ||` does, and a
double that replaced where the real one merges would let a test pass while
production loses fields.

`tests/test_store_contract.py` runs the same suite against both to keep that
honest.
"""

import copy


class MemoryStore:
    def __init__(self) -> None:
        self._agents: dict[str, dict] = {}
        self._flows: dict[str, dict] = {}
        self._connectors: dict[str, dict] = {}

    # --- agents -------------------------------------------------------------

    async def agent_overrides(self) -> dict[str, dict]:
        return copy.deepcopy(self._agents)

    async def upsert_agent(self, agent_id: str, fields: dict) -> dict:
        merged = {**self._agents.get(agent_id, {}), **fields}
        self._agents[agent_id] = merged
        return copy.deepcopy(merged)

    async def delete_agent(self, agent_id: str) -> bool:
        return self._agents.pop(agent_id, None) is not None

    # --- flows --------------------------------------------------------------

    async def flows(self) -> list[dict]:
        return copy.deepcopy(list(self._flows.values()))

    async def upsert_flow(self, flow: dict) -> dict:
        flow_id = flow["id"]
        merged = {**self._flows.get(flow_id, {}), **flow}
        self._flows[flow_id] = merged
        return copy.deepcopy(merged)

    async def delete_flow(self, flow_id: str) -> bool:
        return self._flows.pop(flow_id, None) is not None

    # --- connectors ---------------------------------------------------------

    async def connectors(self) -> list[dict]:
        return copy.deepcopy([self._connectors[k] for k in sorted(self._connectors)])

    async def upsert_connector(self, connector: dict) -> dict:
        cid = connector["id"]
        existing = self._connectors.get(cid)
        merged = {**(existing or {}), **connector}
        if existing is None:
            merged["connected"] = bool(connector.get("connected", False))
        else:
            # PostgresStore's ON CONFLICT updates definition only, so an upsert
            # must never silently re-connect or disconnect an existing row.
            # set_connector_connected is the only way to change that flag.
            merged["connected"] = existing.get("connected", False)
        self._connectors[cid] = merged
        return copy.deepcopy(merged)

    async def set_connector_connected(self, connector_id: str, connected: bool) -> bool:
        if connector_id not in self._connectors:
            return False
        self._connectors[connector_id]["connected"] = connected
        return True

    async def delete_connector(self, connector_id: str) -> bool:
        return self._connectors.pop(connector_id, None) is not None
