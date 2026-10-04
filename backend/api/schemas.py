"""Request and response shapes for the API.

Seven handlers took a bare `body: dict`, so /docs showed no schema and the
payload the frontend sends was documented only in `api.js`. These declare it.

Three rules, settled by measurement rather than taste:

**Request models ignore unknown fields; they do not forbid them.** The agent
editor's Duplicate and Clone buttons both send `{...agent, id: newId}` where
`agent` came straight back from `GET /_debug/agents` - so the payload carries
`granted_tools` and `built_in`, which are not configurable. `extra="forbid"`
would turn both buttons into 422s. Pydantic's default drop matches what the
handlers already do, since they filter to `personas.CONFIGURABLE` anyway.

**Semantic validation stays in the handlers.** An unknown provider, a missing
API key and a voice belonging to another provider all return 400 with a
sentence naming the fix. Re-expressing those as field constraints would turn
them into 422s carrying pydantic's error format, and they are the errors a
user is most likely to see.

**`response_model` drops any field the model does not declare** - verified, not
assumed. So a shape is declared strictly only where the handler returns a
literal dict whose keys are all visible in the source; anything spreading
`**result`, or built from a database row, carries `extra="allow"`, which
preserves undeclared keys. Getting this backwards removes fields from the API
without failing a test.
"""

from pydantic import BaseModel, ConfigDict

# --- shared -----------------------------------------------------------------


class Ok(BaseModel):
    success: bool = True


class Deleted(BaseModel):
    deleted: str


# --- agents -----------------------------------------------------------------


class AgentIn(BaseModel):
    """One agent, as the console saves it.

    The fields mirror `personas.CONFIGURABLE` exactly; `id` is the key rather
    than a setting. Everything else is optional because a partial PUT merges -
    that is what keeps `{id, model}` from wiping an agent's purpose and tools.
    """

    id: str
    label: str | None = None
    blurb: str | None = None
    purpose: str | None = None
    tools: list[str] | None = None
    toolkits: list[str] | None = None
    skills: list[str] | None = None
    model: str | None = None
    provider: str | None = None
    voice: str | None = None
    enable_thinking: bool | None = None
    stt_provider: str | None = None
    stt_model: str | None = None
    tts_provider: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None


class AgentOut(BaseModel):
    """What configuration asked for, and what the agent will really get.

    `tools` and `granted_tools` differ when a config names an unknown tool or
    asks for the shell without ALLOW_SANDBOX_TOOL. Both are sent so the console
    can show the difference rather than imply there is none.
    """

    id: str
    label: str
    blurb: str
    purpose: str
    tools: list[str]
    granted_tools: list[str]
    model: str | None = None
    provider: str | None = None
    voice: str | None = None
    toolkits: list[str]
    skills: list[str]
    enable_thinking: bool
    built_in: bool


class AgentListOut(BaseModel):
    agents: list[AgentOut]
    known_tools: list[str]
    dangerous_tools: list[str]
    sandbox_allowed: bool


class AgentSavedOut(BaseModel):
    saved: str
    granted_tools: list[str]
    note: str


# --- catalog ----------------------------------------------------------------


class ModelOut(BaseModel):
    # MODEL_NOTES is merged in whole, and carries whatever a bench measured.
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    id: str
    provider: str


class ModelListOut(BaseModel):
    models: list[ModelOut]


class VoiceListOut(BaseModel):
    provider: str
    voices: list[dict]


class ProviderOut(BaseModel):
    """Whether the provider's environment variable is present. Never its value."""

    name: str
    verified: bool
    streaming: bool | None = None
    requires: str | None = None
    key_present: bool
    voices: list[dict]


class ProviderListOut(BaseModel):
    providers: dict[str, list[ProviderOut]]


# --- connectors -------------------------------------------------------------


class ComposioConfigIn(BaseModel):
    api_key: str | None = None
    user_id: str | None = None


class ConnectorIn(BaseModel):
    """A connector definition. `core.connectors.normalise` has the final say."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str | None = None
    slug: str | None = None
    toolkit: str | None = None


class ToggleAgentIn(BaseModel):
    agent_id: str


class ConnectorListOut(BaseModel):
    # Catalog entries merged with stored rows, so the key set is open.
    connectors: list[dict]
    composio: dict


class ComposioConfigOut(BaseModel):
    success: bool
    api_key_set: bool
    user_id: str | None = None
    verification: dict
    note: str


class ConnectorSavedOut(BaseModel):
    success: bool
    connector: dict


class ToggleAgentOut(BaseModel):
    success: bool
    agent_id: str
    connector_slug: str
    enabled: bool
    toolkits: list[str]


class ConnectOut(BaseModel):
    success: bool
    id: str
    slug: str
    connected: bool
    auth_url: str | None = None
    message: str


class OAuthUrlOut(BaseModel):
    oauth_url: str
    # Present only when Composio refused and this is the dashboard link.
    fallback: bool | None = None


class DisconnectOut(BaseModel):
    success: bool
    id: str
    connected: bool
    message: str


# --- flows ------------------------------------------------------------------


class FlowIn(BaseModel):
    """A canvas graph.

    Open on purpose, unlike the others: the node tree is the editor's own
    format and gains shapes as node types are added. Pinning it down here would
    mean editing this file every time the canvas learns a node.
    """

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str | None = None
    description: str | None = None
    nodes: list[dict] = []
    edges: list[dict] = []


class FlowListOut(BaseModel):
    flows: list[dict]


class FlowSavedOut(BaseModel):
    success: bool
    flow: dict


class FlowDeletedOut(BaseModel):
    success: bool
    deleted: str


class FlowDeployedOut(BaseModel):
    success: bool
    deployed_agent_id: str
    agent: dict
    message: str


# --- diagnostics ------------------------------------------------------------


class IntegrationOut(BaseModel):
    id: str
    label: str
    kind: str
    connected: bool
    missing: list[str]
    detail: str


class IntegrationListOut(BaseModel):
    integrations: list[IntegrationOut]


class StateOut(BaseModel):
    employees: list[dict]
    tickets: list[dict]
    resets: list[dict]


# --- documents --------------------------------------------------------------


class DocumentQueryIn(BaseModel):
    query: str
    caller_id: str = "default"
    top_k: int = 3


class DocumentQueryOut(BaseModel):
    query: str
    result: str
    caller_id: str
    top_k: int


class UploadOut(BaseModel):
    # rag.index_document's own report is spread in whole.
    model_config = ConfigDict(extra="allow")

    object_key: str | None = None
    stored_chunks: int


class DocumentListOut(BaseModel):
    documents: list[dict]
    pinecone_configured: bool
    storage_configured: bool
    total_chunks: int


class LinkOut(BaseModel):
    url: str


class ChunkDeletedOut(BaseModel):
    deleted: str
    count: int


class ChunksClearedOut(BaseModel):
    cleared: int


# --- service desk -----------------------------------------------------------


class EmployeeOut(BaseModel):
    id: str
    name: str
    email: str
    locked: bool


class TicketOut(BaseModel):
    id: str
    title: str
    status: str
    updated: str


class TicketListOut(BaseModel):
    tickets: list[TicketOut]


class TicketCreatedOut(BaseModel):
    id: str
    status: str


class ResetOut(BaseModel):
    reset_id: str
    status: str
    sent_to: str


# --- session ----------------------------------------------------------------


class TokenOut(BaseModel):
    url: str
    token: str
    room: str
    agent: str


class CallPlacedOut(BaseModel):
    # telephony.place_call's result is spread in whole.
    model_config = ConfigDict(extra="allow")

    agent: str


class TelephonyOut(BaseModel):
    configured: bool
    missing: list[str]
    max_call_seconds: int


# --- telemetry --------------------------------------------------------------


class LogsOut(BaseModel):
    events: list[dict]
    seq: int
    dropped: int


class LogEvent(BaseModel):
    kind: str
    detail: dict = {}


class AnalyticsOut(BaseModel):
    # Each bucket's `stats` is keyed by stage name, which is open by design.
    models: list[dict]
    files: int
    budget_ms: int
    note: str


class TraceRow(BaseModel):
    filename: str
    agent: str
    model: str
    turns: int
    total_seconds: float | None = None
    avg_first_token_ms: float | None = None
    created_at: str


class TraceListOut(BaseModel):
    traces: list[TraceRow]


class TranscriptLine(BaseModel):
    turn: int | None = None
    who: str
    text: str
    intent: str | None = None


class ToolCallOut(BaseModel):
    turn: int | None = None
    tool: str | None = None
    success: bool | None = None
    args: dict | None = None
    result: str | None = None
    recovered: bool | None = None


class ReplyStats(BaseModel):
    n: int
    p50: float | None = None
    slowest: float | None = None
    over_budget: int


class CallDetailOut(BaseModel):
    id: str
    agent: str | None = None
    model: str | None = None
    provider: str | None = None
    started: str | None = None
    transcript: list[TranscriptLine]
    tools: list[ToolCallOut]
    # stage name -> {n, p50, max}
    timings: dict[str, dict]
    reply_ms: ReplyStats
    quality: dict
    event_count: int


class QualityAnalyticsOut(BaseModel):
    # The empty case returns `note` instead of `metric_definitions`.
    model_config = ConfigDict(extra="allow")

    aggregate: dict
    per_model: dict
    per_call: list[dict]
    calls_with_quality_data: int
    total_files: int
