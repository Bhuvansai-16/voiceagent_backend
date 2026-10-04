-- Voice service desk schema.
--
-- Apply over the DIRECT (non-pooler) Neon endpoint. Neon documents that schema
-- migrations require it; the pooled endpoint runs PgBouncer in transaction mode
-- and DDL through it is not reliable.
--
-- Every statement is IF NOT EXISTS so this file is idempotent and doubles as the
-- migration. Three tables is not enough to justify Alembic.

-- ---------------------------------------------------------------------------
-- Configuration state, formerly config/*.json
--
-- JSONB rather than a column per field, deliberately. CONFIGURABLE in
-- core/personas.py is a whitelist that grows: it recently gained stt_provider,
-- stt_model and tts_provider, and the lag in that growth was the bug where
-- months of speech settings were silently dropped at load. Column-per-field
-- would need a migration every time it grows; JSONB keeps the whitelist in
-- _apply, where it already lives and is already tested.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS agents (
    id      TEXT PRIMARY KEY,
    fields  JSONB       NOT NULL DEFAULT '{}'::jsonb,
    updated TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS flows (
    id                TEXT PRIMARY KEY,
    graph             JSONB       NOT NULL DEFAULT '{}'::jsonb,
    deployed_agent_id TEXT,
    updated           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS connectors (
    id         TEXT PRIMARY KEY,
    definition JSONB       NOT NULL DEFAULT '{}'::jsonb,
    connected  BOOLEAN     NOT NULL DEFAULT false,
    updated    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Service desk domain, ported from data/servicedesk.db
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS employees (
    id     TEXT PRIMARY KEY,
    name   TEXT    NOT NULL,
    email  TEXT    NOT NULL,
    locked BOOLEAN NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS tickets (
    id          TEXT PRIMARY KEY,
    employee_id TEXT        NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    title       TEXT        NOT NULL,
    description TEXT        NOT NULL DEFAULT '',
    status      TEXT        NOT NULL DEFAULT 'open',
    updated     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS tickets_employee_idx ON tickets (employee_id, status);

-- idempotency_key is the primary key, not a unique index on a surrogate. It is
-- what makes a replayed password reset return the original result instead of
-- triggering a second one, which is what keeps a barge-in mid-reset safe.
CREATE TABLE IF NOT EXISTS password_resets (
    idempotency_key TEXT PRIMARY KEY,
    reset_id        TEXT        NOT NULL,
    employee_id     TEXT        NOT NULL,
    status          TEXT        NOT NULL,
    sent_to         TEXT        NOT NULL,
    created         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Call traces, formerly calls/*.jsonl
--
-- The API used to glob TRACE_DIR and JSON-parse every file on every request,
-- while the worker wrote those files to its own disk. Two processes, one
-- assumed filesystem.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS calls (
    id       TEXT PRIMARY KEY,          -- livekit job id
    agent    TEXT,
    model    TEXT,
    provider TEXT,
    voice    TEXT,
    room     TEXT,
    started  TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS call_events (
    seq     BIGSERIAL PRIMARY KEY,
    call_id TEXT        NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    stage   TEXT        NOT NULL,
    ms      DOUBLE PRECISION,
    at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload JSONB       NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS call_events_call_idx  ON call_events (call_id);
CREATE INDEX IF NOT EXISTS call_events_stage_idx ON call_events (stage);

-- ---------------------------------------------------------------------------
-- RAG chunks, formerly rag._LOCAL_DOC_STORE
--
-- That was a module-level Python list. The API appended to it on upload and the
-- worker read its own separate copy during search_documents, so uploading a
-- document and then asking the agent about it found nothing unless Pinecone was
-- configured. A table is what makes the two processes agree.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS doc_chunks (
    id          TEXT PRIMARY KEY,
    filename    TEXT        NOT NULL,
    caller_id   TEXT        NOT NULL,
    chunk_index INT         NOT NULL,
    text        TEXT        NOT NULL,
    vector      JSONB,
    object_key  TEXT,                   -- key of the original in object storage
    created     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS doc_chunks_caller_idx ON doc_chunks (caller_id);
