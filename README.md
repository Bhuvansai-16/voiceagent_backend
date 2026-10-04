# Voice Service Desk — Backend

FastAPI control plane and LiveKit voice worker for a browser- and
phone-callable voice agent, backed by Neon Postgres.

```text
backend/
  core/        domain and shared services. imports no livekit, no fastapi
  worker/      LiveKit runtime: session wiring, agent, plugins, telephony
  store/       Store protocol, PostgresStore, MemoryStore, schema.sql
  analytics/   trace analysis, served by the API and by the CLIs
  api/         FastAPI application and routers
  tests/
scripts/       migrate.py, smoke.py, and the report CLIs
skills/        instruction files a persona can load by name
```

The API and the worker are separate processes that share a database, not a
filesystem. `core/` may not import `livekit`, `fastapi`, `backend.api`,
`backend.worker` or `backend.store` — [test_production_layout.py](backend/tests/test_production_layout.py)
walks its AST and fails if that ever stops being true.

## Setup

Requires Python 3.11+ and a Neon project. No virtual environment: dependencies
install into your Python directly.

```bash
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env      # then fill it in
python scripts/migrate.py
```

`requirements.txt` is generated from `uv.lock`, so versions stay pinned even
without uv. Regenerate after changing `pyproject.toml`:

```bash
uv export --no-hashes --no-emit-project --format requirements-txt > requirements.txt
```

### What has to be in .env

- `NEON_CONNECTION_URI` - the **pooled** string; its hostname contains
  `-pooler`. The direct endpoint is derived from it automatically.
- `LIVEKIT_*`, `DEEPGRAM_API_KEY`, and one LLM provider key.
- `AUTH_URL` for Neon Auth. `JWKS_URL` is optional and defaults to
  `<AUTH_URL>/.well-known/jwks.json`, which is where the keys actually are.
- `AWS_ENDPOINT_URL_S3` and the `AWS_*` keys for object storage, if you want
  uploaded documents kept.

Keep `.env` at the repository root. `load_dotenv()` searches upward from the
working directory, so a copy inside `backend/` is not found.

## Run

```bash
uvicorn main:app --reload --port 8002   # API
python -m backend.worker.main dev       # voice worker, separate terminal
```

**`--reload` is not optional on Windows.** psycopg's async mode cannot run on
`ProactorEventLoop`, and uvicorn selects its loop with a factory rather than a
policy:

```python
if sys.platform == "win32" and not use_subprocess:
    return asyncio.ProactorEventLoop
```

so nothing the app sets at import time changes it. `--reload` runs the app in a
subprocess, which takes the selector branch. Without the reloader, use
`python scripts/serve.py --no-reload`, which drives the server on a loop it
chooses. Linux is unaffected.

`--port 8002` is not optional either. uvicorn's own default is 8000, and both
the frontend's vite proxy and the worker's `DEBUG_API_URL` look for 8002, so
the port is part of the command rather than a preference. `scripts/serve.py`
defaults to 8002 and needs no flag.

The API's OpenAPI document is at <http://localhost:8002/docs>. It allows CORS
from <http://localhost:5174>, where the frontend dev server runs.

The UI lives in a separate repository:
<https://github.com/Bhuvansai-16/voiceagent_frontend>

## Tests

```bash
uv run pytest
```

Roughly 120 tests need no network and no credentials. The rest use the `client`
fixture and skip when `NEON_CONNECTION_URI` is absent. The store contract suite
runs the same cases against `MemoryStore` and against Neon, so the test double
cannot quietly drift from real Postgres semantics.

## Diagnostics

```bash
uv run python scripts/smoke.py             # every provider, key and model, end to end
uv run python scripts/latency_report.py    # per-stage p50/p95 from call traces
uv run python scripts/quality_metrics.py   # the eight LLM quality metrics
```

The database starts empty. Employee, ticket and document records come from the
configured service integrations or through the normal API operations; no seed or
demo data is shipped.
