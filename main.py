"""Entry point, so the API starts with one ordinary command:

    uvicorn main:app --reload

`--reload` is not optional on Windows, and the reason is worth knowing.

psycopg's async mode refuses to run on ProactorEventLoop, which is Python's
default there. uvicorn does not pick its loop from the event loop policy - it
picks it with a factory:

    if sys.platform == "win32" and not use_subprocess:
        return asyncio.ProactorEventLoop

so nothing this module can set at import time changes it. But `--reload` runs
the app in a subprocess, which takes the other branch and gets a
SelectorEventLoop. So the reload command works and the plain one does not:

    uvicorn main:app --reload      works everywhere
    uvicorn main:app               fails on Windows with, after 30 seconds,
                                   psycopg_pool.PoolTimeout: pool
                                   initialization incomplete

For production, or any time you do not want the reloader, use:

    python scripts/serve.py --no-reload

which drives the server on a loop it chooses. On Linux none of this applies and
every form works.

The voice worker is a separate process, deliberately - it joins LiveKit rooms
and is scaled independently of the API:

    python -m backend.worker.main dev
"""

from backend.api.main import app

__all__ = ["app"]
