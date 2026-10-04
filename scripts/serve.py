"""Run the API.

    uv run python scripts/serve.py              # reload on, port 8002
    uv run python scripts/serve.py --port 9000
    uv run python scripts/serve.py --no-reload

Exists because of a Windows-only ordering problem. psycopg's async mode cannot
run on ProactorEventLoop, which is Python's default on Windows, and
`backend.eventloop` sets a selector policy to fix that. But running
`uvicorn backend.api.main:app` from the command line creates the event loop
*before* it imports the app, so the policy is set too late to matter: the
connection pool never fills and startup dies with

    psycopg_pool.PoolTimeout: pool initialization incomplete after 30 sec

which names the pool and says nothing about event loops.

Setting the policy here, before uvicorn is even imported, is early enough.
On Linux this changes nothing - the default loop already works - so the plain
uvicorn command remains fine in a container.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / "backend" / ".env")

sys.path.insert(0, str(ROOT))

from backend.eventloop import use_selector_loop_on_windows  # noqa: E402

use_selector_loop_on_windows()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the voice service desk API.")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-reload", action="store_true",
                        help="Disable autoreload (use this in production).")
    args = parser.parse_args(argv)

    import asyncio

    import uvicorn

    from backend.store.pool import configured

    if not configured():
        print(
            "NEON_CONNECTION_URI is not set, so every database-backed endpoint "
            "will fail.\nCopy the pooled connection string from the Neon console "
            "into .env, then run:\n  uv run python scripts/migrate.py",
            file=sys.stderr,
        )

    config = uvicorn.Config(
        "backend.api.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        reload_dirs=[str(ROOT / "backend")],
    )
    server = uvicorn.Server(config)

    if args.no_reload and sys.platform == "win32":
        # uvicorn picks the loop with a factory, not a policy:
        #
        #   if sys.platform == "win32" and not use_subprocess:
        #       return asyncio.ProactorEventLoop
        #
        # so no policy set here can win, and psycopg cannot run on Proactor.
        # Drive the server on a loop we choose instead.
        #
        # The reload path already gets SelectorEventLoop for free, because the
        # reloader spawns a subprocess and that branch returns the selector
        # loop - which is why `--reload` works on Windows and plain
        # `uvicorn backend.api.main:app` does not.
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(server.serve())
        return 0

    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
