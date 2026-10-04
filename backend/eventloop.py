"""Windows needs a selector event loop for async Postgres.

psycopg's async mode refuses to run on ProactorEventLoop, which is Python's
default on Windows:

    InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run in
    async mode.

It does not fail fast or clearly. The pool simply never fills, then times out
after 30 seconds with PoolTimeout, and the real reason is buried in a connection
error the pool logs and swallows.

This is a Windows-only concern. On Linux the default loop uses epoll and psycopg
is happy, so nothing here changes production behaviour.

Call this before the first event loop is created — at import in an entry point,
not inside a coroutine. Setting a policy after a loop exists does nothing.

LiveKit's worker isolates jobs with multiprocessing rather than asyncio
subprocesses, so the selector loop's lack of subprocess support on Windows does
not affect it. That is the one thing worth re-checking if the worker starts
behaving oddly on Windows after this.
"""

import asyncio
import sys


def use_selector_loop_on_windows() -> None:
    if sys.platform != "win32":
        return
    policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy is None:  # non-CPython, or a future release that dropped it
        return
    if not isinstance(asyncio.get_event_loop_policy(), policy):
        asyncio.set_event_loop_policy(policy())
