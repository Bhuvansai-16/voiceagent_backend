"""A terminal the agent can use, in a Daytona sandbox.

Measured on this account. The published figure is "under 90ms"; it is not, and
the async client is far slower than the sync one:

    sync   create 897ms   first exec 915ms   warm exec  261ms   delete 248ms
    async  create + first command 10.9s      warm exec 3275ms

The async path is what an agent uses, so 3-11s is the real cost of a command.
That is firmly deep-lane work: no amount of tuning makes it fit a turn. The
sandbox is still created once per session and reused, because 3.3s beats 10.9s.

The gap between the two clients is unexplained and worth chasing before this is
used in anger. `network_block_all` and `ephemeral` are both set on create and
may account for some of it.

Egress is blocked. On account tiers 1 and 2 Daytona enforces that itself and it
cannot be overridden, but that protection disappears on upgrade to tier 3, so
`network_block_all` is set explicitly rather than inherited. A sandbox that
quietly gains internet access after a billing change is exactly the kind of
thing nobody notices until it matters.

Note that the container reports the host's resources: `nproc` says 64 and `free`
says 755 GiB against an actual 1 vCPU and 1 GiB. Generated code that sizes
thread pools from those numbers will ask for far more than it can have.
"""

import asyncio
import logging
import os

logger = logging.getLogger("service-desk.sandbox")

EXEC_TIMEOUT_S = float(os.getenv("SANDBOX_EXEC_TIMEOUT", "30"))
SANDBOX_NAME = os.getenv("DAYTONA_SANDBOX_NAME", "voice-agent-sandbox")


def configured() -> bool:
    return bool(os.getenv("DAYTONA_API_KEY", "").strip())


class Sandbox:
    """One sandbox for one session. Created on first use, deleted on close."""

    def __init__(self) -> None:
        self._sandbox = None
        self._daytona = None
        self._lock = asyncio.Lock()

    async def _ensure(self):
        if self._sandbox is not None:
            return self._sandbox
        async with self._lock:
            if self._sandbox is not None:      # another turn won the race
                return self._sandbox
            from daytona import AsyncDaytona, CreateSandboxFromSnapshotParams

            self._daytona = AsyncDaytona()
            self._sandbox = await self._daytona.create(
                CreateSandboxFromSnapshotParams(
                    name=SANDBOX_NAME,
                    # Explicit, not inherited from the account tier. See above.
                    network_block_all=True,
                    # Nothing here should outlive the call that created it.
                    ephemeral=True,
                    auto_stop_interval=15,
                )
            )
            logger.info("sandbox %s ready", getattr(self._sandbox, "id", "?"))
            return self._sandbox

    async def run(self, command: str) -> str:
        """Run a shell command and return its output, or a readable failure."""
        try:
            sandbox = await self._ensure()
            async with asyncio.timeout(EXEC_TIMEOUT_S):
                response = await sandbox.process.exec(command)
        except TimeoutError:
            return f"The command was still running after {EXEC_TIMEOUT_S:.0f} seconds, so it was stopped."
        except Exception as exc:
            logger.warning("sandbox command failed: %s", exc)
            return f"The sandbox could not run that: {exc}"

        output = (getattr(response, "result", "") or "").strip()
        if not output:
            return "The command finished and printed nothing."
        # This gets read aloud. A page of output is not a spoken answer.
        return output if len(output) <= 800 else output[:800] + " ... (output truncated)"

    async def close(self) -> None:
        """Delete the sandbox and close the client.

        Both halves matter. Deleting the sandbox stops the per-second billing;
        closing the client releases the aiohttp session, which otherwise leaks a
        connector per call and complains on shutdown.
        """
        if self._sandbox is not None:
            try:
                await self._sandbox.delete()
            except Exception as exc:
                logger.warning("could not delete sandbox: %s", exc)
            finally:
                self._sandbox = None

        if self._daytona is not None:
            try:
                await self._daytona.close()
            except Exception as exc:
                logger.warning("could not close daytona client: %s", exc)
            finally:
                self._daytona = None
