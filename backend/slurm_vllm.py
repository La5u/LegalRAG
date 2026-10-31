"""On-demand Slurm-managed vLLM for the "Local" answer model.

This is for a shared GPU server with a one-job-at-a-time policy. Before committing to local inference, check the Slurm queue: only submit
the on-demand vLLM job when the queue is otherwise empty, so the app never
competes with another user's job for the one available GPU. If the queue is
busy, callers should fail gracefully and let the visitor fall back to a hosted
model with their own API key instead.
"""
import asyncio
import logging
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from backend.config import (
    SLURM_COLD_START_ESTIMATE,
    SLURM_JOB_NAME,
    SLURM_STARTUP_TIMEOUT,
    SLURM_SUBMIT_DIR,
    VLLM_URL,
)

logger = logging.getLogger(__name__)


class SlurmGpuBusy(Exception):
    """Another Slurm job already holds the shared GPU."""


class SlurmStartupTimeout(Exception):
    """The on-demand vLLM job did not become ready in time."""


@dataclass
class StatusUpdate:
    """A progress update for a caller to relay to the visitor.

    phase is one of "checking", "starting", "waiting". estimated_total is a
    rough, informational estimate in seconds (see SLURM_COLD_START_ESTIMATE),
    not a hard limit.
    """

    phase: str
    elapsed: float
    estimated_total: float | None = None


async def _run(*args: str, cwd: str | None = None) -> str:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.warning("%s failed: %s", args[0], stderr.decode().strip())
    return stdout.decode()


async def _queue_rows() -> list[tuple[str, str]]:
    output = await _run("squeue", "-h", "-o", "%j %T")
    rows = []
    for line in output.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            rows.append((parts[0], parts[1]))
    return rows


async def _vllm_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(f"{VLLM_URL.rstrip('/')}/v1/models")
            return response.status_code == 200
    except httpx.HTTPError:
        return False


async def ensure_local_vllm(on_status=None) -> None:
    """Make sure an on-demand vLLM server is reachable at VLLM_URL.

    Raises SlurmGpuBusy if another job already holds the GPU, or
    SlurmStartupTimeout if our own job didn't come up in time. If given,
    on_status(StatusUpdate) is awaited whenever there's a cold start to
    report progress on, so a caller can relay it to the visitor.
    """
    if await _vllm_ready():
        return

    loop = asyncio.get_event_loop()
    start = loop.time()

    async def emit(phase: str) -> None:
        if on_status:
            await on_status(
                StatusUpdate(
                    phase=phase,
                    elapsed=loop.time() - start,
                    estimated_total=SLURM_COLD_START_ESTIMATE,
                )
            )

    await emit("checking")
    rows = await _queue_rows()
    ours = [state for name, state in rows if name == SLURM_JOB_NAME]
    others = [(name, state) for name, state in rows if name != SLURM_JOB_NAME]

    if others and not ours:
        name, state = others[0]
        raise SlurmGpuBusy(f"Shared GPU is busy with another job: {name} ({state})")

    if not ours:
        await emit("starting")
        port = str(urlsplit(VLLM_URL).port or 18080)
        await _run("./submit_if_needed.sh", port, cwd=SLURM_SUBMIT_DIR)

    deadline = start + SLURM_STARTUP_TIMEOUT
    while loop.time() < deadline:
        if await _vllm_ready():
            return
        await emit("waiting")
        await asyncio.sleep(3)
    raise SlurmStartupTimeout("Local vLLM did not become ready in time.")
