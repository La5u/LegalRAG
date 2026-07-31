import asyncio

import pytest

from backend import slurm_vllm


def test_already_ready_skips_queue_check_and_submit(monkeypatch):
    calls = []

    async def fake_vllm_ready():
        return True

    async def fake_queue_rows():
        calls.append("queue")
        return []

    async def fake_run(*args, cwd=None):
        calls.append(("run", args, cwd))
        return ""

    monkeypatch.setattr(slurm_vllm, "_vllm_ready", fake_vllm_ready)
    monkeypatch.setattr(slurm_vllm, "_queue_rows", fake_queue_rows)
    monkeypatch.setattr(slurm_vllm, "_run", fake_run)

    asyncio.run(slurm_vllm.ensure_local_vllm())

    assert calls == []


def test_raises_gpu_busy_when_another_job_holds_the_queue(monkeypatch):
    async def fake_vllm_ready():
        return False

    async def fake_queue_rows():
        return [("someone_elses_job", "RUNNING")]

    submitted = []

    async def fake_run(*args, cwd=None):
        submitted.append(args)
        return ""

    monkeypatch.setattr(slurm_vllm, "_vllm_ready", fake_vllm_ready)
    monkeypatch.setattr(slurm_vllm, "_queue_rows", fake_queue_rows)
    monkeypatch.setattr(slurm_vllm, "_run", fake_run)

    with pytest.raises(slurm_vllm.SlurmGpuBusy):
        asyncio.run(slurm_vllm.ensure_local_vllm())

    assert submitted == []


def test_submits_job_when_queue_is_empty_then_waits_for_ready(monkeypatch):
    ready_calls = []

    async def fake_vllm_ready():
        ready_calls.append(1)
        return len(ready_calls) > 1

    async def fake_queue_rows():
        return []

    submitted = []

    async def fake_run(*args, cwd=None):
        submitted.append((args, cwd))
        return ""

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(slurm_vllm, "_vllm_ready", fake_vllm_ready)
    monkeypatch.setattr(slurm_vllm, "_queue_rows", fake_queue_rows)
    monkeypatch.setattr(slurm_vllm, "_run", fake_run)
    monkeypatch.setattr(slurm_vllm.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(slurm_vllm, "VLLM_URL", "http://127.0.0.1:18080")
    monkeypatch.setattr(slurm_vllm, "SLURM_SUBMIT_DIR", "/home/user/legalrag-infer")

    asyncio.run(slurm_vllm.ensure_local_vllm())

    assert submitted == [
        (("./submit_if_needed.sh", "18080"), "/home/user/legalrag-infer")
    ]
    assert len(ready_calls) == 2


def test_reuses_our_own_already_running_job_without_resubmitting(monkeypatch):
    async def fake_vllm_ready():
        return False

    async def fake_queue_rows():
        return [(slurm_vllm.SLURM_JOB_NAME, "RUNNING")]

    submitted = []

    async def fake_run(*args, cwd=None):
        submitted.append(args)
        return ""

    async def fake_sleep(_seconds):
        raise slurm_vllm.SlurmStartupTimeout("stop test loop")

    monkeypatch.setattr(slurm_vllm, "_vllm_ready", fake_vllm_ready)
    monkeypatch.setattr(slurm_vllm, "_queue_rows", fake_queue_rows)
    monkeypatch.setattr(slurm_vllm, "_run", fake_run)
    monkeypatch.setattr(slurm_vllm.asyncio, "sleep", fake_sleep)

    with pytest.raises(slurm_vllm.SlurmStartupTimeout):
        asyncio.run(slurm_vllm.ensure_local_vllm())

    assert submitted == []


def test_emits_status_updates_while_submitting_and_waiting(monkeypatch):
    ready_calls = []

    async def fake_vllm_ready():
        ready_calls.append(1)
        return len(ready_calls) > 2

    async def fake_queue_rows():
        return []

    async def fake_run(*args, cwd=None):
        return ""

    async def fake_sleep(_seconds):
        return None

    statuses = []

    async def on_status(status):
        statuses.append(status)

    monkeypatch.setattr(slurm_vllm, "_vllm_ready", fake_vllm_ready)
    monkeypatch.setattr(slurm_vllm, "_queue_rows", fake_queue_rows)
    monkeypatch.setattr(slurm_vllm, "_run", fake_run)
    monkeypatch.setattr(slurm_vllm.asyncio, "sleep", fake_sleep)

    asyncio.run(slurm_vllm.ensure_local_vllm(on_status=on_status))

    phases = [status.phase for status in statuses]
    assert phases[0] == "checking"
    assert phases[1] == "starting"
    assert phases[2:] == ["waiting"] * (len(phases) - 2)
    assert all(status.estimated_total == slurm_vllm.SLURM_COLD_START_ESTIMATE for status in statuses)


def test_no_status_updates_when_already_ready(monkeypatch):
    async def fake_vllm_ready():
        return True

    calls = []

    async def on_status(status):
        calls.append(status)

    monkeypatch.setattr(slurm_vllm, "_vllm_ready", fake_vllm_ready)

    asyncio.run(slurm_vllm.ensure_local_vllm(on_status=on_status))

    assert calls == []


def test_raises_startup_timeout_when_never_ready(monkeypatch):
    async def fake_vllm_ready():
        return False

    async def fake_queue_rows():
        return []

    async def fake_run(*args, cwd=None):
        return ""

    monkeypatch.setattr(slurm_vllm, "_vllm_ready", fake_vllm_ready)
    monkeypatch.setattr(slurm_vllm, "_queue_rows", fake_queue_rows)
    monkeypatch.setattr(slurm_vllm, "_run", fake_run)
    monkeypatch.setattr(slurm_vllm, "SLURM_STARTUP_TIMEOUT", 0)

    with pytest.raises(slurm_vllm.SlurmStartupTimeout):
        asyncio.run(slurm_vllm.ensure_local_vllm())
