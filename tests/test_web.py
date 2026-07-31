import asyncio
import json

import pytest
import httpx
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request

from backend.document import Document
from backend.retrieval import RetrievalContext, RetrievalResult
from backend.routing import QueryRoute
from backend.slurm_vllm import SlurmGpuBusy, SlurmStartupTimeout, StatusUpdate
from backend.web import (
    Question,
    ask,
    generation_error_code,
    raw_document,
    require_proxy_secret,
)


PDF_PATH = "contracts/상가건물_임대차_표준계약서.pdf"


def test_raw_document_opens_pdf_inline():
    response = raw_document(PDF_PATH)

    assert response.headers["content-disposition"].startswith("inline;")


def test_raw_document_downloads_pdf_as_attachment():
    response = raw_document(PDF_PATH, download=True)

    assert response.headers["content-disposition"].startswith("attachment;")


def test_question_rejects_unknown_document_category():
    with pytest.raises(ValidationError):
        Question(question="질문", category="unknown")


def test_question_rejects_unknown_generation_model():
    with pytest.raises(ValidationError):
        Question(question="질문", generation_model="paid-or-unknown")


def test_proxy_secret_is_required_only_when_configured(monkeypatch):
    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
    })
    monkeypatch.setattr("backend.web.PROXY_SECRET", "")
    require_proxy_secret(request)

    monkeypatch.setattr("backend.web.PROXY_SECRET", "expected")
    with pytest.raises(HTTPException) as error:
        require_proxy_secret(request)
    assert error.value.status_code == 403

    authorized = Request({
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"x-internal-proxy-secret", b"expected")],
    })
    require_proxy_secret(authorized)


def test_hosted_model_requires_key_before_retrieval():
    class FakeRAG:
        def retrieve(self, *_args, **_kwargs):
            raise AssertionError("retrieval must not run without an API key")

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            ask(
                Question(
                    question="질문",
                    generation_model="gemini-3.1-flash-lite",
                ),
                FakeRAG(),
            )
        )

    assert error.value.status_code == 422


@pytest.mark.parametrize([
    "status", "expected",
], [
    (401, "invalid_api_key"),
    (403, "invalid_api_key"),
    (429, "rate_limit"),
    (400, "provider_rejected"),
    (500, "generation_failed"),
])
def test_generation_error_code_for_provider_status(status, expected):
    request = httpx.Request("POST", "https://provider.test/chat")
    response = httpx.Response(status, request=request)
    error = httpx.HTTPStatusError(
        "provider error",
        request=request,
        response=response,
    )

    assert generation_error_code(error) == expected


def test_generation_error_code_for_slurm_gpu_busy():
    assert generation_error_code(SlurmGpuBusy("busy")) == "gpu_busy"


def test_generation_error_code_for_slurm_startup_timeout():
    assert generation_error_code(SlurmStartupTimeout("timed out")) == "provider_unavailable"


def test_ask_stream_uses_explicit_retrieval_result(monkeypatch):
    result = RetrievalResult(
        strategy="graph",
        context=RetrievalContext(
            query="질문",
            route=QueryRoute(kind="general"),
            source_paths=[],
            is_summary_query=False,
            is_contract_query=False,
            is_case_query=False,
        ),
        evidence=[Document("근거", {"document_name": "법.hwpx"})],
    )

    class FakeRAG:
        def retrieve(self, *_args, **_kwargs):
            return result

        async def generate_async(self, _query, received_result, callback, **_options):
            assert received_result is result
            await callback("답변")

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("backend.web.asyncio.to_thread", run_inline)

    async def collect_events():
        response = await ask(Question(question="질문"), FakeRAG())
        return [
            json.loads(line)
            async for line in response.body_iterator
        ]

    events = asyncio.run(collect_events())

    assert [event["type"] for event in events] == ["sources", "token", "done"]


def test_ask_stream_relays_slurm_status_updates(monkeypatch):
    result = RetrievalResult(
        strategy="graph",
        context=RetrievalContext(
            query="질문",
            route=QueryRoute(kind="general"),
            source_paths=[],
            is_summary_query=False,
            is_contract_query=False,
            is_case_query=False,
        ),
        evidence=[Document("근거", {"document_name": "법.hwpx"})],
    )

    class FakeRAG:
        def retrieve(self, *_args, **_kwargs):
            return result

        async def generate_async(self, _query, _result, callback, on_status=None, **_options):
            await on_status(StatusUpdate(phase="starting", elapsed=0.4, estimated_total=75))
            await callback("답변")

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("backend.web.asyncio.to_thread", run_inline)

    async def collect_events():
        response = await ask(Question(question="질문"), FakeRAG())
        return [json.loads(line) async for line in response.body_iterator]

    events = asyncio.run(collect_events())

    assert [event["type"] for event in events] == ["sources", "status", "token", "done"]
    status_event = events[1]
    assert status_event["phase"] == "starting"
    assert status_event["estimated_total"] == 75


def test_ask_passes_hosted_model_and_key_to_generation(monkeypatch):
    result = RetrievalResult(
        strategy="graph",
        context=RetrievalContext(
            query="질문",
            route=QueryRoute(kind="general"),
            source_paths=[],
            is_summary_query=False,
            is_contract_query=False,
            is_case_query=False,
        ),
        evidence=[Document("근거", {"document_name": "법.hwpx"})],
    )
    captured = {}

    class FakeRAG:
        def retrieve(self, *_args, **_kwargs):
            return result

        async def generate_async(self, _query, _result, callback, **options):
            captured.update(options)
            await callback("답변")

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("backend.web.asyncio.to_thread", run_inline)

    async def collect_events():
        response = await ask(
            Question(
                question="질문",
                generation_model="qwen/qwen3.6-27b",
                api_key="groq-secret",
            ),
            FakeRAG(),
        )
        return [json.loads(line) async for line in response.body_iterator]

    asyncio.run(collect_events())

    assert callable(captured.pop("on_status"))
    assert captured == {
        "hosted_model": "qwen/qwen3.6-27b",
        "api_key": "groq-secret",
    }
