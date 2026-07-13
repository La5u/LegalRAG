import asyncio
import json

import pytest
from pydantic import ValidationError

from backend.document import Document
from backend.retrieval import RetrievalContext, RetrievalResult
from backend.routing import QueryRoute
from backend.web import Question, ask, raw_document


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

        async def generate_async(self, _query, received_result, callback):
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
