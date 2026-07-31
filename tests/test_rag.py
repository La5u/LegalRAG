import asyncio

from backend.document import Document
from backend.rag import LegalRAG, format_evidence_context, grounded_symbolic_context
from backend.retrieval import RetrievalContext, RetrievalResult
from backend.routing import QueryRoute
from tests.conftest import make_graph_fact


def retrieval_result(evidence, trace=None, graph_facts=None):
    return RetrievalResult(
        strategy="neuro_symbolic" if trace else "graph",
        context=RetrievalContext(
            query="질문",
            route=QueryRoute(kind="general"),
            source_paths=[],
            is_summary_query=False,
            is_contract_query=False,
            is_case_query=False,
        ),
        evidence=evidence,
        graph_facts=graph_facts or [],
        symbolic_trace=trace,
    )


def test_grounded_symbolic_context_preserves_grounded_rule_role():
    chunks = [
        Document(
            "처리자는 통지하여야 한다.",
            {"document_name": "law.hwpx", "chunk_number": 7},
        )
    ]
    trace = {
        "query_type": "obligation",
        "obligations_checked": [{"document": "law.hwpx", "chunk": 7}],
    }

    context = grounded_symbolic_context(chunks, trace)

    assert context is not None
    assert "[1] 의무" in context
    assert "결론이 아님" in context


def test_grounded_symbolic_context_labels_exception_without_deciding_outcome():
    chunks = [
        Document(
            "다만 특별한 경우에는 제외한다.",
            {"document_name": "law.hwpx", "chunk_number": 3},
        )
    ]
    trace = {
        "query_type": "exception",
        "exceptions_checked": [{"document": "law.hwpx", "chunk": 3}],
    }

    context = grounded_symbolic_context(chunks, trace)

    assert context is not None
    assert "[1] 예외" in context


def test_empty_evidence_returns_without_calling_model(monkeypatch):
    async def fail_stream(*_args, **_kwargs):
        raise AssertionError("the model must not run without evidence")

    monkeypatch.setattr("backend.rag.stream_answer", fail_stream)
    tokens = []
    rag = object.__new__(LegalRAG)

    async def collect(token):
        tokens.append(token)

    asyncio.run(rag.generate_async("질문", retrieval_result([]), collect))

    assert tokens == ["검색된 문서에서 질문을 뒷받침할 근거를 찾지 못했습니다."]


def test_generation_uses_compact_document_metadata(monkeypatch):
    captured = {}

    async def capture_stream(query, context, callback, symbolic_context=None, on_status=None):
        captured["context"] = context

    monkeypatch.setattr("backend.rag.stream_answer", capture_stream)
    chunk = Document(
        "법률 내용",
        {
            "document_title": "개인정보 보호법",
            "version_label": "법률 제20897호",
            "specific_citation": "제15조제1항제4호",
            "page_number": 12,
        },
    )
    rag = object.__new__(LegalRAG)

    async def ignore(_token):
        pass

    asyncio.run(rag.generate_async("질문", retrieval_result([chunk]), ignore))

    assert "개인정보 보호법, 법률 제20897호, 제15조제1항제4호" in captured["context"]
    assert "12" not in captured["context"]


def test_format_evidence_context_is_pure_and_compact():
    chunk = Document(
        "법률 내용",
        {
            "document_title": "개인정보 보호법",
            "version_label": "법률 제20897호",
            "article_number": "제15조",
        },
    )
    result = retrieval_result([chunk])

    assert format_evidence_context(result) == (
        "[1] 개인정보 보호법, 법률 제20897호, 제15조\n법률 내용"
    )


def test_graph_fact_is_formatted_beneath_its_supporting_source():
    chunk = Document(
        "처리자는 통지하여야 한다.",
        {"document_name": "법.hwpx", "chunk_number": 7},
    )
    fact = make_graph_fact(
        id="fact-1",
        document="법.hwpx",
        source_path="documents/raw/statutes/법.hwpx",
        chunk=7,
        path="처리자 -MUST_NOTIFY-> 정보주체",
        sentence="처리자는 통지하여야 한다.",
        score=0.9,
    )

    context = format_evidence_context(retrieval_result([chunk], graph_facts=[fact]))

    assert "구조화된 관계: 처리자 -MUST_NOTIFY-> 정보주체" in context
    assert context.count("[1]") == 1
