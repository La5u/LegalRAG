from backend.document import Document
from backend.reasoning import (
    build_symbolic_trace,
    classify_conclusion,
)


def test_build_symbolic_trace_prioritizes_legal_signals():
    evidence = [
        Document(
            "개인정보처리자는 정보주체에게 통지하여야 한다",
            {
                "document_name": "개인정보 보호법.hwpx",
                "chunk_number": 1,
                "article_number": "제15조",
                "graph_relation": "MUST_NOTIFY",
            },
        ),
        Document(
            "다만 예외적인 경우에는 달리 정한다",
            {
                "document_name": "개인정보 보호법.hwpx",
                "chunk_number": 2,
                "article_number": "제15조",
            },
        ),
    ]

    trace = build_symbolic_trace("개인정보처리자의 통지 의무는?", evidence)

    assert trace["query_type"] == "obligation"
    assert trace["important_evidence"]
    assert trace["matched_concepts"]
    assert trace["conclusion"]["answer_type"] in {"yes", "conditional"}


def test_privacy_trace_refuses_conclusion_when_rules_are_missing():
    trace = build_symbolic_trace("개인정보를 국외 이전해도 되나요?", [])
    assert trace["workflow"] == "privacy_compliance"
    assert trace["conclusion"]["answer_type"] == "needs_facts"


def test_classify_conclusion_core_answer_types():
    assert classify_conclusion("질문", "prohibition", [], [Document("금지", {})], [], [], [], [])[
        "answer_type"
    ] == "no"
    assert classify_conclusion("질문", "obligation", [], [], [], [Document("하여야 한다", {})], [], [])[
        "answer_type"
    ] == "yes"
    assert classify_conclusion("질문", "general", [], [], [], [], [Document("다만 예외", {})], [])[
        "answer_type"
    ] == "conditional"
    assert classify_conclusion("질문", "general", [], [], [], [], [], [])[
        "answer_type"
    ] == "not_answered"
