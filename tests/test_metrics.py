from backend.document import Document
from evaluation.metrics import citation_matches, context_precision


def test_context_precision_matches_gold_document_filename_from_metadata():
    chunk = Document(
        "relevant answer text",
        {
            "document_name": "신용카드_이용자_생활법령_QA.pdf",
            "source_path": "documents/raw/qna/신용카드_이용자_생활법령_QA.pdf",
        },
    )

    assert context_precision(
        [chunk],
        ["신용카드_이용자_생활법령_QA.pdf"],
    ) == 1.0


def test_context_precision_rejects_right_article_from_wrong_document():
    chunk = Document(
        "제2조 관련 내용",
        {
            "article_number": "제2조",
            "document_name": "다른법.hwpx",
            "source_path": "documents/raw/statutes/다른법.hwpx",
        },
    )

    assert context_precision(
        [chunk],
        ["제2조"],
        "documents/raw/statutes/대상법.hwpx",
    ) == 0.0


def test_citation_matches_clause_descriptors():
    assert citation_matches("제5조제1항 단서", "제5조제1항")
    assert citation_matches("제3조제2항 각 호", "제3조제2항")
