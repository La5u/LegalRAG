import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so `backend.*` imports resolve.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from backend.document import Document
from backend.graph_retrieve import GraphFact


def make_graph_fact(**overrides):
    defaults = {
        "id": "gf-1",
        "document": "법.pdf",
        "source_path": "documents/raw/statutes/법.pdf",
        "chunk": "graph-1",
        "relation": "MUST_NOTIFY",
        "path": "개인정보처리자 -MUST_NOTIFY-> 정보주체",
        "sentence": "통지하여야 한다",
        "score": 0.5,
    }
    return GraphFact(**(defaults | overrides))


DEFAULT_META = {
    "chunk_number": 1,
    "document_name": "test_statute.hwpx",
    "source_path": "documents/raw/statutes/test_statute.hwpx",
    "group_id": 0,
    "part": 0,
    "chapter_number": "제1장",
    "chapter_title": "총칙",
    "chapter_order": 1,
    "section_number": "제1절",
    "section_title": "목적",
    "section_order": 1,
    "article_number": "제1조",
    "article_title": "목적",
    "article_order": 1,
}


def make_document(content: str, **overrides) -> Document:
    meta = {**DEFAULT_META, **overrides}
    return Document(content=content, meta=meta)


@pytest.fixture
def make_doc():
    return make_document


@pytest.fixture
def statute_chunks():
    return [
        make_document(
            "제1조(목적) 이 법은 개인정보의 처리 및 보호에 관한 기준을 정하는 것을 목적으로 한다.",
            chunk_number=1,
            article_number="제1조",
            article_title="목적",
            article_order=1,
        ),
        make_document(
            '"개인정보"란 성명, 주민등록번호 및 영상 등 개인을 식별할 수 있는 정보를 말한다.',
            chunk_number=2,
            article_number="제2조",
            article_title="정의",
            article_order=2,
        ),
        make_document(
            "개인정보처리자는 정보주체에게 개인정보의 처리 목적을 통지하여야 한다.",
            chunk_number=3,
            article_number="제15조",
            article_title="개인정보의 수집ㆍ이용",
            article_order=3,
        ),
    ]


@pytest.fixture
def regulation_chunks():
    return [
        make_document(
            "제30조(비밀준수 의무) 개인정보처리자는 법령에 따라 개인정보를 처리함에 있어서 비밀을 유지하여야 한다.",
            chunk_number=1,
            document_name="test_enforcement_decree.hwpx",
            source_path="documents/raw/regulations/test_enforcement_decree.hwpx",
            article_number="제30조",
            article_title="비밀준수 의무",
            article_order=1,
        ),
    ]
