import os

import pytest

from backend.config import NEO4J_VECTOR_INDEX
from backend.document import Document
from backend.graph_retrieve import graph_facts
from backend.rag import LegalRAG
from backend.records import document_from_record
from backend.retrieval import expand_legal_units
from backend.routing import classify_query


pytestmark = pytest.mark.skipif(
    os.getenv("LEGALRAG_INTEGRATION") != "1",
    reason="set LEGALRAG_INTEGRATION=1 to run Neo4j integration checks",
)

SOURCE_PATH = "documents/raw/statutes/_integration_test.hwpx"
STRUCTURE = {
    "chapter_number": "제1장", "chapter_title": "총칙", "chapter_order": 1,
    "section_number": "제1절", "section_title": "목적", "section_order": 1,
}
REQUIRED_CONSTRAINTS = {
    "concept_name", "document_source_path", "chunk_id", "definition_id",
    "legal_unit_id", "term_id",
}


class FakeEmbeddingModel:
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def encode(self, texts, **_kwargs):
        vector = [0.0] * self.dimensions
        return vector if isinstance(texts, str) else [vector for _ in texts]


@pytest.fixture(scope="module")
def rag():
    with LegalRAG() as system:
        yield system


def test_required_constraints_exist(rag):
    records = rag._run_read(
        "SHOW CONSTRAINTS YIELD name RETURN collect(name) AS names"
    )
    assert REQUIRED_CONSTRAINTS <= set(records[0]["names"])


def _chunk(number, text, article, title, order):
    return Document(text, {
        "chunk_number": number, "document_name": "integration-test.hwpx",
        "group_id": number - 1, "part": 0, **STRUCTURE,
        "article_number": article, "article_title": title,
        "article_order": order,
    })


def test_graph_tree_write_retrieve_and_cleanup(rag):
    dimensions = rag._run_read(
        """
        SHOW VECTOR INDEXES YIELD name, options
        WHERE name = $name
        RETURN options.indexConfig.`vector.dimensions` AS dimensions
        """,
        name=NEO4J_VECTOR_INDEX,
    )
    dimensions = dimensions[0]["dimensions"] if dimensions else 16
    chunks = [
        _chunk(1, "제1조(목적) 이 법은 테스트 목적을 규정한다.", "제1조", "목적", 1),
        _chunk(2, "개인정보처리자는 정보주체에게 테스트 사실을 통지하여야 한다.", "제1조", "목적", 1),
        _chunk(3, "제2조(정의) “테스트용어”란 테스트에서 사용하는 용어를 말한다.", "제2조", "정의", 2),
    ]
    rag.delete_document(SOURCE_PATH)
    try:
        with LegalRAG(
            embedding_model=FakeEmbeddingModel(dimensions),
            embedding_dimensions=dimensions,
        ) as writer:
            writer.index(chunks, SOURCE_PATH, source_mtime_ns=1)

        tree = rag._run_read(
            """
            MATCH (:Document {source_path: $path})-[:CONTAINS*1..3]
                ->(article:LegalUnit {kind: "article"})
            MATCH (article)-[:CONTAINS*0..3]->(:LegalUnit)-[:SUPPORTED_BY]->(chunk:Chunk)
            RETURN article.number AS article_number, article.order AS article_order,
                count(chunk) AS chunks
            ORDER BY article_order
            """,
            path=SOURCE_PATH,
        )
        assert [(row["article_number"], row["chunks"]) for row in tree] == [
            ("제1조", 2), ("제2조", 1),
        ]

        seed = rag._run_read(
            """
            MATCH (chunk:Chunk {source_path: $path, chunk_number: 2})
            RETURN chunk.id AS id, chunk.content AS content,
                chunk.heading_path AS heading_path, chunk.chunk_number AS chunk_number,
                chunk.document_name AS document_name, chunk.group_id AS group_id,
                chunk.part AS part, chunk.source_path AS source_path, 1.0 AS score
            """,
            path=SOURCE_PATH,
        )[0]
        document = document_from_record(seed)
        expanded = expand_legal_units(
            rag._run_read, [document], classify_query("제1조"), [],
        )
        assert [item.meta["chunk_number"] for item in expanded] == [2, 1]

        facts = graph_facts(
            rag._run_read, "개인정보처리자의 통지 의무", [document],
        )
        assert any("-MUST_NOTIFY->" in fact.content for fact in facts)
    finally:
        rag.delete_document(SOURCE_PATH)

    leftovers = rag._run_read(
        "MATCH (node {source_path: $path}) RETURN count(node) AS count",
        path=SOURCE_PATH,
    )
    assert leftovers[0]["count"] == 0


@pytest.mark.parametrize(("document", "article"), [
    ("개인정보 보호법(법률)", "제15조"),
    ("개인정보 보호법 시행령", "제30조"),
    ("대한민국헌법", "제10조"),
    ("인공지능 발전과 신뢰 기반 조성 등에 관한 기본법", "제4조"),
])
def test_indexed_article_lookup(rag, document, article):
    rows = rag._run_read(
        """
        MATCH (document:Document)-[:CONTAINS*1..3]->
            (article:LegalUnit {kind: "article", number: $article})
        WHERE document.name CONTAINS $document
        RETURN count(article) AS count
        """,
        document=document,
        article=article,
    )
    assert rows[0]["count"] > 0


@pytest.mark.parametrize(("document", "phrase", "article"), [
    ("개인정보 보호법(법률)", "개인정보의 수집ㆍ이용", "제15조"),
    ("개인정보 보호법(법률)", "자동화된 결정", "제37조의2"),
    ("인공지능 발전과 신뢰 기반 조성 등에 관한 기본법", "인공지능 기본계획", "제6조"),
    ("대한민국헌법", "국민의 모든 자유와 권리", "제37조"),
])
def test_indexed_phrase_lookup(rag, document, phrase, article):
    rows = rag._run_read(
        """
        MATCH (document:Document)-[:CONTAINS*1..3]->
            (article:LegalUnit {kind: "article"})
        MATCH (article)-[:CONTAINS*0..3]->(unit:LegalUnit)
            -[:SUPPORTED_BY]->(chunk:Chunk)
        WHERE document.name CONTAINS $document AND chunk.content CONTAINS $phrase
        RETURN collect(DISTINCT article.number) AS articles
        """,
        document=document,
        phrase=phrase,
    )
    assert article in rows[0]["articles"]
