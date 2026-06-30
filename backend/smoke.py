import argparse
from pathlib import Path
import py_compile
from time import perf_counter

from backend.rag import LegalRAG
from backend.ingest import Document
from backend.config import NEO4J_VECTOR_INDEX
from backend.records import document_from_record

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMP_SOURCE_PATH = "documents/_smoke.hwpx"
REQUIRED_CONSTRAINTS = {
    "concept_name",
    "document_source_path",
    "chunk_id",
    "definition_id",
    "legal_unit_id",
    "term_id",
}
SMOKE_STRUCTURE = {
    "chapter_number": "제1장",
    "chapter_title": "총칙",
    "chapter_order": 1,
    "section_number": "제1절",
    "section_title": "목적",
    "section_order": 1,
}

# Korean legal terms used in these checks:
# 법령 = law/document, 장 = chapter, 절 = section, 조문 = article,
# 항 = clause, 호 = item, 목 = sub-item, 문서조각 = chunk.
# The current graph is a deterministic tree:
# Document -> LegalUnit -> LegalUnit -> Chunk.


class FakeEmbeddingModel:
    """Small stand-in for the real embedding model during graph smoke tests."""

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def encode(self, texts, **_kwargs):
        vector = [0.0] * self.dimensions
        if isinstance(texts, str):
            return vector
        return [vector for _text in texts]


def pass_message(message: str) -> None:
    print(f"PASS {message}")


def fail(message: str) -> None:
    raise SystemExit(f"FAIL {message}")


def compile_backend() -> None:
    for path in sorted((PROJECT_ROOT / "backend").glob("*.py")):
        py_compile.compile(path, doraise=True)
    pass_message("backend modules compile")


def assert_constraints(rag: LegalRAG) -> None:
    records = rag._run_read(
        """
        SHOW CONSTRAINTS
        YIELD name
        WHERE name IN $required_constraints
        RETURN collect(name) AS names
        """,
        required_constraints=list(REQUIRED_CONSTRAINTS),
    )
    names = set(records[0]["names"]) if records else set()
    missing = REQUIRED_CONSTRAINTS - names
    if missing:
        fail(f"missing Neo4j constraints: {sorted(missing)}")
    pass_message("Neo4j constraints exist")


def vector_dimensions(rag: LegalRAG, fallback_dimensions: int) -> int:
    records = rag._run_read(
        """
        SHOW VECTOR INDEXES
        YIELD name, options
        WHERE name = $index_name
        RETURN options.indexConfig.`vector.dimensions` AS dimensions
        """,
        index_name=NEO4J_VECTOR_INDEX,
    )
    return records[0]["dimensions"] if records else fallback_dimensions


def smoke_chunk(
    chunk_number: int,
    text: str,
    article_number: str,
    article_title: str,
    article_order: int,
) -> Document:
    return Document(
        text,
        {
            "chunk_number": chunk_number,
            "document_name": "smoke.hwpx",
            "group_id": chunk_number - 1,
            "part": 0,
            **SMOKE_STRUCTURE,
            "article_number": article_number,
            "article_title": article_title,
            "article_order": article_order,
        },
    )


def assert_tree_write_and_cleanup(rag: LegalRAG) -> None:
    """Create a tiny graph tree, verify it, then delete it."""
    dimensions = vector_dimensions(rag, 16)
    rag._embedding_model = FakeEmbeddingModel(dimensions)
    rag._embedding_dimensions = dimensions
    rag.delete_document(TEMP_SOURCE_PATH)

    chunks = [
        smoke_chunk(
            1,
            "제1조(목적) 이 법은 테스트 목적을 규정한다.",
            "제1조",
            "목적",
            1,
        ),
        smoke_chunk(
            2,
            "테스트 목적을 추가로 설명한다.",
            "제1조",
            "목적",
            1,
        ),
        smoke_chunk(
            3,
            "제2조(정의) “테스트용어”란 테스트에서 사용하는 용어를 말한다.",
            "제2조",
            "정의",
            2,
        ),
    ]
    try:
        rag.index(chunks, source_path=TEMP_SOURCE_PATH, source_mtime_ns=1)

        records = rag._run_read(
            """
            MATCH (:Document {source_path: $source_path})
                -[:CONTAINS]->(:LegalUnit {kind: "chapter"})
                -[:CONTAINS]->(:LegalUnit {kind: "section"})
                -[:CONTAINS]->(article:LegalUnit {kind: "article"})
                -[:HAS_CHUNK]->(chunk:Chunk)
            RETURN article.order AS article_order,
                   article.number AS article_number,
                   count(chunk) AS chunk_count
            ORDER BY article_order
            """,
            source_path=TEMP_SOURCE_PATH,
        )
        found = [(record["article_number"], record["chunk_count"]) for record in records]
        if found != [("제1조", 2), ("제2조", 1)]:
            fail(f"unexpected tree graph result: {found}")

        seed_records = rag._run_read(
            """
            MATCH (chunk:Chunk {source_path: $source_path, chunk_number: 2})
            RETURN chunk.id AS id,
                   chunk.content AS content,
                   chunk.chunk_number AS chunk_number,
                   chunk.document_name AS document_name,
                   chunk.group_id AS group_id,
                   chunk.part AS part,
                   chunk.source_path AS source_path,
                   chunk.legal_unit_id AS legal_unit_id,
                   chunk.chapter_id AS chapter_id,
                   chunk.chapter_number AS chapter_number,
                   chunk.chapter_title AS chapter_title,
                   chunk.chapter_order AS chapter_order,
                   chunk.section_id AS section_id,
                   chunk.section_number AS section_number,
                   chunk.section_title AS section_title,
                   chunk.section_order AS section_order,
                   chunk.article_id AS article_id,
                   chunk.article_number AS article_number,
                   chunk.article_title AS article_title,
                   chunk.article_order AS article_order,
                   1.0 AS score
            """,
            source_path=TEMP_SOURCE_PATH,
        )
        if not seed_records:
            fail("missing seed chunk for graph expansion check")
        expanded_chunks = rag._expand_legal_units(
            [document_from_record(seed_records[0])]
        )
        expanded_numbers = [
            chunk.meta["chunk_number"]
            for chunk in expanded_chunks
        ]
        if expanded_numbers != [2, 1]:
            fail(f"unexpected graph expansion result: {expanded_numbers}")

        definition_records = rag._run_read(
            """
            MATCH (term:Term {name: "테스트용어"})
                -[:DEFINED_AS]->(definition:Definition)
                -[:DEFINED_IN]->(:LegalUnit {kind: "article", number: "제2조"})
            MATCH (definition)-[:SUPPORTED_BY]->(:Chunk {
                source_path: $source_path,
                chunk_number: 3
            })
            RETURN term.name AS term
            """,
            source_path=TEMP_SOURCE_PATH,
        )
        found_terms = [record["term"] for record in definition_records]
        if found_terms != ["테스트용어"]:
            fail(f"unexpected definition graph result: {found_terms}")
    finally:
        rag.delete_document(TEMP_SOURCE_PATH)

    leftovers = rag._run_read(
        """
        MATCH (node)
        WHERE node.source_path = $source_path
        RETURN count(node) AS count
        """,
        source_path=TEMP_SOURCE_PATH,
    )
    if leftovers[0]["count"]:
        fail(f"temporary smoke nodes left behind: {leftovers[0]['count']}")
    pass_message(
        "temporary Document -> LegalUnit -> Chunk tree writes and cleans up"
    )


def assert_exact_article_lookup(
    rag: LegalRAG,
    doc_keyword: str,
    article_number: str,
) -> None:
    """Check questions like: what is in Article X?"""
    records = rag._run_read(
        """
        MATCH (document:Document)-[:CONTAINS*1..3]->(article:LegalUnit {kind: "article"})
            -[:HAS_CHUNK]->(:Chunk)
        WHERE document.name CONTAINS $doc_keyword
            AND article.number = $article_number
        RETURN count(article) AS count
        """,
        doc_keyword=doc_keyword,
        article_number=article_number,
    )
    if not records or not records[0]["count"]:
        fail(f"missing article lookup: {doc_keyword} {article_number}")


def assert_phrase_article_lookup(
    rag: LegalRAG,
    doc_keyword: str,
    phrase: str,
    expected_article: str,
) -> None:
    """Check questions like: which article contains this phrase/concept?"""
    records = rag._run_read(
        """
        MATCH (document:Document)-[:CONTAINS*1..3]->(article:LegalUnit {kind: "article"})
        MATCH (article)-[:CONTAINS*0..3]->(unit:LegalUnit)-[:HAS_CHUNK]->(chunk:Chunk)
        WHERE document.name CONTAINS $doc_keyword
            AND chunk.content CONTAINS $phrase
        RETURN article.order AS article_order,
               article.number AS article_number,
               count(chunk) AS matched_chunks
        ORDER BY matched_chunks DESC, article_order
        LIMIT 5
        """,
        doc_keyword=doc_keyword,
        phrase=phrase,
    )
    found = [record["article_number"] for record in records]
    if expected_article not in found:
        fail(
            f"missing phrase lookup: {phrase} "
            f"expected {expected_article}, found {found}"
        )


def assert_article_questions(rag: LegalRAG) -> None:
    start = perf_counter()
    exact_tests = [
        ("개인정보 보호법(법률)", "제15조"),
        ("개인정보 보호법(법률)", "제17조"),
        ("개인정보 보호법 시행령", "제30조"),
        ("대한민국헌법", "제10조"),
        ("인공지능 발전과 신뢰 기반 조성 등에 관한 기본법", "제4조"),
    ]
    phrase_tests = [
        ("개인정보 보호법(법률)", "개인정보의 수집ㆍ이용", "제15조"),
        ("개인정보 보호법(법률)", "자동화된 결정", "제37조의2"),
        ("개인정보 보호법(법률)", "정보주체는 자신의 개인정보 처리와 관련하여", "제4조"),
        ("인공지능 발전과 신뢰 기반 조성 등에 관한 기본법", "인공지능 기본계획", "제6조"),
        ("대한민국헌법", "국민의 모든 자유와 권리", "제37조"),
    ]

    for doc_keyword, article_number in exact_tests:
        assert_exact_article_lookup(rag, doc_keyword, article_number)
    for doc_keyword, phrase, expected_article in phrase_tests:
        assert_phrase_article_lookup(rag, doc_keyword, phrase, expected_article)

    pass_message(f"10 article graph lookups pass in {perf_counter() - start:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local LegalRAG smoke checks.")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only check that backend Python modules compile.",
    )
    parser.add_argument(
        "--skip-write",
        action="store_true",
        help="Skip the temporary Document -> LegalUnit -> Chunk write/cleanup check.",
    )
    parser.add_argument(
        "--skip-lookups",
        action="store_true",
        help="Skip the 10 corpus-dependent article lookup checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compile_backend()
    if args.compile_only:
        return

    rag = LegalRAG()
    assert_constraints(rag)
    if not args.skip_write:
        assert_tree_write_and_cleanup(rag)
    if not args.skip_lookups:
        assert_article_questions(rag)


if __name__ == "__main__":
    main()
