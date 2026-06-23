import re

from backend.connections import CONNECTION_RELATIONS
from backend.ingest import Document, STRUCTURE_KINDS
from backend.triples import LEGAL_RELATIONS


RELATION_TYPES = LEGAL_RELATIONS + CONNECTION_RELATIONS
QUERY_STOP_TERMS = {
    "what",
    "which",
    "where",
    "when",
    "shall",
    "does",
    "against",
    "defined",
    "united",
    "states",
    "article",
    "section",
}


def build_legal_unit_rows(
    chunk: Document,
    source_path: str,
    document_name: str,
    rows: list[dict],
) -> dict:
    unit_ids = {}
    parent_id = None
    path_parts = []
    for kind in STRUCTURE_KINDS:
        number = chunk.meta.get(f"{kind}_number")
        if not number:
            continue

        path_parts.append(f"{kind}-{number}")
        unit_id = f"{source_path}::legal-unit::{'/'.join(path_parts)}"
        unit_ids[kind] = unit_id
        rows.append(
            {
                "id": unit_id,
                "kind": kind,
                "number": number,
                "title": chunk.meta.get(f"{kind}_title"),
                "order": chunk.meta.get(f"{kind}_order"),
                "source_path": source_path,
                "document_name": document_name,
                "parent_id": parent_id,
            }
        )
        parent_id = unit_id

    return {"ids": unit_ids, "deepest_id": parent_id}


def write_triples(run_write, triple_rows: list[dict]) -> None:
    for relation in LEGAL_RELATIONS:
        relation_rows = [
            row
            for row in triple_rows
            if row["relation"] == relation
        ]
        if relation_rows:
            write_relation_edges(run_write, relation, relation_rows, "subject", "object")


def write_connections(run_write, connection_rows: list[dict]) -> None:
    for relation in CONNECTION_RELATIONS:
        relation_rows = [
            row
            for row in connection_rows
            if row["relation"] == relation
        ]
        if relation_rows:
            write_relation_edges(run_write, relation, relation_rows, "source", "target")


def write_relation_edges(
    run_write,
    relation: str,
    rows: list[dict],
    source_key: str,
    target_key: str,
) -> None:
    run_write(
        f"""
        UNWIND $rows AS row
        MERGE (source:Concept {{name: row.{source_key}}})
        MERGE (target:Concept {{name: row.{target_key}}})
        CREATE (source)-[:{relation} {{
            source_path: row.source_path,
            document_name: row.document_name,
            chunk_id: row.chunk_id,
            chunk_number: row.chunk_number,
            legal_unit_id: row.legal_unit_id,
            article_number: row.article_number,
            sentence: row.sentence
        }}]->(target)
        """,
        rows=rows,
    )


def graph_fact_documents(run_read, query: str, chunks: list[Document]) -> list[Document]:
    relation_records = run_read(
        """
        MATCH (source:Concept)-[relation]->(target:Concept)
        WHERE type(relation) IN $relations
            AND (
                relation.chunk_id IN $chunk_ids
                OR relation.legal_unit_id IN $legal_unit_ids
                OR relation.article_number IN $article_numbers
                OR any(term IN $query_terms WHERE source.name CONTAINS term)
                OR any(term IN $query_terms WHERE target.name CONTAINS term)
                OR any(term IN $query_terms WHERE relation.sentence CONTAINS term)
            )
        WITH source, relation, target,
            CASE WHEN type(relation) IN $preferred_relations THEN 0 ELSE 1 END
                AS relation_priority,
            CASE
                WHEN source.name IN $article_terms THEN 0
                WHEN target.name IN $article_terms THEN 0
                WHEN relation.article_number IN $article_terms THEN 0
                ELSE 1
            END AS article_priority
        ORDER BY relation_priority, article_priority, relation.chunk_number
        RETURN
            source.name AS source,
            type(relation) AS relation,
            target.name AS target,
            relation.sentence AS sentence,
            relation.document_name AS document_name,
            relation.source_path AS source_path,
            relation.chunk_id AS chunk_id,
            relation.chunk_number AS chunk_number,
            relation.legal_unit_id AS legal_unit_id,
            relation.article_number AS article_number
        LIMIT 20
        """,
        relations=RELATION_TYPES,
        preferred_relations=preferred_relations(query),
        article_terms=article_terms(query),
        chunk_ids=[chunk.meta["id"] for chunk in chunks if chunk.meta.get("id")],
        legal_unit_ids=[
            chunk.meta["legal_unit_id"]
            for chunk in chunks
            if chunk.meta.get("legal_unit_id")
        ],
        article_numbers=[
            chunk.meta["article_number"]
            for chunk in chunks
            if chunk.meta.get("article_number")
        ],
        query_terms=query_terms(query),
    )
    relation_documents = [
        Document(
            content=(
                "Graph fact:\n"
                f"{record['source']} -{record['relation']}-> {record['target']}\n"
                f"Source sentence: {record['sentence']}"
            ),
            meta={
                "id": (
                    f"graph-fact::{record['relation']}::"
                    f"{record['chunk_id']}::{index}"
                ),
                "chunk_number": f"graph-{index}",
                "document_name": record["document_name"],
                "group_id": f"graph-{index}",
                "part": 0,
                "source_path": record["source_path"],
                "legal_unit_id": record["legal_unit_id"],
                "chapter_id": None,
                "chapter_number": None,
                "chapter_title": None,
                "chapter_order": None,
                "section_id": None,
                "section_number": None,
                "section_title": None,
                "section_order": None,
                "article_id": record["legal_unit_id"],
                "article_number": record["article_number"],
                "article_title": None,
                "article_order": None,
            },
        )
        for index, record in enumerate(relation_records, start=1)
    ]
    definition_records = []
    if wants_definition(query):
        definition_records = run_read(
            """
            MATCH (term:Term)-[:DEFINED_AS]->(definition:Definition)-[:SUPPORTED_BY]->(chunk:Chunk)
            WHERE chunk.id IN $chunk_ids
                OR definition.article_id IN $legal_unit_ids
                OR definition.article_number IN $article_numbers
                OR any(query_term IN $query_terms WHERE term.name CONTAINS query_term)
            RETURN
                term.name AS term,
                definition.text AS text,
                definition.document_name AS document_name,
                definition.source_path AS source_path,
                chunk.id AS chunk_id,
                definition.chunk_number AS chunk_number,
                definition.article_id AS article_id,
                definition.article_number AS article_number
            ORDER BY definition.chunk_number
            LIMIT 10
            """,
            chunk_ids=[chunk.meta["id"] for chunk in chunks if chunk.meta.get("id")],
            legal_unit_ids=[
                chunk.meta["legal_unit_id"]
                for chunk in chunks
                if chunk.meta.get("legal_unit_id")
            ],
            article_numbers=[
                chunk.meta["article_number"]
                for chunk in chunks
                if chunk.meta.get("article_number")
            ],
            query_terms=query_terms(query),
        )
    definition_documents = [
        Document(
            content=(
                "Graph fact:\n"
                f"{record['term']} -DEFINED_AS-> Definition\n"
                f"Definition: {record['text']}"
            ),
            meta={
                "id": f"graph-fact::DEFINED_AS::{record['chunk_id']}::{index}",
                "chunk_number": f"graph-definition-{index}",
                "document_name": record["document_name"],
                "group_id": f"graph-definition-{index}",
                "part": 0,
                "source_path": record["source_path"],
                "legal_unit_id": record["article_id"],
                "chapter_id": None,
                "chapter_number": None,
                "chapter_title": None,
                "chapter_order": None,
                "section_id": None,
                "section_number": None,
                "section_title": None,
                "section_order": None,
                "article_id": record["article_id"],
                "article_number": record["article_number"],
                "article_title": None,
                "article_order": None,
            },
        )
        for index, record in enumerate(definition_records, start=1)
    ]
    return relation_documents + definition_documents


def query_terms(query: str) -> list[str]:
    terms = re.findall(r"[가-힣A-Za-z0-9]{2,}", query)
    return [
        term
        for term in terms
        if term.casefold() not in QUERY_STOP_TERMS
    ][:8]


def wants_definition(query: str) -> bool:
    lowered = query.casefold()
    return any(
        term in lowered
        for term in ["정의", "이란", "란 무엇", "define", "defined", "definition", "mean"]
    )


def article_terms(query: str) -> list[str]:
    return re.findall(r"제\d+조(?:의\d+)?", query)


def preferred_relations(query: str) -> list[str]:
    preferred = []
    if "예외" in query:
        preferred.append("EXCEPTION_TO")
    if "인용" in query or re.search(r"제\d+조", query):
        preferred.append("CITES")
    if "통지" in query or "알려" in query:
        preferred.append("MUST_NOTIFY")
    if "요구" in query or "청구" in query:
        preferred.append("MAY_REQUEST")
    if "금지" in query:
        preferred.append("PROHIBITS")
    if "의무" in query:
        preferred.append("HAS_DUTY")
    if "할 수" in query or "권리" in query:
        preferred.append("MAY")
    return preferred
