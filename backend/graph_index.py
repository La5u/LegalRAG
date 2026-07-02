import re

from backend.connections import CONNECTION_RELATIONS
from backend.constants import CONCEPT_STOP_TERMS, QUERY_STOP_TERMS
from backend.document_types import (
    document_type_from_source_path,
    has_structural_legal_units,
    is_precedent_document,
)
from backend.ingest import Document, STRUCTURE_KINDS
from backend.records import chunk_meta
from backend.triples import CONCEPT_RELATION_TYPES, LEGAL_RELATIONS



LEGAL_REFERENCE_CONCEPT_PATTERN = re.compile(
    r"^제\d+(?:조(?:의\d+)?|항|호|목)$|^\d+(?:항|호|목)$|^[가-하]목$"
)
def build_legal_unit_rows(
    chunk: Document,
    source_path: str,
    document_name: str,
) -> dict:
    if not has_structural_legal_units(source_path):
        return {"ids": {}, "deepest_id": None, "rows": []}

    rows = []
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

    return {"ids": unit_ids, "deepest_id": parent_id, "rows": rows}


def case_number_from_source(source_path: str, document_name: str) -> str:
    name = document_name.rsplit(".", maxsplit=1)[0]
    if name.startswith("판례_"):
        return name.removeprefix("판례_")
    if name.startswith("헌재결정례_"):
        return name.removeprefix("헌재결정례_")
    if name.startswith("lbox_precedent_"):
        return name
    return name or source_path


def build_case_rows(
    chunks: list[Document],
    source_path: str,
    document_name: str,
) -> dict:
    if not is_precedent_document(source_path):
        return {"case_row": None, "holding_rows": []}

    case_number = case_number_from_source(source_path, document_name)
    case_id = f"{source_path}::case"
    holding_rows = []
    for chunk in chunks:
        if is_holding_chunk(chunk.content):
            holding_rows.append(
                {
                    "id": f"{source_path}::holding-{chunk.meta['chunk_number']}",
                    "case_id": case_id,
                    "text": chunk.content,
                    "chunk_id": f"{source_path}::chunk-{chunk.meta['chunk_number']}",
                    "chunk_number": chunk.meta["chunk_number"],
                }
            )

    if not holding_rows:
        for chunk in chunks[:3]:
            holding_rows.append(
                {
                    "id": f"{source_path}::holding-{chunk.meta['chunk_number']}",
                    "case_id": case_id,
                    "text": chunk.content,
                    "chunk_id": f"{source_path}::chunk-{chunk.meta['chunk_number']}",
                    "chunk_number": chunk.meta["chunk_number"],
                }
            )

    return {
        "case_row": {
            "id": case_id,
            "source_path": source_path,
            "document_name": document_name,
            "document_type": document_type_from_source_path(source_path),
            "case_number": case_number,
        },
        "holding_rows": holding_rows,
    }


def is_holding_chunk(content: str) -> bool:
    holding_terms = ("판시", "쟁점", "주문", "이유", "판결요지", "결정요지")
    return any(term in content for term in holding_terms)


def write_case_nodes(run_write, case_row: dict | None, holding_rows: list[dict]) -> None:
    if not case_row:
        return

    run_write(
        """
        MATCH (document:Document {source_path: $source_path})
        MERGE (case_node:Case {source_path: $source_path})
        SET
            case_node.id = $case_id,
            case_node.case_number = $case_number,
            case_node.document_name = $document_name,
            case_node.document_type = $document_type
        MERGE (document)-[:CONTAINS]->(case_node)
        """,
        source_path=case_row["source_path"],
        case_id=case_row["id"],
        case_number=case_row["case_number"],
        document_name=case_row["document_name"],
        document_type=case_row["document_type"],
    )
    run_write(
        """
        UNWIND $holding_rows AS row
        MATCH (case_node:Case {source_path: $source_path})
        MATCH (chunk:Chunk {id: row.chunk_id})
        MERGE (holding:Holding {id: row.id})
        SET
            holding.text = row.text,
            holding.chunk_number = row.chunk_number,
            holding.source_path = $source_path,
            holding.document_name = $document_name
        MERGE (case_node)-[:HAS_HOLDING]->(holding)
        MERGE (holding)-[:SUPPORTED_BY]->(chunk)
        """,
        source_path=case_row["source_path"],
        document_name=case_row["document_name"],
        holding_rows=holding_rows,
    )


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
    citation_rows = [
        row
        for row in connection_rows
        if row["relation"] == "CITES"
    ]
    if citation_rows:
        write_citation_nodes(run_write, citation_rows)

    for relation in CONNECTION_RELATIONS:
        if relation == "CITES":
            continue
        relation_rows = [
            row
            for row in connection_rows
            if row["relation"] == relation
        ]
        if relation_rows:
            write_relation_edges(run_write, relation, relation_rows, "source", "target")


def write_citation_nodes(run_write, rows: list[dict]) -> None:
    run_write(
        """
        UNWIND $rows AS row
        MATCH (chunk:Chunk {id: row.chunk_id})
        MERGE (citation:Citation {id: row.chunk_id + "::citation::" + row.raw_text})
        SET
            citation.raw_text = row.raw_text,
            citation.law_name = row.law_name,
            citation.article = row.article,
            citation.clause = row.clause,
            citation.sentence = row.sentence,
            citation.source_path = row.source_path,
            citation.document_name = row.document_name,
            citation.chunk_number = row.chunk_number
        MERGE (chunk)-[:CITES]->(citation)
        WITH row, citation
        OPTIONAL MATCH (target_document:Document)
        WHERE
            (
                row.law_name IS NULL
                AND target_document.source_path = row.source_path
            )
            OR (
                row.law_name IS NOT NULL
                AND target_document.name CONTAINS row.law_name
            )
        WITH row, citation, collect(target_document)[0] AS target_document
        FOREACH (_ IN CASE WHEN target_document IS NULL THEN [] ELSE [1] END |
            MERGE (citation)-[:REFERS_TO]->(target_document)
        )
        WITH row, citation, target_document
        OPTIONAL MATCH (target_article:LegalUnit {
            source_path: target_document.source_path,
            kind: "article",
            number: row.article
        })
        FOREACH (_ IN CASE WHEN target_article IS NULL THEN [] ELSE [1] END |
            MERGE (citation)-[:REFERS_TO_ARTICLE]->(target_article)
        )
        """,
        rows=rows,
    )


def write_relation_edges(
    run_write,
    relation: str,
    rows: list[dict],
    source_key: str,
    target_key: str,
) -> None:
    if relation not in CONCEPT_RELATION_TYPES:
        raise ValueError(f"Unsupported graph relation: {relation}")

    rows = clean_relation_rows(rows, source_key, target_key)
    if not rows:
        return

    run_write(
        f"""
        UNWIND $rows AS row
        MERGE (source:Concept {{name: row.{source_key}}})
        MERGE (target:Concept {{name: row.{target_key}}})
        MERGE (source)-[edge:{relation} {{
            source_path: row.source_path,
            chunk_id: row.chunk_id,
            sentence: row.sentence
        }}]->(target)
        SET
            edge.document_name = row.document_name,
            edge.chunk_number = row.chunk_number,
            edge.relation_phrase = row.relation_phrase
        """,
        rows=rows,
    )


def clean_relation_rows(
    rows: list[dict],
    source_key: str,
    target_key: str,
) -> list[dict]:
    cleaned_rows = []
    for row in rows:
        source = clean_concept_name(row.get(source_key))
        target = clean_concept_name(row.get(target_key))
        if is_noisy_concept(source) or is_noisy_concept(target) or source == target:
            continue

        cleaned = dict(row)
        cleaned[source_key] = source
        cleaned[target_key] = target
        cleaned_rows.append(cleaned)
    return cleaned_rows


def clean_concept_name(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip()
    if value.startswith("n") and len(value) > 1 and "가" <= value[1] <= "힣":
        value = value[1:]
    return value


def is_noisy_concept(value: str | None) -> bool:
    return bool(
        not value
        or len(value) <= 1
        or value in CONCEPT_STOP_TERMS
        or is_legal_reference_concept(value)
    )


def is_legal_reference_concept(value: str | None) -> bool:
    return bool(value and LEGAL_REFERENCE_CONCEPT_PATTERN.fullmatch(value))
