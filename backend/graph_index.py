import re

from backend.config import CHUNKING_VERSION, DEFAULT_MAX_WORDS, EMBEDDING_MODEL
from backend.connections import CONNECTION_RELATIONS
from backend.constants import CONCEPT_STOP_TERMS, STRUCTURE_KINDS
from backend.document import Document
from backend.document_types import (
    document_type_from_source_path,
    has_structural_legal_units,
    is_precedent_document,
)
from backend.text import strip_korean_particle
from backend.triples import CONCEPT_RELATION_TYPES, LEGAL_RELATIONS



LEGAL_REFERENCE_CONCEPT_PATTERN = re.compile(
    r"^제?\d+(?:조(?:의\d+)?|항|호|목)$|^\d+(?:항|호|목)$|^[가-하]목$|^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]$"
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
    compact = " ".join(content.split())
    if compact in {"【판시사항】", "【판결요지】", "【결정요지】"}:
        return False
    if re.fullmatch(r"[가-하]\.\s*.{1,24}", compact):
        return False
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
        WITH row, citation, target_document,
            CASE
                WHEN row.law_name IS NULL THEN 0
                WHEN target_document.name STARTS WITH row.law_name + "(" THEN 0
                WHEN target_document.name STARTS WITH row.law_name + " " THEN 1
                WHEN target_document.document_type IN ["statute", "regulation"] THEN 2
                ELSE 3
            END AS target_priority
        ORDER BY target_priority, target_document.name
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


def relink_citation_targets(run_write) -> None:
    run_write(
        """
        MATCH (citation:Citation)
        OPTIONAL MATCH (citation)-[old_link:REFERS_TO|REFERS_TO_ARTICLE]->()
        DELETE old_link
        WITH DISTINCT citation
        OPTIONAL MATCH (target_document:Document)
        WHERE
            (
                citation.law_name IS NULL
                AND target_document.source_path = citation.source_path
            )
            OR (
                citation.law_name IS NOT NULL
                AND target_document.name CONTAINS citation.law_name
            )
        WITH citation, target_document,
            CASE
                WHEN citation.law_name IS NULL THEN 0
                WHEN target_document.name STARTS WITH citation.law_name + "(" THEN 0
                WHEN target_document.name STARTS WITH citation.law_name + " " THEN 1
                WHEN target_document.document_type IN ["statute", "regulation"] THEN 2
                ELSE 3
            END AS target_priority
        ORDER BY citation.id, target_priority, target_document.name
        WITH citation, collect(target_document)[0] AS target_document
        FOREACH (_ IN CASE WHEN target_document IS NULL THEN [] ELSE [1] END |
            MERGE (citation)-[:REFERS_TO]->(target_document)
        )
        WITH citation, target_document
        OPTIONAL MATCH (target_article:LegalUnit {
            source_path: target_document.source_path,
            kind: "article",
            number: citation.article
        })
        FOREACH (_ IN CASE WHEN target_article IS NULL THEN [] ELSE [1] END |
            MERGE (citation)-[:REFERS_TO_ARTICLE]->(target_article)
        )
        """
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
    value = strip_korean_particle(value.strip())
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


def delete_source_paths(run_write, source_paths: list[str]) -> None:
    if not source_paths:
        return
    run_write(
        """
        MATCH (document:Document)
        WHERE document.source_path IN $source_paths
        WITH collect(document) AS documents,
            collect(document.source_path) AS deleted_source_paths
        OPTIONAL MATCH (:Concept)-[legal_relation]->(:Concept)
        WHERE legal_relation.source_path IN deleted_source_paths
        WITH documents, deleted_source_paths, collect(legal_relation) AS legal_relations
        FOREACH (legal_relation IN legal_relations | DELETE legal_relation)
        WITH documents, deleted_source_paths
        OPTIONAL MATCH (chunk:Chunk)
        WHERE chunk.source_path IN deleted_source_paths
        WITH documents, deleted_source_paths, collect(chunk) AS chunks
        OPTIONAL MATCH (definition:Definition)
        WHERE definition.source_path IN deleted_source_paths
        WITH documents, deleted_source_paths, chunks, collect(definition) AS definitions
        OPTIONAL MATCH (citation:Citation)
        WHERE citation.source_path IN deleted_source_paths
        WITH documents, deleted_source_paths, chunks, definitions, collect(citation) AS citations
        OPTIONAL MATCH (case_node:Case)
        WHERE case_node.source_path IN deleted_source_paths
        WITH documents, deleted_source_paths, chunks, definitions, citations, collect(case_node) AS cases
        OPTIONAL MATCH (holding:Holding)
        WHERE holding.source_path IN deleted_source_paths
        WITH documents, deleted_source_paths, chunks, definitions, citations, cases, collect(holding) AS holdings
        OPTIONAL MATCH (legal_unit:LegalUnit)
        WHERE legal_unit.source_path IN deleted_source_paths
        WITH documents, chunks, definitions, citations, cases, holdings, collect(legal_unit) AS legal_units
        FOREACH (definition IN definitions | DETACH DELETE definition)
        FOREACH (citation IN citations | DETACH DELETE citation)
        FOREACH (holding IN holdings | DETACH DELETE holding)
        FOREACH (case_node IN cases | DETACH DELETE case_node)
        FOREACH (chunk IN chunks | DETACH DELETE chunk)
        FOREACH (legal_unit IN legal_units | DETACH DELETE legal_unit)
        FOREACH (document IN documents | DETACH DELETE document)
        WITH 1 AS _
        MATCH (term:Term)
        WHERE NOT (term)--()
        DETACH DELETE term
        WITH 1 AS _
        MATCH (concept:Concept)
        WHERE NOT (concept)--()
        DETACH DELETE concept
        """,
        source_paths=source_paths,
    )


def write_document_node(run_write, source_path, document_name, source_mtime_ns, chunk_count):
    run_write(
        """
        MERGE (document:Document {source_path: $source_path})
        SET
            document.name = $document_name,
            document.mtime_ns = $source_mtime_ns,
            document.embedding_model = $embedding_model,
            document.chunking_version = $chunking_version,
            document.document_type = $document_type,
            document.max_words = $max_words,
            document.chunk_count = $chunk_count,
            document.indexed_at = datetime()
        """,
        source_path=source_path,
        document_name=document_name,
        source_mtime_ns=source_mtime_ns,
        embedding_model=EMBEDDING_MODEL,
        chunking_version=CHUNKING_VERSION,
        document_type=document_type_from_source_path(source_path),
        max_words=DEFAULT_MAX_WORDS,
        chunk_count=chunk_count,
    )


def write_chunk_nodes(run_write, chunk_rows):
    run_write(
        """
        UNWIND $chunk_rows AS row
        MATCH (document:Document {source_path: row.source_path})
        CREATE (chunk:Chunk:SourceChunk {
            id: row.id,
            content: row.content,
            heading_path: row.heading_path,
            embedding: row.embedding,
            chunk_number: row.chunk_number,
            document_name: row.document_name,
            group_id: row.group_id,
            part: row.part,
            source_path: row.source_path
        })
        MERGE (document)-[:CONTAINS]->(chunk)
        """,
        chunk_rows=chunk_rows,
    )


def write_legal_unit_nodes(run_write, source_path, legal_unit_rows):
    run_write(
        """
        MATCH (document:Document {source_path: $source_path})
        UNWIND $legal_unit_rows AS row
        MERGE (legal_unit:LegalUnit {id: row.id})
        SET
            legal_unit.kind = row.kind,
            legal_unit.number = row.number,
            legal_unit.title = row.title,
            legal_unit.order = row.order,
            legal_unit.source_path = row.source_path,
            legal_unit.document_name = row.document_name
        WITH document, collect(row) AS rows
        UNWIND rows AS row
        MATCH (legal_unit:LegalUnit {id: row.id})
        FOREACH (_ IN CASE WHEN row.parent_id IS NULL THEN [1] ELSE [] END |
            MERGE (document)-[:CONTAINS]->(legal_unit)
        )
        WITH row, legal_unit
        OPTIONAL MATCH (parent:LegalUnit {id: row.parent_id})
        FOREACH (_ IN CASE WHEN parent IS NULL THEN [] ELSE [1] END |
            MERGE (parent)-[:CONTAINS]->(legal_unit)
        )
        """,
        source_path=source_path,
        legal_unit_rows=legal_unit_rows,
    )


def write_chunk_legal_unit_links(run_write, chunk_rows):
    run_write(
        """
        UNWIND $chunk_rows AS row
        MATCH (chunk:Chunk {id: row.id})
        OPTIONAL MATCH (legal_unit:LegalUnit {id: row.legal_unit_id})
        FOREACH (_ IN CASE WHEN legal_unit IS NULL THEN [] ELSE [1] END |
            MERGE (legal_unit)-[:SUPPORTED_BY]->(chunk)
        )
        """,
        chunk_rows=chunk_rows,
    )


def write_definition_nodes(run_write, definition_rows):
    run_write(
        """
        UNWIND $definition_rows AS row
        MERGE (term:Term {id: row.term_id})
        SET
            term.name = row.term
        MERGE (definition:Definition {id: row.id})
        SET
            definition.text = row.text,
            definition.chunk_id = row.chunk_id,
            definition.chunk_number = row.chunk_number,
            definition.source_path = row.source_path,
            definition.document_name = row.document_name,
            definition.article_number = row.article_number,
            definition.article_title = row.article_title
        MERGE (term)-[:DEFINED_AS]->(definition)
        WITH row, definition
        OPTIONAL MATCH (article:LegalUnit {id: row.article_id, kind: "article"})
        FOREACH (_ IN CASE WHEN article IS NULL THEN [] ELSE [1] END |
            MERGE (definition)-[:DEFINED_IN]->(article)
        )
        """,
        definition_rows=definition_rows,
    )


def write_index_transaction(
    tx,
    source_path,
    document_name,
    source_mtime_ns,
    chunk_rows,
    legal_unit_rows,
    definition_rows,
    triple_rows,
    connection_rows,
    case_row,
    holding_rows,
):
    def run_write(query, **parameters):
        tx.run(query, **parameters).consume()

    delete_source_paths(run_write, [source_path])
    write_document_node(
        run_write,
        source_path,
        document_name,
        source_mtime_ns,
        len(chunk_rows),
    )
    if not chunk_rows:
        return

    write_chunk_nodes(run_write, chunk_rows)
    write_legal_unit_nodes(run_write, source_path, legal_unit_rows)
    write_chunk_legal_unit_links(run_write, chunk_rows)
    write_definition_nodes(run_write, definition_rows)
    write_case_nodes(run_write, case_row, holding_rows)
    write_triples(run_write, triple_rows)
    write_connections(run_write, connection_rows)
