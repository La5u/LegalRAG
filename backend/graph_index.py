import re

from backend.config import CHUNKING_VERSION, EMBEDDING_MODEL, INDEXING_VERSION
from backend.constants import CONCEPT_STOP_TERMS, STRUCTURE_KINDS
from backend.document import Document
from backend.document_types import (
    document_type_from_source_path,
)
from backend.text import strip_korean_particle
from backend.triples import CONCEPT_RELATION_TYPES, LEGAL_RELATIONS



LEGAL_REFERENCE_CONCEPT_PATTERN = re.compile(
    r"^제?\d+(?:조(?:의\d+)?|항|호|목)$|^\d+(?:항|호|목)$|^[가-하]목$|^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]$"
)
WRITE_BATCH_SIZE = 1000

_DELETE_NODE_TYPES = [
    ("chunk", "Chunk"),
    ("definition", "Definition"),
    ("case_node", "Case"),
    ("holding", "Holding"),
    ("legal_unit", "LegalUnit"),
]

_DOCUMENT_PROPS = {
    "name": "document_name",
    "title": "title",
    "logical_id": "logical_id",
    "version_label": "version_label",
    "effective_date": "effective_date",
    "source_name": "source_name",
    "source_url": "source_url",
    "pdf_path": "pdf_path",
    "mtime_ns": "source_mtime_ns",
    "embedding_model": "embedding_model",
    "chunking_version": "chunking_version",
    "indexing_version": "indexing_version",
    "document_type": "document_type",
    "chunk_count": "chunk_count",
}


def row_batches(rows: list[dict]):
    """Yield bounded lists so one Neo4j request cannot grow with a whole statute."""
    for start in range(0, len(rows), WRITE_BATCH_SIZE):
        yield rows[start:start + WRITE_BATCH_SIZE]


def run_batched_query(run_write, query: str, rows: list[dict], **params):
    for batch in row_batches(rows):
        run_write(query, rows=batch, **params)


def build_legal_unit_rows(
    chunk: Document,
    source_path: str,
    document_name: str,
    document_type: str | None = None,
) -> dict:
    if (document_type or document_type_from_source_path(source_path)) not in {
        "statute", "regulation", "contract",
    }:
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
    document_type: str | None = None,
) -> dict:
    resolved_type = document_type or document_type_from_source_path(source_path)
    if resolved_type not in {"precedent", "constitutional_decision"}:
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
            "document_type": resolved_type,
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
    query = """
        UNWIND $rows AS row
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
        """
    run_batched_query(run_write, query, holding_rows,
        source_path=case_row["source_path"],
        document_name=case_row["document_name"])


def _write_filtered_relations(run_write, rows: list[dict], relations: list[str], source_key: str, target_key: str) -> None:
    for relation in relations:
        relation_rows = [row for row in rows if row["relation"] == relation]
        if relation_rows:
            write_relation_edges(run_write, relation, relation_rows, source_key, target_key)


def write_triples(run_write, triple_rows: list[dict]) -> None:
    _write_filtered_relations(run_write, triple_rows, LEGAL_RELATIONS, "subject", "object")


def write_exceptions(run_write, rows: list[dict]) -> None:
    write_relation_edges(run_write, "EXCEPTION_TO", rows, "source", "target")


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

    query = f"""
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
        """
    run_batched_query(run_write, query, rows)


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
    for relation in CONCEPT_RELATION_TYPES:
        run_write(
            f"""
            MATCH ()-[edge:{relation}]->()
            WHERE edge.source_path IN $source_paths
            DELETE edge
            """,
            source_paths=source_paths,
        )
    for _variable, label in _DELETE_NODE_TYPES:
        run_write(
            f"""
            MATCH (node:{label})
            WHERE node.source_path IN $source_paths
            DETACH DELETE node
            """,
            source_paths=source_paths,
        )
    run_write(
        """
        MATCH (document:Document)
        WHERE document.source_path IN $source_paths
        DETACH DELETE document
        """,
        source_paths=source_paths,
    )


def delete_orphan_graph_nodes(run_write) -> None:
    """Remove shared lookup nodes after a group of document changes finishes."""
    run_write(
        """
        MATCH (term:Term)
        WHERE NOT (term)--()
        DETACH DELETE term
        WITH 1 AS _
        MATCH (concept:Concept)
        WHERE NOT (concept)--()
        DETACH DELETE concept
        """
    )


def write_document_node(
    run_write, source_path, document_name, document_metadata, source_mtime_ns, chunk_count,
):
    document_type = (
        document_metadata.get("document_type")
        or document_type_from_source_path(source_path)
    )
    params = {
        "source_path": source_path,
        "document_name": document_name,
        "title": document_metadata.get("title") or document_name.rsplit(".", 1)[0],
        "logical_id": document_metadata.get("logical_id"),
        "version_label": document_metadata.get("version_label", ""),
        "effective_date": document_metadata.get("effective_date"),
        "source_name": document_metadata.get("source_name", ""),
        "source_url": document_metadata.get("source_url", ""),
        "pdf_path": document_metadata.get("pdf_path"),
        "source_mtime_ns": source_mtime_ns,
        "embedding_model": EMBEDDING_MODEL,
        "chunking_version": CHUNKING_VERSION,
        "indexing_version": INDEXING_VERSION,
        "document_type": document_type,
        "chunk_count": chunk_count,
    }
    set_clause = ",\n".join(
        f"            document.{prop} = ${param}"
        for prop, param in _DOCUMENT_PROPS.items()
    )
    run_write(
        f"""
        MERGE (document:Document {{source_path: $source_path}})
        SET
{set_clause},
            document.indexed_at = datetime()
        REMOVE document.max_words
        """,
        **params,
    )


def write_chunk_nodes(run_write, chunk_rows):
    query = """
        UNWIND $rows AS row
        MATCH (document:Document {source_path: row.source_path})
        CREATE (chunk:Chunk:SourceChunk {
            id: row.id,
            content: row.content,
            heading_path: row.heading_path,
            embedding: row.embedding,
            embedding_hash: row.embedding_hash,
            chunk_number: row.chunk_number,
            document_name: row.document_name,
            group_id: row.group_id,
            part: row.part,
            source_path: row.source_path,
            document_title: row.document_title,
            logical_id: row.logical_id,
            version_label: row.version_label,
            effective_date: row.effective_date,
            document_type: row.document_type,
            source_url: row.source_url,
            pdf_path: row.pdf_path,
            page_number: row.page_number
        })
        MERGE (document)-[:CONTAINS]->(chunk)
        """
    run_batched_query(run_write, query, chunk_rows)


def write_legal_unit_nodes(run_write, source_path, legal_unit_rows):
    query = """
        MATCH (document:Document {source_path: $source_path})
        UNWIND $rows AS row
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
        """
    run_batched_query(run_write, query, legal_unit_rows, source_path=source_path)


def write_chunk_legal_unit_links(run_write, chunk_rows):
    query = """
        UNWIND $rows AS row
        MATCH (chunk:Chunk {id: row.id})
        UNWIND row.legal_unit_ids AS legal_unit_id
        MATCH (legal_unit:LegalUnit {id: legal_unit_id})
        MERGE (legal_unit)-[support:SUPPORTED_BY]->(chunk)
        SET support.primary = CASE
            WHEN legal_unit_id = row.legal_unit_id THEN true
            ELSE false
        END
        """
    run_batched_query(run_write, query, chunk_rows)


def write_definition_nodes(run_write, definition_rows):
    query = """
        UNWIND $rows AS row
        MERGE (term:Term {id: row.term_id})
        SET
            term.name = row.term,
            term.source_path = row.source_path,
            term.document_name = row.document_name,
            term.article_number = row.article_number
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
        """
    run_batched_query(run_write, query, definition_rows)


def write_index_transaction(
    tx,
    source_path,
    source_mtime_ns,
    batch,
):
    def run_write(query, **parameters):
        tx.run(query, **parameters).consume()

    delete_source_paths(run_write, [source_path])
    write_document_node(
        run_write,
        source_path,
        batch.document_name,
        batch.document_metadata,
        source_mtime_ns,
        len(batch.chunk_rows),
    )
    if not batch.chunk_rows:
        return

    write_chunk_nodes(run_write, batch.chunk_rows)
    write_legal_unit_nodes(run_write, source_path, batch.legal_unit_rows)
    write_chunk_legal_unit_links(run_write, batch.chunk_rows)
    write_definition_nodes(run_write, batch.definition_rows)
    write_case_nodes(run_write, batch.case_row, batch.holding_rows)
    write_triples(run_write, batch.triple_rows)
    write_exceptions(run_write, batch.exception_rows)
