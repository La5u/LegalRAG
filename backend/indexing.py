import logging

from backend.config import (
    CHUNKING_VERSION,
    DEFAULT_MAX_WORDS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    NEO4J_DATABASE,
    NEO4J_VECTOR_INDEX,
)
from backend.connections import extract_connections
from backend.definitions import extract_definitions
from backend.document_types import document_type_from_source_path
from backend.graph_index import (
    build_case_rows,
    build_legal_unit_rows,
    write_case_nodes,
    write_connections,
    write_triples,
)
from backend.ingest import Document
from backend.schema import setup_vector_index
from backend.triples import extract_triples


logger = logging.getLogger(__name__)


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
        OPTIONAL MATCH (definition:Definition)-[:SUPPORTED_BY]->(chunk:Chunk)
        WHERE chunk.source_path IN deleted_source_paths
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
            definition.chunk_number = row.chunk_number
        MERGE (term)-[:DEFINED_AS]->(definition)
        WITH row, definition
            MATCH (chunk:Chunk {id: row.chunk_id})
        MERGE (definition)-[:SUPPORTED_BY]->(chunk)
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


def encode_chunks(embedding_model, chunks):
    chunk_texts = [chunk.content for chunk in chunks]
    if not chunk_texts:
        return []
    return embedding_model.encode(
        chunk_texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def prepare_index_rows(chunks, source_path, document_name, chunk_embeddings):
    legal_unit_rows = []
    chunk_rows = []
    chunk_article_ids = {}
    for chunk, embedding in zip(chunks, chunk_embeddings, strict=True):
        legal_units = build_legal_unit_rows(chunk, source_path, document_name)
        legal_unit_rows.extend(legal_units["rows"])
        chunk_article_ids[chunk.meta["chunk_number"]] = legal_units["ids"].get("article")
        chunk_rows.append(
            {
                "id": f"{source_path}::chunk-{chunk.meta['chunk_number']}",
                "content": chunk.content,
                "embedding": (
                    embedding.tolist()
                    if hasattr(embedding, "tolist")
                    else list(embedding)
                ),
                "chunk_number": chunk.meta["chunk_number"],
                "document_name": chunk.meta["document_name"],
                "group_id": chunk.meta["group_id"],
                "part": chunk.meta["part"],
                "source_path": source_path,
                "legal_unit_id": legal_units["deepest_id"],
                "article_id": legal_units["ids"].get("article"),
                "article_number": chunk.meta.get("article_number"),
            }
        )

    definition_rows = [
        {
            "id": f"{source_path}::definition-{definition.chunk_number}-{definition.term}",
            "term": definition.term,
            "term_id": f"{source_path}::term-{definition.term}",
            "text": definition.text,
            "chunk_id": f"{source_path}::chunk-{definition.chunk_number}",
            "chunk_number": definition.chunk_number,
            "article_id": chunk_article_ids.get(definition.chunk_number),
            "article_number": definition.article_number,
        }
        for definition in extract_definitions(chunks)
    ]
    chunk_documents = [Document(row["content"], row) for row in chunk_rows]
    case_data = build_case_rows(chunks, source_path, document_name)
    return {
        "chunk_rows": chunk_rows,
        "legal_unit_rows": legal_unit_rows,
        "definition_rows": definition_rows,
        "triple_rows": extract_triples(chunk_documents),
        "connection_rows": extract_connections(chunk_documents),
        "case_row": case_data["case_row"],
        "holding_rows": case_data["holding_rows"],
    }


def index_document(
    run_write,
    run_read,
    embedding_model,
    chunks,
    source_path,
    source_mtime_ns,
    embedding_dimensions,
):
    logger.info("Indexing %s with %s chunks.", source_path, len(chunks))
    document_name = chunks[0].meta.get("document_name", source_path) if chunks else source_path
    setup_vector_index(run_write, run_read, NEO4J_VECTOR_INDEX, embedding_dimensions)

    chunk_embeddings = encode_chunks(embedding_model, chunks)

    index_rows = prepare_index_rows(chunks, source_path, document_name, chunk_embeddings)
    logger.info(
        "Prepared graph rows for %s: %s legal units, %s definitions, "
        "%s triples, %s connections, %s holdings.",
        source_path,
        len(index_rows["legal_unit_rows"]),
        len(index_rows["definition_rows"]),
        len(index_rows["triple_rows"]),
        len(index_rows["connection_rows"]),
        len(index_rows["holding_rows"]),
    )
    return index_rows


def document_is_current(run_read, source_path, source_mtime_ns):
    records = run_read(
        """
        MATCH (document:Document {source_path: $source_path})
        WITH document
        WHERE document.mtime_ns IS NOT NULL
            AND document.embedding_model IS NOT NULL
            AND document.chunking_version IS NOT NULL
            AND document.max_words IS NOT NULL
        RETURN document.mtime_ns = $source_mtime_ns
            AND document.embedding_model = $embedding_model
            AND document.chunking_version = $chunking_version
            AND document.max_words = $max_words
            AS is_current
        """,
        source_path=source_path,
        source_mtime_ns=source_mtime_ns,
        embedding_model=EMBEDDING_MODEL,
        chunking_version=CHUNKING_VERSION,
        max_words=DEFAULT_MAX_WORDS,
    )
    return bool(records and records[0]["is_current"])


def prune_missing_documents(run_read, run_write, source_paths):
    records = run_read(
        """
        MATCH (document:Document)
        WHERE NOT document.source_path IN $source_paths
        RETURN collect(document.source_path) AS source_paths
        """,
        source_paths=source_paths,
    )
    missing_source_paths = records[0]["source_paths"] if records else []
    delete_source_paths(run_write, missing_source_paths)
