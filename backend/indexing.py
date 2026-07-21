import logging
import re

from backend.config import (
    CHUNKING_VERSION,
    DEFAULT_MAX_WORDS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    NEO4J_VECTOR_INDEX,
)
from backend.connections import extract_connections
from backend.constants import STRUCTURE_KINDS
from backend.definitions import extract_definitions
from backend.document import Document
from backend.document_types import document_type_from_source_path, has_structural_legal_units
from backend.graph_index import (
    build_case_rows,
    build_legal_unit_rows,
    delete_source_paths,
)
from backend.schema import setup_vector_index
from backend.triples import extract_triples


logger = logging.getLogger(__name__)

EMBEDDING_HEADING_PATTERN = re.compile(
    r"(?:(?<=\s)|^)(?:"
    r"제\d+장\s*[^\s.!?。！？\n]*\.?\s*|"
    r"제\d+절\s*[^\s.!?。！？\n]*\.?\s*|"
    r"제\d+조(?:의\d+)?(?:\([^)]*\))?\s*|"
    r"Article\s+[IVXLCDM\d]+\.?\s*[^\s.!?。！？\n]*\.?\s*|"
    r"Section\s+\d+\.?\s*[^\s.!?。！？\n]*\.?\s*"
    r")\s*",
    re.IGNORECASE,
)


def structure_label(meta: dict, kind: str) -> str | None:
    number = meta.get(f"{kind}_number")
    if not number:
        return None
    title = meta.get(f"{kind}_title")
    if title and kind in {"chapter", "section"}:
        return f"{number} {title}"
    if title and kind == "article":
        return f"{number}({title})"
    return number


def heading_path(meta: dict) -> str | None:
    labels = [
        label
        for kind in STRUCTURE_KINDS
        if (label := structure_label(meta, kind))
    ]
    return " > ".join(labels) if labels else None


def chunk_body(chunk: Document) -> str:
    content = chunk.content.strip()
    deepest_kind = next(
        (
            kind
            for kind in reversed(STRUCTURE_KINDS)
            if chunk.meta.get(f"{kind}_number")
        ),
        None,
    )
    if not deepest_kind:
        return content

    label = structure_label(chunk.meta, deepest_kind)
    if not label:
        return content

    body = content.removeprefix(label).strip()
    if deepest_kind == "article":
        number = chunk.meta.get("article_number")
        title = chunk.meta.get("article_title")
        if number and title:
            body = body.removeprefix(number).strip()
            body = body.removeprefix(f"({title})").strip()
    return body or content


def embedding_text(content: str) -> str:
    return EMBEDDING_HEADING_PATTERN.sub("", content).strip() or content


def encode_chunks(embedding_model, chunks):
    chunk_texts = [embedding_text(chunk_body(chunk)) for chunk in chunks]
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
    has_legal_structure = has_structural_legal_units(source_path)
    for chunk, embedding in zip(chunks, chunk_embeddings, strict=True):
        legal_units = build_legal_unit_rows(chunk, source_path, document_name)
        body = chunk_body(chunk) if has_legal_structure else chunk.content
        path = heading_path(chunk.meta) if has_legal_structure else None
        legal_unit_rows.extend(legal_units["rows"])
        chunk_article_ids[chunk.meta["chunk_number"]] = legal_units["ids"].get("article")
        chunk_rows.append(
            {
                "id": f"{source_path}::chunk-{chunk.meta['chunk_number']}",
                "content": body,
                "heading_path": path,
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
                "article_title": chunk.meta.get("article_title"),
            }
        )

    chunk_documents = [Document(row["content"], row) for row in chunk_rows]

    definition_rows = [
        {
            "id": f"{source_path}::definition-{definition.chunk_number}-{definition.term}",
            "term": definition.term,
            "term_id": f"{source_path}::term-{definition.term}",
            "text": definition.text,
            "chunk_id": f"{source_path}::chunk-{definition.chunk_number}",
            "chunk_number": definition.chunk_number,
            "source_path": source_path,
            "document_name": document_name,
            "article_id": chunk_article_ids.get(definition.chunk_number),
            "article_number": definition.article_number,
            "article_title": definition.article_title,
        }
        for definition in extract_definitions(chunk_documents)
    ]
    case_data = build_case_rows(chunk_documents, source_path, document_name)
    return {
        "document_name": document_name,
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
