import logging
import re
from dataclasses import dataclass

from backend.config import (
    CHUNKING_VERSION,
    DEFAULT_MAX_WORDS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_PASSAGE_PREFIX,
)
from backend.connections import extract_connections
from backend.constants import STRUCTURE_KINDS
from backend.definitions import extract_definitions
from backend.document import Document
from backend.graph_index import (
    build_case_rows,
    build_legal_unit_rows,
    delete_source_paths,
)
from backend.triples import extract_triples


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexBatch:
    """All rows prepared for one atomic document index transaction."""

    document_name: str
    document_metadata: dict
    chunk_rows: list[dict]
    legal_unit_rows: list[dict]
    definition_rows: list[dict]
    triple_rows: list[dict]
    connection_rows: list[dict]
    case_row: dict | None
    holding_rows: list[dict]


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
    chunk_texts = [
        EMBEDDING_PASSAGE_PREFIX + embedding_text(chunk_body(chunk))
        for chunk in chunks
    ]
    if not chunk_texts:
        return []
    return embedding_model.encode(
        chunk_texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def prepare_index_rows(chunks, source_path, document_name, chunk_embeddings):
    legal_unit_rows_by_id = {}
    chunk_rows = []
    chunk_article_ids = {}
    document_type = chunks[0].meta.get("document_type") if chunks else None
    has_legal_structure = document_type in {"statute", "regulation", "contract"}
    for chunk, embedding in zip(chunks, chunk_embeddings, strict=True):
        path_metadata = chunk.meta.get("legal_unit_paths") or [chunk.meta]
        legal_unit_sets = [
            build_legal_unit_rows(
                Document(chunk.content, metadata),
                source_path,
                document_name,
                document_type=document_type,
            )
            for metadata in path_metadata
        ]
        legal_units = legal_unit_sets[0]
        supported_unit_ids = []
        for unit_set in legal_unit_sets:
            for row in unit_set["rows"]:
                legal_unit_rows_by_id[row["id"]] = row
            if unit_set["deepest_id"]:
                supported_unit_ids.append(unit_set["deepest_id"])
        body = chunk_body(chunk) if has_legal_structure else chunk.content
        path = heading_path(chunk.meta) if has_legal_structure else None
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
                "document_title": chunk.meta.get("title") or document_name.rsplit(".", 1)[0],
                "logical_id": chunk.meta.get("logical_id"),
                "version_label": chunk.meta.get("version_label", ""),
                "effective_date": chunk.meta.get("effective_date"),
                "document_type": chunk.meta.get("document_type"),
                "source_url": chunk.meta.get("source_url", ""),
                "pdf_path": chunk.meta.get("pdf_path"),
                "page_number": chunk.meta.get("page_number"),
                "legal_unit_id": legal_units["deepest_id"],
                "legal_unit_ids": list(dict.fromkeys(supported_unit_ids)),
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
    case_data = build_case_rows(
        chunk_documents, source_path, document_name, document_type=document_type,
    )
    return IndexBatch(
        document_name=document_name,
        document_metadata={
            "title": chunks[0].meta.get("title") if chunks else document_name.rsplit(".", 1)[0],
            "logical_id": chunks[0].meta.get("logical_id") if chunks else None,
            "version_label": chunks[0].meta.get("version_label", "") if chunks else "",
            "effective_date": chunks[0].meta.get("effective_date") if chunks else None,
            "document_type": chunks[0].meta.get("document_type") if chunks else None,
            "source_name": chunks[0].meta.get("source_name", "") if chunks else "",
            "source_url": chunks[0].meta.get("source_url", "") if chunks else "",
            "pdf_path": chunks[0].meta.get("pdf_path") if chunks else None,
        },
        chunk_rows=chunk_rows,
        legal_unit_rows=list(legal_unit_rows_by_id.values()),
        definition_rows=definition_rows,
        triple_rows=extract_triples(chunk_documents),
        connection_rows=extract_connections(chunk_documents),
        case_row=case_data["case_row"],
        holding_rows=case_data["holding_rows"],
    )


def index_document(
    embedding_model,
    chunks,
    source_path,
    source_mtime_ns,
):
    logger.info("Indexing %s with %s chunks.", source_path, len(chunks))
    document_name = chunks[0].meta.get("document_name", source_path) if chunks else source_path
    chunk_embeddings = encode_chunks(embedding_model, chunks)

    batch = prepare_index_rows(chunks, source_path, document_name, chunk_embeddings)
    logger.info(
        "Prepared graph rows for %s: %s legal units, %s definitions, "
        "%s triples, %s connections, %s holdings.",
        source_path,
        len(batch.legal_unit_rows),
        len(batch.definition_rows),
        len(batch.triple_rows),
        len(batch.connection_rows),
        len(batch.holding_rows),
    )
    return batch


def document_is_current(run_read, source_path, source_mtime_ns):
    records = run_read(
        """
        MATCH (document:Document {source_path: $source_path})
        WITH properties(document) AS metadata
        WHERE metadata.mtime_ns IS NOT NULL
            AND metadata.embedding_model IS NOT NULL
            AND metadata.chunking_version IS NOT NULL
            AND metadata.max_words IS NOT NULL
        RETURN metadata.mtime_ns = $source_mtime_ns
            AND metadata.embedding_model = $embedding_model
            AND metadata.chunking_version = $chunking_version
            AND metadata.max_words = $max_words
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
