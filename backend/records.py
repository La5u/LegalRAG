from backend.document import Document


CHUNK_FIELDS = (
    "id",
    "content",
    "heading_path",
    "chunk_number",
    "document_name",
    "group_id",
    "part",
    "source_path",
)
CHUNK_METADATA_FIELDS = (
    "document_title",
    "logical_id",
    "version_label",
    "effective_date",
    "document_type",
    "source_url",
    "pdf_path",
    "page_number",
)


def chunk_return_fields(alias: str, score: str, rank: str | None = None) -> str:
    fields = [
        f"{alias}.{field} AS {field}"
        for field in (*CHUNK_FIELDS, *CHUNK_METADATA_FIELDS)
    ]
    fields.append(f"{score} AS score")
    if rank:
        fields.append(f"{rank} AS rank")
    return ",\n                ".join(fields)


def chunk_meta(**overrides) -> dict:
    meta = {
        field: None
        for field in (*CHUNK_FIELDS, *CHUNK_METADATA_FIELDS)
        if field != "content"
    }
    meta.update(overrides)
    return meta


def document_from_record(record) -> Document:
    record_keys = set(record.keys())
    missing_fields = [
        field
        for field in ("content", *CHUNK_FIELDS)
        if field not in record_keys
    ]
    if missing_fields:
        raise KeyError(
            f"Neo4j record is missing required fields: {', '.join(missing_fields)}"
        )
    return Document(
        content=record["content"],
        meta=chunk_meta(
            **{
                field: record[field]
                for field in (*CHUNK_FIELDS, *CHUNK_METADATA_FIELDS)
                if field != "content"
                and field in record_keys
            }
        ),
        score=record.get("score"),
    )


def documents_from_records(records) -> list[Document]:
    return [document_from_record(record) for record in records]
