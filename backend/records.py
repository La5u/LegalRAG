from backend.ingest import Document


CHUNK_FIELDS = (
    "id",
    "content",
    "chunk_number",
    "document_name",
    "group_id",
    "part",
    "source_path",
    "legal_unit_id",
    "chapter_id",
    "chapter_number",
    "chapter_title",
    "chapter_order",
    "section_id",
    "section_number",
    "section_title",
    "section_order",
    "article_id",
    "article_number",
    "article_title",
    "article_order",
)


def chunk_return_fields(alias: str, score: str, rank: str | None = None) -> str:
    fields = [
        f"{alias}.{field} AS {field}"
        for field in CHUNK_FIELDS
    ]
    fields.append(f"{score} AS score")
    if rank:
        fields.append(f"{rank} AS rank")
    return ",\n                ".join(fields)


def document_from_record(record) -> Document:
    return Document(
        content=record["content"],
        meta={
            field: record[field]
            for field in CHUNK_FIELDS
            if field != "content"
        },
        score=record.get("score"),
    )
