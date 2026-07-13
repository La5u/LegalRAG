from backend.document import Document
from backend.indexing import (
    IndexBatch,
    chunk_body,
    document_is_current,
    encode_chunks,
    embedding_text,
    heading_path,
    prepare_index_rows,
    structure_label,
)
from backend.chunker import chunk_text
from backend.graph_index import row_batches


def test_prepare_index_rows_returns_named_index_batch():
    batch = prepare_index_rows([], "documents/raw/empty.txt", "empty.txt", [])

    assert isinstance(batch, IndexBatch)
    assert batch.document_name == "empty.txt"
    assert batch.chunk_rows == []


def test_prepare_index_rows_deduplicates_shared_legal_hierarchy():
    chunks = chunk_text(
        (
            "제4조(의무) 사업주는 조치를 하여야 한다.\n\n"
            "① 다음 각 호의 조치를 한다.\n\n"
            "1. 인력을 배치한다.\n\n"
            "2. 예산을 편성한다."
        ),
        document_name="law.hwpx",
        document_type="statute",
    )
    for chunk in chunks:
        chunk.meta["document_type"] = "statute"

    batch = prepare_index_rows(
        chunks,
        "documents/raw/statutes/law.hwpx",
        "law.hwpx",
        [[0.0] for _chunk in chunks],
    )

    unit_ids = [row["id"] for row in batch.legal_unit_rows]
    assert len(unit_ids) == len(set(unit_ids))
    assert len(batch.chunk_rows[0]["legal_unit_ids"]) == 4


def test_neo4j_rows_are_split_into_bounded_batches():
    rows = [{"id": index} for index in range(1_201)]

    batches = list(row_batches(rows))

    assert [len(batch) for batch in batches] == [500, 500, 201]


class RecordingEmbedder:
    def encode(self, texts, **_kwargs):
        self.texts = texts
        return [[0.0] for _text in texts]


def test_e5_passage_prefix_is_added_before_indexing():
    embedder = RecordingEmbedder()
    encode_chunks(embedder, [Document("법률 내용", {})])
    assert embedder.texts == ["passage: 법률 내용"]


# ---------------------------------------------------------------------------
# structure_label — build "제15조(목적)" or "제1장 총칙" style labels
# ---------------------------------------------------------------------------
class TestStructureLabel:
    def test_article_with_title(self):
        meta = {"article_number": "제15조", "article_title": "목적"}
        assert structure_label(meta, "article") == "제15조(목적)"

    def test_article_without_title(self):
        meta = {"article_number": "제15조"}
        assert structure_label(meta, "article") == "제15조"

    def test_chapter_with_title(self):
        meta = {"chapter_number": "제1장", "chapter_title": "총칙"}
        assert structure_label(meta, "chapter") == "제1장 총칙"

    def test_chapter_without_title(self):
        meta = {"chapter_number": "제1장"}
        assert structure_label(meta, "chapter") == "제1장"

    def test_section_with_title(self):
        meta = {"section_number": "제1절", "section_title": "정의"}
        assert structure_label(meta, "section") == "제1절 정의"

    def test_missing_number_returns_none(self):
        meta = {}
        assert structure_label(meta, "article") is None

    def test_article_title_ignored_for_chapter(self):
        meta = {"chapter_number": "제1장", "chapter_title": "총칙"}
        assert structure_label(meta, "chapter") == "제1장 총칙"


# ---------------------------------------------------------------------------
# heading_path — build "제1장 총칙 > 제1절 정의 > 제15조(목적)" path
# ---------------------------------------------------------------------------
class TestHeadingPath:
    def test_full_path(self):
        meta = {
            "chapter_number": "제1장",
            "chapter_title": "총칙",
            "section_number": "제1절",
            "section_title": "정의",
            "article_number": "제15조",
            "article_title": "목적",
        }
        assert heading_path(meta) == "제1장 총칙 > 제1절 정의 > 제15조(목적)"

    def test_article_only(self):
        meta = {"article_number": "제15조"}
        assert heading_path(meta) == "제15조"

    def test_chapter_and_article(self):
        meta = {
            "chapter_number": "제1장",
            "article_number": "제15조",
            "article_title": "목적",
        }
        result = heading_path(meta)
        assert result == "제1장 > 제15조(목적)"

    def test_empty_meta(self):
        assert heading_path({}) is None


# ---------------------------------------------------------------------------
# chunk_body — strip heading labels from content
# ---------------------------------------------------------------------------
class TestChunkBody:
    def test_article_label_stripped(self):
        chunk = Document(
            content="제15조(목적) 이 법의 목적은 개인정보를 보호하는 것이다.",
            meta={"article_number": "제15조", "article_title": "목적"},
        )
        body = chunk_body(chunk)
        assert body == "이 법의 목적은 개인정보를 보호하는 것이다."

    def test_chapter_label_stripped(self):
        chunk = Document(
            content="제1장 총칙 이 법은 개인정보를 보호한다.",
            meta={"chapter_number": "제1장", "chapter_title": "총칙"},
        )
        body = chunk_body(chunk)
        assert body == "이 법은 개인정보를 보호한다."

    def test_no_structure_returns_content(self):
        chunk = Document(
            content="이 법은 개인정보를 보호한다.",
            meta={},
        )
        assert chunk_body(chunk) == "이 법은 개인정보를 보호한다."

    def test_article_number_only_stripped(self):
        chunk = Document(
            content="제15조 이 법의 목적은 개인정보를 보호하는 것이다.",
            meta={"article_number": "제15조"},
        )
        body = chunk_body(chunk)
        assert body == "이 법의 목적은 개인정보를 보호하는 것이다."

    def test_fallback_to_content_when_body_empty(self):
        chunk = Document(
            content="제15조(목적)",
            meta={"article_number": "제15조", "article_title": "목적"},
        )
        body = chunk_body(chunk)
        assert body == "제15조(목적)"


# ---------------------------------------------------------------------------
# embedding_text — strip heading patterns for embedding
# ---------------------------------------------------------------------------
class TestEmbeddingText:
    def test_removes_article_number(self):
        text = "제15조 이 법은 개인정보를 보호한다."
        result = embedding_text(text)
        assert result == "이 법은 개인정보를 보호한다."

    def test_removes_chapter_heading_up_to_punctuation(self):
        text = "제1장 총칙. 이 법은 개인정보를 보호한다."
        result = embedding_text(text)
        assert result == "이 법은 개인정보를 보호한다."

    def test_removes_section_heading_up_to_punctuation(self):
        text = "제1절 정의. 이 법에서 사용하는 용어의 뜻은 다음 각 호와 같다."
        result = embedding_text(text)
        assert result == "이 법에서 사용하는 용어의 뜻은 다음 각 호와 같다."

    def test_preserves_non_heading_text(self):
        text = "개인정보보호법 제15조 목적 이 법은 개인정보를 보호한다."
        result = embedding_text(text)
        assert "개인정보보호법" in result

    def test_fallback_to_content_when_only_heading(self):
        text = "제15조"
        result = embedding_text(text)
        assert result == "제15조"

    def test_strips_leading_trailing_whitespace(self):
        text = "  제15조  이 법은  "
        result = embedding_text(text)
        assert result == "이 법은"

    def test_article_with_number_suffix(self):
        text = "제15조의2 범위 이 법은 개인정보를 보호한다."
        result = embedding_text(text)
        assert result == "범위 이 법은 개인정보를 보호한다."


# ---------------------------------------------------------------------------
# document_is_current — compare stored index metadata without Neo4j warnings
# ---------------------------------------------------------------------------
class TestDocumentIsCurrent:
    def test_reads_metadata_through_property_map(self):
        captured = {}

        def run_read(query, **params):
            captured["query"] = query
            captured["params"] = params
            return [{"is_current": False}]

        assert document_is_current(run_read, "documents/raw/test.pdf", 123) is False
        assert "WITH properties(document) AS metadata" in captured["query"]
        assert "document.mtime_ns" not in captured["query"]
        assert captured["params"]["source_path"] == "documents/raw/test.pdf"
        assert captured["params"]["source_mtime_ns"] == 123
