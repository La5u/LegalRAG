import pytest

from backend.document import Document
from backend.records import chunk_meta, chunk_return_fields


def test_document_with_meta_preserves_original_document():
    original = Document("내용", {"id": "chunk-1"}, 0.8)

    updated = original.with_meta(article_number="제1조")

    assert original.meta == {"id": "chunk-1"}
    assert updated.meta == {"id": "chunk-1", "article_number": "제1조"}
    assert updated.content == original.content
    assert updated.score == original.score


def test_document_evidence_key_prefers_id_then_source_location():
    assert Document("내용", {"id": "chunk-1"}).evidence_key == "chunk-1"
    assert Document(
        "내용", {"source_path": "law.hwpx", "chunk_number": 2},
    ).evidence_key == ("law.hwpx", 2)


# ---------------------------------------------------------------------------
# chunk_return_fields — build Neo4j return field list
# ---------------------------------------------------------------------------
class TestChunkReturnFields:
    def test_basic_fields(self):
        result = chunk_return_fields("c", "score")
        assert "c.content AS content" in result
        assert "c.id AS id" in result
        assert "score AS score" in result
        assert "rank" not in result

    def test_with_rank(self):
        result = chunk_return_fields("c", "score", rank="rank")
        assert "rank AS rank" in result

    def test_all_chunk_fields_present(self):
        result = chunk_return_fields("chunk", "vector.similarity")
        for field in ("id", "content", "heading_path", "chunk_number",
                      "document_name", "group_id", "part", "source_path"):
            assert f"chunk.{field} AS {field}" in result


# ---------------------------------------------------------------------------
# chunk_meta — create chunk metadata dict
# ---------------------------------------------------------------------------
class TestChunkMeta:
    def test_default_has_all_fields_except_content(self):
        meta = chunk_meta()
        assert "id" in meta
        assert "content" not in meta
        assert "heading_path" in meta
        assert "chunk_number" in meta
        assert "document_name" in meta

    def test_defaults_are_none(self):
        meta = chunk_meta()
        for value in meta.values():
            assert value is None

    def test_overrides(self):
        meta = chunk_meta(id="test-id", chunk_number=5)
        assert meta["id"] == "test-id"
        assert meta["chunk_number"] == 5
        assert meta["heading_path"] is None
