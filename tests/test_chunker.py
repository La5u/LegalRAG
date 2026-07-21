import pytest

from backend.chunker import (
    ChunkingState,
    add_chunk,
    chunk_text,
    split_long_block,
    structure_metadata,
    update_chunking_structure,
)
from backend.constants import STRUCTURE_KINDS


# ---------------------------------------------------------------------------
# structure_metadata — detect legal structure from text
# ---------------------------------------------------------------------------
class TestStructureMetadata:
    @pytest.mark.parametrize(
        "text, expected_kind, expected_number",
        [
            ("제1장 총칙", "chapter", "제1장"),
            ("제2장 개인정보의 처리", "chapter", "제2장"),
            ("제1절 목적", "section", "제1절"),
            ("제15조(목적)", "article", "제15조"),
            ("제37조의2 자동화된 결정", "article", "제37조의2"),
            ("① 개인정보란", "clause", "①"),
            ("② 처리방침", "clause", "②"),
            ("1. 수집 항목", "item", "1."),
            ("가. 성명", "subitem", "가."),
            # No match
            ("개인정보 처리에 관한 법률", None, None),
            ("일반 텍스트", None, None),
        ],
    )
    def test_structure_metadata(self, text, expected_kind, expected_number):
        kind, number, title = structure_metadata(text)
        assert kind == expected_kind
        assert number == expected_number

    def test_title_extraction(self):
        kind, number, title = structure_metadata("제15조(목적)")
        assert title == "목적"

    def test_no_title(self):
        kind, number, title = structure_metadata("제15조")
        assert title is None


# ---------------------------------------------------------------------------
# split_long_block — split oversized text at sentence boundaries
# ---------------------------------------------------------------------------
class TestSplitLongBlock:
    def test_short_text_not_split(self):
        text = "이 법은 목적을 규정한다."
        parts = split_long_block(text, max_words=50)
        assert len(parts) == 1

    def test_long_text_split(self):
        # split_long_block uses split_sentences which needs Korean sentence endings.
        sentences = [f"이것은 테스트 문장 번호 {i}이다." for i in range(10)]
        text = " ".join(sentences)
        parts = split_long_block(text, max_words=8)
        assert len(parts) >= 2

    def test_single_oversized_sentence(self):
        # A single long sentence with a Korean ending
        words = " ".join([f"단어{i}" for i in range(15)])
        text = f"{words}이다."
        parts = split_long_block(text, max_words=5)
        assert len(parts) >= 2


# ---------------------------------------------------------------------------
# ChunkingState — tracking structure across blocks
# ---------------------------------------------------------------------------
class TestChunkingState:
    def test_initial_state(self):
        state = ChunkingState()
        assert state.chunks == []
        assert state.structure_counts == {}
        for kind in STRUCTURE_KINDS:
            assert state.current_structures[kind]["number"] is None

    def test_add_chunk(self):
        state = ChunkingState()
        add_chunk(state, "test content", group_id=0, part=0, document_name="test.hwpx")
        assert len(state.chunks) == 1
        assert state.chunks[0].content == "test content"
        assert state.chunks[0].meta["chunk_number"] == 1
        assert state.chunks[0].meta["document_name"] == "test.hwpx"


# ---------------------------------------------------------------------------
# update_chunking_structure — track structure hierarchy
# ---------------------------------------------------------------------------
class TestUpdateChunkingStructure:
    def test_article_detection(self):
        state = ChunkingState()
        update_chunking_structure(state, "제15조(목적)")
        assert state.current_structures["article"]["number"] == "제15조"
        assert state.current_structures["article"]["title"] == "목적"

    def test_chapter_resets_children(self):
        state = ChunkingState()
        # Set an article first
        update_chunking_structure(state, "제15조(목적)")
        assert state.current_structures["article"]["number"] == "제15조"
        # New chapter should reset article
        update_chunking_structure(state, "제2장 처리")
        assert state.current_structures["article"]["number"] is None

    def test_clause_detection(self):
        state = ChunkingState()
        update_chunking_structure(state, "① 개인정보란")
        assert state.current_structures["clause"]["number"] == "①"


# ---------------------------------------------------------------------------
# chunk_text — full chunking pipeline
# ---------------------------------------------------------------------------
class TestChunkText:
    def test_simple_text_produces_chunks(self):
        text = "이 법은 목적을 규정한다."
        chunks = chunk_text(text, max_words=140, document_name="test.hwpx")
        assert len(chunks) >= 1
        assert chunks[0].meta["document_name"] == "test.hwpx"

    def test_multiple_articles(self):
        text = (
            "제1조(목적) 이 법은 개인정보 보호를 목적으로 한다.\n\n"
            "제2조(정의) 개인정보란 개인을 식별할 수 있는 정보를 말한다."
        )
        chunks = chunk_text(text, max_words=140, document_name="test.hwpx")
        assert len(chunks) >= 2

    def test_structure_metadata_propagated(self):
        text = "제15조(목적) 이 법은 개인정보 보호를 목적으로 한다."
        chunks = chunk_text(text, max_words=140, document_name="test.hwpx")
        assert chunks[0].meta["article_number"] == "제15조"
        assert chunks[0].meta["article_title"] == "목적"

    def test_long_blocks_split(self):
        sentences = [f"이것은 테스트 문장 번호 {i}이다." for i in range(20)]
        long_text = "\n\n".join(sentences)
        chunks = chunk_text(long_text, max_words=20, document_name="test.hwpx")
        assert len(chunks) > 1

    def test_empty_text(self):
        chunks = chunk_text("", max_words=140, document_name="test.hwpx")
        assert chunks == []

    def test_chunk_numbers_sequential(self):
        text = "첫번째 문장이다.\n\n두번째 문장이다.\n\n세번째 문장이다."
        chunks = chunk_text(text, max_words=140, document_name="test.hwpx")
        numbers = [c.meta["chunk_number"] for c in chunks]
        assert numbers == list(range(1, len(chunks) + 1))
