import pytest

from backend.document import Document
from backend.definitions import Definition, extract_definitions


def _make_chunk(content: str, article_title: str = "정의", **overrides) -> Document:
    meta = {
        "chunk_number": 1,
        "document_name": "test.hwpx",
        "source_path": "documents/raw/statutes/test.hwpx",
        "article_number": "제2조",
        "article_title": article_title,
        "article_order": 1,
        "group_id": 0,
        "part": 0,
        **overrides,
    }
    return Document(content=content, meta=meta)


class TestExtractDefinitions:
    def test_korean_quoted_definition(self):
        chunk = _make_chunk(
            '"개인정보"란 성명, 주민등록번호 등 개인을 식별할 수 있는情報を 말한다.'
        )
        defs = extract_definitions([chunk])
        assert len(defs) == 1
        assert defs[0].term == "개인정보"

    def test_korean_definition_with_plain_quotes(self):
        chunk = _make_chunk(
            '"주민등록번호"란 주민등록증에 기록된 번호를 말한다.'
        )
        defs = extract_definitions([chunk])
        assert len(defs) == 1
        assert defs[0].term == "주민등록번호"

    def test_english_definition(self):
        chunk = _make_chunk(
            "Personal Information means any information relating to an identified person.",
            article_title="definitions",
        )
        defs = extract_definitions([chunk])
        assert len(defs) == 1
        assert defs[0].term == "Personal Information"

    def test_english_shall_mean(self):
        chunk = _make_chunk(
            "Data means any digital information. Data shall mean processed information.",
            article_title="definitions",
        )
        defs = extract_definitions([chunk])
        assert len(defs) >= 1

    def test_non_definition_chunk_rejected(self):
        chunk = _make_chunk(
            "이 법은 개인정보 보호를 목적으로 한다.",
            article_title="목적",
        )
        defs = extract_definitions([chunk])
        assert defs == []

    def test_non_definition_without_english_rejected(self):
        chunk = _make_chunk(
            "개인정보처리자는 정보주체의 동의를 받아야 한다.",
            article_title="일반규칙",
        )
        defs = extract_definitions([chunk])
        assert defs == []

    def test_multiple_definitions_in_one_chunk(self):
        chunk = _make_chunk(
            '"개인정보"란 성명을 말한다. "처리자"란 개인정보를 처리하는 자를 말한다.'
        )
        defs = extract_definitions([chunk])
        assert len(defs) == 2
        assert {d.term for d in defs} == {"개인정보", "처리자"}

    def test_definition_metadata(self):
        chunk = _make_chunk(
            '"개인정보"란 성명을 말한다.',
            chunk_number=5,
            article_number="제3조",
            article_title="정의",
        )
        defs = extract_definitions([chunk])
        assert defs[0].chunk_number == 5
        assert defs[0].article_number == "제3조"
        assert defs[0].article_title == "정의"


class TestDefinitionDataclass:
    def test_frozen(self):
        d = Definition(term="test", text="test text", chunk_number=1, article_number=None, article_title=None)
        with pytest.raises(AttributeError):
            d.term = "changed"
