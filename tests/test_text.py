import pytest

from backend.text import extract_tokens, split_sentences, strip_korean_particle
from backend.loaders import HANJA_PATTERN, KOREAN_PATTERN


# ---------------------------------------------------------------------------
# split_sentences
# ---------------------------------------------------------------------------
class TestSplitSentences:
    @pytest.mark.parametrize(
        "text, expected",
        [
            # Korean period ending
            ("이 법은 목적을 규정한다.", ["이 법은 목적을 규정한다."]),
            # Multiple sentences
            (
                "제1조는 목적을 규정한다. 제2조는 정의를 규정한다.",
                ["제1조는 목적을 규정한다.", "제2조는 정의를 규정한다."],
            ),
            # Korean sentence endings without period
            ("이 법은 개인정보를 보호한다", ["이 법은 개인정보를 보호한다"]),
            # Multiple endings
            (
                "정보주체는 동의하여야 한다. 개인정보처리자는 통지하여야 한다.",
                ["정보주체는 동의하여야 한다.", "개인정보처리자는 통지하여야 한다."],
            ),
            # Empty / whitespace only
            ("", []),
            ("   ", []),
            # Newlines between sentences
            (
                "제1조 목적이다.\n제2조 정의이다.",
                ["제1조 목적이다.", "제2조 정의이다."],
            ),
        ],
    )
    def test_split_sentences(self, text, expected):
        assert split_sentences(text) == expected


# ---------------------------------------------------------------------------
# extract_tokens
# ---------------------------------------------------------------------------
class TestExtractTokens:
    @pytest.mark.parametrize(
        "query, expected",
        [
            ("개인정보 처리", ["개인정보", "처리"]),
            ("Hello World 123", ["Hello", "World", "123"]),
            # Deduplication preserves first occurrence
            ("개인정보 개인정보 처리", ["개인정보", "처리"]),
            # Short tokens (single char) are excluded by TOKEN_PATTERN
            ("a 테스트", ["테스트"]),
        ],
    )
    def test_extract_tokens(self, query, expected):
        result = extract_tokens(query)
        assert result == expected


# ---------------------------------------------------------------------------
# strip_korean_particle
# ---------------------------------------------------------------------------
class TestStripKoreanParticle:
    @pytest.mark.parametrize(
        "term, expected",
        [
            ("개인정보처리자는", "개인정보처리자"),
            ("정보주체에게", "정보주체"),
            ("개인정보를", "개인정보"),
            ("성명은", "성명"),
            # Kiwi splits "이 법은" into separate tokens; combine_tokens preserves spaces
            ("이 법은", "이 법"),
            # Already stripped
            ("개인정보", "개인정보"),
            # Empty string
            ("", ""),
        ],
    )
    def test_strip_particle(self, term, expected):
        assert strip_korean_particle(term) == expected


# ---------------------------------------------------------------------------
# Hanja pattern (Chinese characters in Korean legal text)
# ---------------------------------------------------------------------------
class TestHanjaPattern:
    """HANJA_PATTERN matches CJK Unified Ideographs used in Korean legal docs.
    Hanja appears in older statutes and constitutional court decisions."""

    @pytest.mark.parametrize(
        "text, should_match",
        [
            # Pure Korean — no hanja
            ("개인정보 보호법", False),
            # Hanja characters (의 = obligation, 인 = person, 법 = law)
            ("義務", True),  # 의무
            ("人事", True),  # 인사
            ("法律", True),  # 법률
            # Mixed Korean + hanja
            ("개인정보(個人情報) 보호법", True),
            # English only — no hanja
            ("Personal Information Protection Act", False),
        ],
    )
    def test_hanja_detection(self, text, should_match):
        assert bool(HANJA_PATTERN.search(text)) == should_match

    def test_hanja_count_in_legal_text(self):
        text = "헌법 제10조는 人間의 尊嚴과 王者의 價値를 보장한다."
        count = len(HANJA_PATTERN.findall(text))
        assert count >= 4

    def test_korean_pattern_does_not_match_hanja(self):
        text = "義務"
        assert not KOREAN_PATTERN.search(text)
        assert HANJA_PATTERN.search(text)


# ---------------------------------------------------------------------------
# Korean pattern basics
# ---------------------------------------------------------------------------
class TestKoreanPattern:
    @pytest.mark.parametrize(
        "text, should_match",
        [
            ("개인정보", True),
            ("abc", False),
            ("123", False),
            ("한글", True),
        ],
    )
    def test_korean_detection(self, text, should_match):
        assert bool(KOREAN_PATTERN.search(text)) == should_match
