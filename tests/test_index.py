import pytest
from pathlib import Path

from backend.index import _normalized_stem, document_id, PROJECT_ROOT


# ---------------------------------------------------------------------------
# _normalized_stem — remove parenthesized content, lowercase
# ---------------------------------------------------------------------------
class TestNormalizedStem:
    @pytest.mark.parametrize(
        "path, expected",
        [
            (Path("/a/b/个人信息保護法.pdf"), "个人信息保護法"),
            (Path("/a/b/Hello (enacted 2023).txt"), "hello"),
            (Path("/a/b/NoParentheses.pdf"), "noparentheses"),
            (Path("/a/b/Mixed(1).hwpx"), "mixed"),
        ],
    )
    def test_removes_parens_and_lowercases(self, path, expected):
        assert _normalized_stem(path) == expected

    def test_empty_parens(self):
        assert _normalized_stem(Path("/a/b/file().pdf")) == "file"

    def test_nested_parens_only_removes_last(self):
        result = _normalized_stem(Path("/a/b/file(foo)bar.pdf"))
        assert result == "filebar"


# ---------------------------------------------------------------------------
# document_id — relative path as posix
# ---------------------------------------------------------------------------
class TestDocumentId:
    def test_returns_posix_relative_path(self):
        path = PROJECT_ROOT / "documents" / "raw" / "statutes" / "test.pdf"
        assert document_id(path) == "documents/raw/statutes/test.pdf"

    def test_strips_project_root(self):
        path = PROJECT_ROOT / "some" / "nested" / "file.txt"
        assert document_id(path) == "some/nested/file.txt"
