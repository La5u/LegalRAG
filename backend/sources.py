import re
from pathlib import Path

from backend.constants import TOKEN_PATTERN
from backend.document_types import document_type_from_source_path


SOURCE_ALIASES = {
    "국세기본법": ("국세기본법",),
    "근로기준법": ("근로기준법",),
    "노동조합법": ("노동조합 및 노동관계조정법",),
    "노동조합 및 노동관계조정법": ("노동조합 및 노동관계조정법",),
    "상가임대차법 시행령": ("상가건물 임대차보호법 시행령",),
    "상가건물 임대차보호법 시행령": ("상가건물 임대차보호법 시행령",),
    "상가임대차법": ("상가건물 임대차보호법(",),
    "상가건물 임대차보호법": ("상가건물 임대차보호법(",),
    "상가 임대차": ("상가건물 임대차보호법(",),
    "상법": ("상법(",),
    "전자상거래법": ("전자상거래 등에서의 소비자보호에 관한 법률",),
    "주택임대차보호법": ("주택임대차보호법",),
    "주택임대차": ("주택임대차보호법",),
    "임차주택": ("주택임대차보호법",),
    "행정소송법": ("행정소송법",),
    "행정심판법": ("행정심판법",),
    "민법": ("민법(",),
    "형법": ("형법(",),
    "약관법": ("약관의 규제에 관한 법률",),
    "표시광고법": ("표시ㆍ광고의 공정화에 관한 법률", "표시광고"),
    "개인정보 보호법 시행령": ("개인정보 보호법 시행령",),
    "개인정보 보호법": ("개인정보 보호법(",),
    "표준 개인정보 보호지침": ("표준 개인정보 보호지침",),
    "전자금융거래법": ("전자금융거래법",),
    "소비자기본법": ("소비자기본법",),
    "헌법": ("대한민국헌법",),
    "주택임대차 표준계약서": ("주택임대차 표준계약서",),
    "상가건물 표준임대차계약서": ("상가건물_임대차_표준계약서",),
}

SOURCE_FILE_PATTERN = re.compile(
    r"[\w가-힣ㄱ-ㅎㅏ-ㅣ·ㆍ+()[\]_,.\s-]+\.(?:pdf|hwpx|hwp|txt)"
)

DOCUMENT_SUMMARY_TERMS: frozenset[str] = frozenset({
    "주요 내용", "핵심 내용", "어떤 대상", "어떤 업무", "안내서", "목적",
})

CONTRACT_TERMS: frozenset[str] = frozenset({
    "계약서", "임대인", "임차인", "보증금", "원상회복", "임대차", "수선", "해지",
})

CASE_TERMS: frozenset[str] = frozenset({
    "판례", "판결", "결정", "사건번호", "대법원", "헌재", "헌법재판소", "lbox",
})

SOURCE_NAME_MIN_LENGTH = 8

GENERIC_DOCUMENT_TOKENS: frozenset[str] = frozenset({
    "안내서", "가이드라인", "매뉴얼", "지침",
    "법률", "시행령", "시행규칙", "고시", "규정",
    "기준", "표준",
    "pdf", "hwpx", "hwp", "txt",
})


def normalize_source_text(text: str) -> str:
    text = re.sub(r"^(?:documents/raw/)?", "", text)
    text = re.sub(r"\.(?:pdf|hwpx|hwp|txt)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\s\[\]★_+().,·ㆍ-]+", "", text)
    return text.casefold()


def source_name_overlap(
    normalized_query: str, name: str, source_path: str
) -> bool:
    raw_text = f"{name} {source_path}"
    tokens = [
        normalize_source_text(token)
        for token in TOKEN_PATTERN.findall(raw_text)
        if token not in {"raw", "documents", "원본게시용"}
    ]
    tokens = [t for t in tokens if len(t) >= 3]
    specific_tokens = [t for t in tokens if t not in GENERIC_DOCUMENT_TOKENS]
    if specific_tokens:
        if any(token in normalized_query for token in specific_tokens):
            return True
        if any(len(t) > 2 and t[:-1] in normalized_query for t in specific_tokens):
            return True
        return sum(1 for token in specific_tokens if token in normalized_query) >= 2
    return any(token in normalized_query for token in tokens)


def is_document_summary_query(
    query: str, source_paths: list[str] | None = None
) -> bool:
    if "안내서" in query:
        return True
    if not any(term in query for term in DOCUMENT_SUMMARY_TERMS):
        return False
    if source_paths:
        return any(
            document_type_from_source_path(sp) in {"qna", "guidance"}
            for sp in source_paths
        )
    return True


def is_contract_query(
    query: str, source_paths: list[str]
) -> bool:
    if any(term in query for term in CONTRACT_TERMS):
        return True
    if source_paths:
        return any(
            document_type_from_source_path(sp) == "contract"
            for sp in source_paths
        )
    return False


def is_case_query(
    query: str, source_paths: list[str]
) -> bool:
    if any(term in query for term in CASE_TERMS):
        return True
    if source_paths:
        return any(
            document_type_from_source_path(sp) in {"precedent", "constitutional_decision"}
            for sp in source_paths
        )
    return False


def explicit_source_terms(query: str) -> list[str]:
    terms = []
    for match in SOURCE_FILE_PATTERN.findall(query):
        cleaned = match.strip(' "\'`.,')
        if not cleaned:
            continue
        terms.append(cleaned)
        terms.append(re.sub(r"\.(?:pdf|hwpx|hwp|txt)$", "", cleaned))
    return list(dict.fromkeys(terms))


class SourceMatcher:
    def __init__(self, run_read):
        self._run_read = run_read
        self._document_records = None
        self._alias_index = None

    def invalidate(self) -> None:
        self._document_records = None
        self._alias_index = None

    def constraints(self, query: str) -> list[str]:
        aliases = [alias for alias in SOURCE_ALIASES if alias in query]
        aliases = [
            alias for alias in aliases
            if not any(alias != other and alias in other for other in aliases)
        ]
        if "상가임대차법 시행령" in aliases:
            aliases = ["상가임대차법 시행령"]
        elif "상가건물 임대차보호법 시행령" in aliases:
            aliases = ["상가건물 임대차보호법 시행령"]
        if "표준 개인정보 보호지침" in aliases:
            aliases = ["표준 개인정보 보호지침"]
        elif "개인정보 보호법 시행령" in aliases:
            aliases = [
                alias for alias in aliases if alias != "개인정보 보호법"
            ]
        source_terms = explicit_source_terms(query)
        case_terms = re.findall(
            r"\d{4}[가-힣]{1,3}\d+(?:,\s*\d+)?"
            r"|lbox_precedent_\d+"
            r"|판례_[0-9가-힣, ]+"
            r"|헌재결정례_[0-9가-힣]+",
            query,
        )
        allow_named_overlap = (
            bool(source_terms)
            or bool(case_terms)
            or any(term in query for term in DOCUMENT_SUMMARY_TERMS | CASE_TERMS | CONTRACT_TERMS)
        )
        records = self._documents()
        normalized_query = normalize_source_text(query)
        embedded_names = []
        for record in records:
            source_path = record.get("source_path") or ""
            names = {
                normalize_source_text(record.get("title") or ""),
                normalize_source_text(record.get("name") or ""),
                normalize_source_text(Path(source_path).stem),
            }
            longest_match = max(
                (
                    name for name in names
                    if len(name) >= SOURCE_NAME_MIN_LENGTH and name in normalized_query
                ),
                key=len,
                default="",
            )
            if longest_match:
                embedded_names.append((len(longest_match), source_path))
        if embedded_names:
            best_length = max(length for length, _path in embedded_names)
            return list(dict.fromkeys(
                path for length, path in embedded_names if length == best_length
            ))
        if not aliases and not source_terms and not case_terms and not allow_named_overlap:
            return []
        exact_matches = self._exact_source_matches(records, source_terms)
        alias_matches = self._alias_matches(aliases)
        case_content_matches = self._case_content_source_matches(case_terms)
        direct_matches = list(dict.fromkeys(
            exact_matches + alias_matches + case_content_matches
        ))
        if direct_matches:
            return direct_matches
        named_matches = (
            self._named_document_matches(records, normalized_query)
            if allow_named_overlap
            else []
        )
        return named_matches or []

    def category_constraints(self, category: str | None) -> list[str]:
        """Return indexed source paths in one requested document category."""
        if not category or category in {"all", "auto"}:
            return []
        return [
            record["source_path"]
            for record in self._documents()
            if (
                record.get("document_type")
                or document_type_from_source_path(record.get("source_path") or "")
            ) == category
        ]

    def _documents(self) -> list:
        if self._document_records is None:
            self._document_records = self._run_read(
                """
                MATCH (document:Document)
                RETURN document.source_path AS source_path,
                    document.name AS name,
                    document.title AS title,
                    document.document_type AS document_type,
                    document.logical_id AS logical_id,
                    document.version_label AS version_label
                """
            )
        return self._document_records

    def _aliases(self) -> dict[str, list[str]]:
        if self._alias_index is None:
            index = {alias: [] for alias in SOURCE_ALIASES}
            for record in self._documents():
                name = record.get("title") or record.get("name") or ""
                source_path = record.get("source_path") or ""
                for alias, targets in SOURCE_ALIASES.items():
                    if any(target in name or target in source_path for target in targets):
                        index[alias].append(source_path)
            self._alias_index = {
                alias: list(dict.fromkeys(source_paths))
                for alias, source_paths in index.items()
            }
        return self._alias_index

    def _alias_matches(self, aliases: list[str]) -> list[str]:
        if not aliases:
            return []
        matches = []
        alias_index = self._aliases()
        for alias in aliases:
            matches.extend(alias_index.get(alias, []))
        return list(dict.fromkeys(matches))

    def _exact_source_matches(
        self, records: list, terms: list[str]
    ) -> list[str]:
        normalized_terms = [
            normalize_source_text(term)
            for term in terms
            if len(normalize_source_text(term)) >= SOURCE_NAME_MIN_LENGTH
        ]
        if not normalized_terms:
            return []
        matches = []
        for record in records:
            name = record.get("name") or ""
            source_path = record.get("source_path") or ""
            normalized_sources = [
                normalize_source_text(record.get("title") or ""),
                normalize_source_text(name),
                normalize_source_text(Path(source_path).name),
                normalize_source_text(source_path),
            ]
            if any(
                source == term or source.endswith(term) or term.endswith(source)
                for term in normalized_terms
                for source in normalized_sources
                if source
            ):
                matches.append(source_path)
        return list(dict.fromkeys(matches))

    def _case_content_source_matches(self, case_terms: list[str]) -> list[str]:
        if not case_terms:
            return []
        records = self._run_read(
            """
            MATCH (chunk:Chunk)
            WHERE any(term IN $case_terms WHERE chunk.content CONTAINS term)
            RETURN DISTINCT chunk.source_path AS source_path
            ORDER BY source_path
            LIMIT 5
            """,
            case_terms=case_terms,
        )
        return [record["source_path"] for record in records]

    def _named_document_matches(
        self, records: list, normalized_query: str
    ) -> list[str]:
        matches = []
        for record in records:
            name = record.get("name") or ""
            source_path = record.get("source_path") or ""
            if source_name_overlap(normalized_query, name, source_path):
                matches.append(source_path)
        return list(dict.fromkeys(matches))
