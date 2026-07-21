import re


ARTICLE_PATTERN = re.compile(r"제\d+조(?:의\d+)?")
CITATION_PATTERN = re.compile(
    r"제\d+조(?:의\d+)?(?:제\d+(?:항|호|목)|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])*"
)
FULL_CITATION_PATTERN = re.compile(
    r"(?:[「\"](?P<law_name>[^」\"]+)[」\"]\s*)?"
    r"(?P<article>제\d+조(?:의\d+)?)"
    r"(?P<clause>제\d+(?:항|호|목)|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])?"
)
LOCAL_CLAUSE_PATTERN = re.compile(
    r"(?P<clause>제\d+(?:항|호|목)|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|[가-하]목)"
)
CIRCLED_CLAUSE_MARKS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9·ㆍ]{2,}")
CONCEPT_STOP_TERMS = {
    "일반규칙",
    "개인",
    "것",
    "기관",
    "결정",
    "때",
    "법",
    "내용",
    "조치",
    "사유",
    "결과",
    "사실",
    "사항",
    "경우",
    "일부",
    "방법",
    "기준",
    "자료",
    "업무",
    "포함",
    "해당",
    "이",
    "자",
    "정보",
    "처분",
    "항",
    "호",
    "수",
    "제1항",
    "제2항",
    "제3항",
    "제4항",
    "제5항",
    "제1호",
    "제2호",
    "제3호",
    "제4호",
    "제5호",
}
CONCEPT_STOP_TERMS_LIST = sorted(CONCEPT_STOP_TERMS)
GRAPH_FACT_ID_PREFIX = "graph-fact::"
QUERY_STOP_TERMS = {
    "what",
    "which",
    "where",
    "when",
    "shall",
    "does",
    "against",
    "defined",
    "united",
    "states",
    "article",
    "section",
    "에서",
    "으로",
    "하고",
    "하며",
    "무엇",
    "어느",
    "몇조",
    "몇",
    "조에",
    "규정",
    "규정되어",
    "있나요",
    "인가요",
    "설명하세요",
    "따르면",
    "없이",
    "수",
    "가능",
}
STRUCTURE_PATTERNS = [
    ("chapter", re.compile(r"^(?P<number>제\d+장)\s*(?P<title>.*)")),
    (
        "chapter",
        re.compile(
            r"^(?P<number>Article\s+[IVXLCDM\d]+)\.?\s*(?P<title>.*)",
            re.IGNORECASE,
        ),
    ),
    ("chapter", re.compile(r"^(?P<number>ARTICLE\s+[A-Z]+)\.?\s*(?P<title>.*)")),
    ("section", re.compile(r"^(?P<number>제\d+절)\s*(?P<title>.*)")),
    ("article", re.compile(r"^(?P<number>제\d+조(?:의\d+)?)(?:\((?P<title>[^)]*)\))?")),
    (
        "article",
        re.compile(
            r"^(?P<number>Section\s+\d+)\.?\s*(?P<title>.*)",
            re.IGNORECASE,
        ),
    ),
    ("clause", re.compile(r"^(?P<number>[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])\s*(?P<title>.*)")),
    ("item", re.compile(r"^(?P<number>\d+\.)\s*(?P<title>.*)")),
    ("subitem", re.compile(r"^(?P<number>[가-하]\.)\s*(?P<title>.*)")),
]
STRUCTURE_KINDS = list(dict.fromkeys(kind for kind, _pattern in STRUCTURE_PATTERNS))
