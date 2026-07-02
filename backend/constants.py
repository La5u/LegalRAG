import re


ARTICLE_PATTERN = re.compile(r"제\d+조(?:의\d+)?")
CITATION_PATTERN = re.compile(
    r"제\d+조(?:의\d+)?(?:제\d+(?:항|호|목)|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])?"
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
    "때",
    "법",
    "내용",
    "조치",
    "사유",
    "결과",
    "사실",
    "사항",
    "경우",
    "방법",
    "기준",
    "자료",
    "업무",
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
