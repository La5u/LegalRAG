import re


SENTENCE_PATTERN = re.compile(
    r"[^.!?。！？\n]+?(?:[.!?。！？]|(?:습니다|합니다|된다|한다|없다|있다|이다|니다|까|함|음|됨|다|요|죠)(?=\s+|$))"
)

KOREAN_PARTICLE_SUFFIXES = (
    "되나요",
    "인가요",
    "법상",
    "위한",
    "인한",
    "하면",
    "에서",
    "으로",
    "부터",
    "까지",
    "에게",
    "에는",
    "하는",
    "하여",
    "하고",
    "해야",
    "할",
    "한",
    "은",
    "는",
    "이",
    "가",
    "에",
    "을",
    "를",
    "의",
    "로",
)


def extract_tokens(query: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"[가-힣A-Za-z0-9·ㆍ]{2,}", query)))


def strip_korean_particle(term: str) -> str:
    for suffix in KOREAN_PARTICLE_SUFFIXES:
        if term.endswith(suffix) and len(term) > len(suffix):
            return term[: -len(suffix)]
    return term


def split_sentences(text: str) -> list[str]:
    return [
        match.group(0).strip()
        for match in SENTENCE_PATTERN.finditer(text)
        if match.group(0).strip()
    ]
