import re


SENTENCE_PATTERN = re.compile(
    r"[^.!?。！？\n]+?(?:[.!?。！？]|(?:습니다|합니다|된다|한다|없다|있다|이다|니다|까|함|음|됨|다|요|죠)(?=\s+|$)|$)"
)


def split_sentences(text: str) -> list[str]:
    return [
        match.group(0).strip()
        for match in SENTENCE_PATTERN.finditer(text)
        if match.group(0).strip()
    ]
