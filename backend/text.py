import re


SENTENCE_PATTERN = re.compile(r"[^.!?。！？\n]+(?:[.!?。！？]|$)")


def split_sentences(text: str) -> list[str]:
    return [
        match.group(0).strip()
        for match in SENTENCE_PATTERN.finditer(text)
        if match.group(0).strip()
    ]
