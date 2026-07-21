import re

from backend.constants import ARTICLE_PATTERN
from backend.document import Document
from backend.text import split_sentences


EXCEPTION_PATTERN = re.compile(r"(다만|예외로 한다|제외한다|그러하지 아니하다)")


def extract_exceptions(chunks: list[Document]) -> list[dict]:
    exceptions = []
    seen = set()

    for chunk in chunks:
        source = chunk.meta.get("article_number")
        if not source:
            continue

        for sentence in split_sentences(chunk.content):
            exception_match = EXCEPTION_PATTERN.search(sentence)
            if exception_match:
                target = "일반규칙"
                citations = list(ARTICLE_PATTERN.finditer(sentence))
                if citations:
                    target = citations[-1].group(0)
                key = ("EXCEPTION_TO", source, target, chunk.meta["chunk_number"], sentence)
                if key not in seen:
                    seen.add(key)
                    exceptions.append({
                        "source": source,
                        "target": target,
                        "relation": "EXCEPTION_TO",
                        "relation_phrase": exception_match.group(0),
                        "sentence": sentence,
                        "source_path": chunk.meta["source_path"],
                        "document_name": chunk.meta["document_name"],
                        "chunk_id": chunk.meta["id"],
                        "chunk_number": chunk.meta["chunk_number"],
                    })

    return exceptions
