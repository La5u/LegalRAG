import re

from backend.ingest import Document
from backend.text import split_sentences


CITATION_PATTERN = re.compile(r"제\d+조(?:의\d+)?|제\d+항|제\d+호|[가-하]목")
EXCEPTION_PATTERN = re.compile(r"(다만|예외로 한다|제외한다|그러하지 아니하다)")
CONNECTION_RELATIONS = ["CITES", "EXCEPTION_TO"]


def article_concept(chunk: Document) -> str | None:
    article_number = chunk.meta.get("article_number")
    if not article_number:
        return None
    return article_number


def extract_connections(chunks: list[Document]) -> list[dict]:
    connections = []
    seen = set()

    for chunk in chunks:
        source = article_concept(chunk)
        if not source:
            continue

        for sentence in split_sentences(chunk.content):
            for match in CITATION_PATTERN.finditer(sentence):
                target = match.group(0)
                if target == source:
                    continue
                key = ("CITES", source, target, chunk.meta["chunk_number"], sentence)
                if key not in seen:
                    seen.add(key)
                    connections.append(connection_row(chunk, "CITES", source, target, sentence))

            if EXCEPTION_PATTERN.search(sentence):
                target = "일반규칙"
                citations = CITATION_PATTERN.findall(sentence)
                if citations:
                    target = citations[-1]
                key = ("EXCEPTION_TO", source, target, chunk.meta["chunk_number"], sentence)
                if key not in seen:
                    seen.add(key)
                    connections.append(
                        connection_row(chunk, "EXCEPTION_TO", source, target, sentence)
                    )

    return connections


def connection_row(
    chunk: Document,
    relation: str,
    source: str,
    target: str,
    sentence: str,
) -> dict:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "sentence": sentence,
        "source_path": chunk.meta["source_path"],
        "document_name": chunk.meta["document_name"],
        "chunk_id": chunk.meta["id"],
        "chunk_number": chunk.meta["chunk_number"],
        "legal_unit_id": chunk.meta.get("legal_unit_id"),
        "article_number": chunk.meta.get("article_number"),
    }
