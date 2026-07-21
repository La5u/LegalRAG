import re

from backend.constants import FULL_CITATION_PATTERN, LOCAL_CLAUSE_PATTERN
from backend.document import Document
from backend.text import split_sentences


EXCEPTION_PATTERN = re.compile(r"(다만|예외로 한다|제외한다|그러하지 아니하다)")
CONNECTION_RELATIONS = ["CITES", "EXCEPTION_TO"]


def extract_connections(chunks: list[Document]) -> list[dict]:
    connections = []
    seen = set()

    for chunk in chunks:
        source = chunk.meta.get("article_number")
        if not source:
            continue

        for sentence in split_sentences(chunk.content):
            for citation in citation_rows(chunk, sentence):
                key = (
                    "CITES",
                    citation["article"],
                    citation.get("clause"),
                    citation.get("law_name"),
                    chunk.meta["chunk_number"],
                    citation["raw_text"],
                )
                if key in seen:
                    continue
                seen.add(key)
                connections.append(citation)

            exception_match = EXCEPTION_PATTERN.search(sentence)
            if exception_match:
                target = "일반규칙"
                citations = list(FULL_CITATION_PATTERN.finditer(sentence))
                if citations:
                    target = citations[-1].group("article")
                key = ("EXCEPTION_TO", source, target, chunk.meta["chunk_number"], sentence)
                if key not in seen:
                    seen.add(key)
                    connections.append(
                        connection_row(
                            chunk,
                            "EXCEPTION_TO",
                            source,
                            target,
                            sentence,
                            exception_match.group(0),
                        )
                    )

    return connections


def citation_rows(chunk: Document, sentence: str) -> list[dict]:
    rows = []
    matched_spans = []
    source_article = chunk.meta.get("article_number")
    for match in FULL_CITATION_PATTERN.finditer(sentence):
        article = match.group("article")
        matched_spans.append(match.span())
        rows.append(
            citation_row(
                chunk=chunk,
                sentence=sentence,
                raw_text=match.group(0),
                law_name=match.group("law_name"),
                article=article,
                clause=match.group("clause"),
            )
        )

    if source_article:
        for match in LOCAL_CLAUSE_PATTERN.finditer(sentence):
            if any(start <= match.start() < end for start, end in matched_spans):
                continue
            rows.append(
                citation_row(
                    chunk=chunk,
                    sentence=sentence,
                    raw_text=match.group("clause"),
                    law_name=None,
                    article=source_article,
                    clause=match.group("clause"),
                )
            )
    return rows


def citation_row(
    chunk: Document,
    sentence: str,
    raw_text: str,
    law_name: str | None,
    article: str,
    clause: str | None,
) -> dict:
    row = connection_row(
        chunk,
        "CITES",
        chunk.meta.get("article_number", ""),
        article,
        sentence,
        raw_text,
    )
    row.update(
        {
            "raw_text": raw_text,
            "law_name": law_name,
            "article": article,
            "clause": clause,
        }
    )
    return row


def connection_row(
    chunk: Document,
    relation: str,
    source: str,
    target: str,
    sentence: str,
    relation_phrase: str | None = None,
) -> dict:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "relation_phrase": relation_phrase,
        "sentence": sentence,
        "source_path": chunk.meta["source_path"],
        "document_name": chunk.meta["document_name"],
        "chunk_id": chunk.meta["id"],
        "chunk_number": chunk.meta["chunk_number"],
        "legal_unit_id": chunk.meta.get("legal_unit_id"),
        "article_number": chunk.meta.get("article_number"),
    }
