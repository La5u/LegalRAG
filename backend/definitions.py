from dataclasses import dataclass
import re

from backend.document import Document


DEFINITION_PATTERN = re.compile(
    r"(?:[“\"「](?P<quoted_term>[^”\"」]{1,60})[”\"」]\s*"
    r"(?:이란|란|이라\s*함은|라\s*함은)"
    r"|(?P<plain_term>[A-Z][A-Za-z][A-Za-z \-]{0,58})\s+"
    r"(?:means|shall mean|shall be understood to mean)\b"
    r"|(?P<consists_term>[A-Z][A-Za-z][A-Za-z \-]{0,58}),\s+"
    r"shall consist only in\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Definition:
    term: str
    text: str
    chunk_number: int
    source_path: str
    document_name: str
    article_number: str | None
    article_title: str | None


def extract_definitions(chunks: list[Document]) -> list[Definition]:
    definitions = []

    for chunk in chunks:
        article_title = chunk.meta.get("article_title") or ""
        has_definition_title = article_title.casefold() in {"정의", "definitions"}
        has_english_definition = re.search(
            r"\b(means|shall mean|shall be understood to mean|shall consist only in)\b",
            chunk.content,
            re.IGNORECASE,
        )
        if not has_definition_title and not has_english_definition:
            continue

        matches = list(DEFINITION_PATTERN.finditer(chunk.content))
        for index, match in enumerate(matches):
            term = (
                match.group("quoted_term")
                or match.group("plain_term")
                or match.group("consists_term")
            )
            next_match_start = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(chunk.content)
            )
            definitions.append(
                Definition(
                    term=term.strip(),
                    text=chunk.content[match.start():next_match_start].strip(),
                    chunk_number=chunk.meta["chunk_number"],
                    source_path=chunk.meta.get("source_path", ""),
                    document_name=chunk.meta.get("document_name", ""),
                    article_number=chunk.meta.get("article_number"),
                    article_title=chunk.meta.get("article_title"),
                )
            )

    return definitions
