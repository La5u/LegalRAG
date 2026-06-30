import re

from kiwipiepy import Kiwi

from backend.ingest import Document
from backend.text import split_sentences


LEGAL_RELATION_PATTERNS = [
    ("MUST_NOTIFY", re.compile(r"(통지하여야 한다|알려야 한다)")),
    ("MAY_REQUEST", re.compile(r"(요구할 수 있다|청구할 수 있다)")),
    ("PROHIBITS", re.compile(r"(금지한다|해서는 아니 된다)")),
    ("HAS_DUTY", re.compile(r"(하여야 한다|해야 한다)")),
    ("MAY", re.compile(r"할 수 있다")),
]
LEGAL_RELATIONS = [relation for relation, _pattern in LEGAL_RELATION_PATTERNS]
SUBJECT_PARTICLES = {"은", "는", "이", "가", "께서는"}
RECIPIENT_PARTICLES = {"에게", "께", "한테", "에"}
OBJECT_PARTICLES = {"을", "를"}


_kiwi: Kiwi | None = None


def get_kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


def noun_phrases(sentence: str) -> list[dict]:
    analyzed = get_kiwi().analyze(sentence)
    if not analyzed:
        return []

    tokens = analyzed[0][0]
    phrases = []
    index = 0
    while index < len(tokens):
        tag = tokens[index].tag
        if not (tag.startswith("N") or tag in {"SL", "SN"}):
            index += 1
            continue

        forms = []
        while index < len(tokens):
            tag = tokens[index].tag
            if not (tag.startswith("N") or tag in {"SL", "SN", "XSN"}):
                break
            forms.append(tokens[index].form)
            index += 1

        particle = None
        if index < len(tokens) and tokens[index].tag.startswith("J"):
            particle = tokens[index].form

        phrase = "".join(forms).strip()
        if phrase:
            phrases.append({"text": phrase, "particle": particle})

    return phrases


def pick_subject(phrases: list[dict]) -> str | None:
    for phrase in phrases:
        if phrase["particle"] in SUBJECT_PARTICLES:
            return phrase["text"]
    return phrases[0]["text"] if phrases else None


def pick_object(phrases: list[dict], relation: str) -> str | None:
    if relation in {"MUST_NOTIFY", "MAY_REQUEST"}:
        for phrase in reversed(phrases):
            if phrase["particle"] in RECIPIENT_PARTICLES:
                return phrase["text"]

    for phrase in reversed(phrases):
        if phrase["particle"] in OBJECT_PARTICLES:
            return phrase["text"]

    return phrases[-1]["text"] if phrases else None


def extract_triples(chunks: list[Document]) -> list[dict]:
    triples = []
    seen = set()

    for chunk in chunks:
        for sentence in split_sentences(chunk.content):
            for relation, pattern in LEGAL_RELATION_PATTERNS:
                match = pattern.search(sentence)
                if not match:
                    continue

                phrases = noun_phrases(sentence[:match.start()])
                subject = pick_subject(phrases)
                object_ = pick_object(phrases, relation)
                if not subject or not object_ or subject == object_:
                    continue

                key = (
                    relation,
                    subject,
                    object_,
                    chunk.meta["chunk_number"],
                    sentence,
                )
                if key in seen:
                    continue
                seen.add(key)
                triples.append(
                    {
                        "subject": subject,
                        "object": object_,
                        "relation": relation,
                        "sentence": sentence,
                        "source_path": chunk.meta["source_path"],
                        "document_name": chunk.meta["document_name"],
                        "chunk_id": chunk.meta["id"],
                        "chunk_number": chunk.meta["chunk_number"],
                        "legal_unit_id": chunk.meta.get("legal_unit_id"),
                        "article_number": chunk.meta.get("article_number"),
                    }
                )
                break

    return triples
