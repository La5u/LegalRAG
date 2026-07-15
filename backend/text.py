import re

from kiwipiepy import Kiwi

from backend.constants import TOKEN_PATTERN


def deduplicate(items):
    return list(dict.fromkeys(items))


SENTENCE_PATTERN = re.compile(
    r"[^.!?。！？\n]+?(?:[.!?。！？]|(?:습니다|합니다|된다|한다|없다|있다|이다|니다|까|함|음|됨|다|요|죠)(?=\s+|$))"
)

PARTICLE_SUFFIX_TAG_PREFIXES = ("E", "J")
PARTICLE_SUFFIX_FORMS = {"상"}

_kiwi: Kiwi | None = None


def get_kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


def extract_tokens(query: str) -> list[str]:
    return deduplicate(TOKEN_PATTERN.findall(query))


def query_terms(
    query: str,
    *,
    initial_terms: list[str] | None = None,
    ignored_terms: set[str] | frozenset[str] = frozenset(),
    max_terms: int | None = None,
    min_length: int = 1,
    strip_particles: bool = False,
) -> list[str]:
    terms = []
    for term in (initial_terms or []) + extract_tokens(query):
        cleaned = strip_korean_particle(term) if strip_particles else term
        if len(cleaned) < min_length or cleaned.casefold() in ignored_terms:
            continue
        terms.append(cleaned)
    terms = deduplicate(terms)
    return terms[:max_terms] if max_terms is not None else terms


def strip_korean_particle(term: str) -> str:
    if not term:
        return term
    analyzed = get_kiwi().analyze(term)
    if analyzed:
        tokens = list(analyzed[0][0])
        while tokens and (
            tokens[-1].tag.startswith(PARTICLE_SUFFIX_TAG_PREFIXES)
            or tokens[-1].form in PARTICLE_SUFFIX_FORMS
        ):
            tokens.pop()
        if any(token.tag.startswith("N") or token.tag in {"SL", "SN"} for token in tokens):
            return combine_tokens(term, tokens)
    return term


def combine_tokens(original: str, tokens: list) -> str:
    """Rejoin token forms while preserving original whitespace between tokens."""
    if not tokens:
        return ""
    parts = []
    for token in tokens:
        start = token.start
        if parts and start > 0:
            gap = original[parts[-1][1]:start]
            if gap:
                parts.append((gap, start + len(gap)))
        parts.append((token.form, start + token.len))
    return "".join(p[0] for p in parts)


def split_sentences(text: str) -> list[str]:
    return [
        match.group(0).strip()
        for match in SENTENCE_PATTERN.finditer(text)
        if match.group(0).strip()
    ]
