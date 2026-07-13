import re
from dataclasses import dataclass, field, replace

from backend.config import (
    MAX_EVIDENCE_CHUNKS,
    MAX_CONTEXT_WORDS,
    NEO4J_VECTOR_INDEX, EMBEDDING_QUERY_PREFIX,
    normalize_retrieval_strategy,
)
from backend.constants import CIRCLED_CLAUSE_MARKS, GRAPH_FACT_ID_PREFIX, QUERY_STOP_TERMS
from backend.document import Document
from backend.document_types import document_type_from_source_path
from backend.graph_retrieve import GraphFact, graph_facts
from backend.reasoning import build_symbolic_trace
from backend.records import chunk_return_fields, document_from_record
from backend.routing import QueryRoute, classify_query
from backend.schema import vector_index_exists
from backend.sources import (
    CONTRACT_TERMS,
    SOURCE_ALIASES,
    is_case_query,
    is_contract_query,
    is_document_summary_query,
)
from backend.text import query_terms


LEGAL_UNIT_MATCH_LIMIT = 8
LEGAL_UNIT_BROAD_MATCH_LIMIT = 4
LEGAL_UNIT_EXPANSION_LIMIT = 6
SPARSE_CHUNK_MATCH_LIMIT = 8
SOURCE_VECTOR_OVERFETCH_FACTOR = 50
SOURCE_MATCH_SCORES = {
    "article": 100,
    "start_strong": 80,
    "start_weak": 20,
    "content": 8,
    "title": 20,
    "number": 15,
}
LEGAL_UNIT_MATCH_SCORES = {
    "article": 100,
    "title": 20,
    "number": 15,
    "document": 4,
    "content": 1,
}
START_OF_DOCUMENT_TERMS = ("첫 부분", "첫부분", "앞부분", "서두", "목적", "적용 대상")
GENERIC_PAIR_TERMS = {
    "개인정보",
    "개인정보처리자",
    "정보주체",
    "필요",
    "경우",
    "어떤",
    "무엇",
    "꼭",
}
RANK_FUSION_OFFSET = 60
RETRIEVAL_CHANNEL_WEIGHTS = {
    "exact_phrase": 1.25,
    "vector": 1.0,
    "summary": 0.9,
    "contract": 0.8,
    "case": 0.8,
    # A source-scoped provision match is more precise than a corpus-wide term
    # match, but only the first two candidates receive this channel.
    "provision_focus": 0.9,
    "legal_unit": 0.55,
    # Within an explicitly named document, issue terms are reliable evidence;
    # the source constraint prevents generic words from matching the corpus.
    "source_terms": 0.9,
    "sparse_terms": 0.25,
    "authority": 0.15,
}
GRAPH_RELATIONS_BY_QUERY_TYPE = {
    "article_lookup": set(),
    "definition": {"DEFINED_AS"},
    "obligation": {"HAS_DUTY", "MUST_NOTIFY"},
    "prohibition": {"PROHIBITS"},
    # An EXCEPTION_TO edge is an extraction candidate rather than a complete
    # legal exception.  The source passage itself is safer answer context.
    "exception": set(),
    "citation": {"HAS_HOLDING"},
    "condition": {
        "MAY",
        "MAY_REQUEST",
        "HAS_DUTY",
        "MUST_NOTIFY",
        "PROHIBITS",
        "EXCEPTION_TO",
    },
}


LEGAL_UNIT_QUERY_PATTERN = re.compile(
    r"제\d+조(?:의\d+)?|제\d+항|제\d+호|[가-하]목|"
    r"\bArticle\s+[IVXLCDM\d]+\b|\bSection\s+\d+\b",
    re.IGNORECASE,
)
LEGAL_UNIT_QUERY_TERMS = (
    "조문",
    "몇 조",
    "어느 조",
    "어느 조문",
    "article",
    "section",
)
ENUMERATED_LEAD_IN_PATTERN = re.compile(
    r"다음\s+각\s+(?:호|목)|following\s+(?:items|subparagraphs)",
    re.IGNORECASE,
)
SOURCE_QUESTION_PREFIX = re.compile(r"^.+?(?:에\s*따르면|에서)\s*,?\s*")


def source_question_focus(query: str) -> str:
    """Remove an explicit document prefix from a source-scoped question."""
    return SOURCE_QUESTION_PREFIX.sub("", query, count=1)


@dataclass(frozen=True)
class RetrievalContext:
    query: str
    route: QueryRoute
    source_paths: list[str]
    is_summary_query: bool
    is_contract_query: bool
    is_case_query: bool
    source_constraint_mismatch: bool = False


@dataclass(frozen=True)
class RetrievalResult:
    """Evidence and mode-specific output from one retrieval run."""

    strategy: str
    context: RetrievalContext
    evidence: list[Document]
    graph_facts: list[GraphFact] = field(default_factory=list)
    symbolic_trace: dict | None = None


class RetrievalUnavailable(RuntimeError):
    """Raised when retrieval cannot run because the index is not ready."""


def should_expand_legal_units(query: str) -> bool:
    normalized_query = query.casefold()
    return bool(LEGAL_UNIT_QUERY_PATTERN.search(query)) or any(
        term in normalized_query for term in LEGAL_UNIT_QUERY_TERMS
    )


def quoted_terms(query: str) -> list[str]:
    return [
        term.strip()
        for term in re.findall(r"[\"'“”‘’](.*?)[\"'“”‘’]", query)
        if term.strip()
    ]


def source_query_terms(query: str, route: QueryRoute) -> list[str]:
    return query_terms(
        query,
        initial_terms=route.search_terms,
        ignored_terms=set(SOURCE_ALIASES) | {"대법원", "헌재", "헌법재판소"},
        max_terms=12,
        min_length=2,
        strip_particles=True,
    )


def adjacent_query_phrases(query: str) -> list[str]:
    """Return useful adjacent surface-word pairs from the original query."""
    words = [
        word for word in re.findall(r"[가-힣A-Za-z0-9·ㆍ]+", query)
        if len(word) >= 2 and word.casefold() not in QUERY_STOP_TERMS
    ]
    return list(dict.fromkeys(
        f"{left} {right}" for left, right in zip(words, words[1:])
    ))[:10]


def retrieve_exact_phrase_matches(
    run_read, query, source_paths=None, include_adjacent=False,
) -> list[Document]:
    phrases = quoted_terms(query)
    if include_adjacent:
        phrases = list(dict.fromkeys(
            phrases + adjacent_query_phrases(query)[-6:]
        ))
    if not phrases:
        return []
    source_paths = source_paths or []
    records = run_read(
        f"""
        MATCH (chunk:Chunk)
        WHERE ($source_paths = [] OR chunk.source_path IN $source_paths)
            AND any(phrase IN $phrases WHERE chunk.content CONTAINS phrase)
        WITH chunk,
            reduce(score = 0, phrase IN $phrases |
                score + CASE WHEN chunk.content CONTAINS phrase THEN 1 ELSE 0 END
            ) AS phrase_score
        RETURN
            {chunk_return_fields("chunk", "phrase_score * 1.0")}
        ORDER BY phrase_score DESC, chunk.chunk_number
        LIMIT 8
        """,
        source_paths=source_paths,
        phrases=phrases,
    )
    documents = []
    for record in records:
        document = document_from_record(record)
        documents.append(document.with_meta(
            exact_phrase_match=True,
            matched_phrases=phrases,
        ))
    return documents


def retrieve_vector_chunks(
    run_read, embedding_model, top_k, query, source_paths=None
) -> list[Document]:
    if not vector_index_exists(run_read, NEO4J_VECTOR_INDEX):
        raise RetrievalUnavailable("No vector index exists. Run indexing first.")

    query_embedding = embedding_model.encode(
        EMBEDDING_QUERY_PREFIX + query, normalize_embeddings=True,
    ).tolist()
    records = run_read(
        f"""
        CALL db.index.vector.queryNodes(
            $index_name,
            $candidate_limit,
            $query_embedding
        )
        YIELD node, score
        WHERE $source_paths = [] OR node.source_path IN $source_paths
        RETURN
            {chunk_return_fields("node", "score")}
        ORDER BY score DESC
        """,
        index_name=NEO4J_VECTOR_INDEX,
        # Vector indexes cannot filter before searching. Over-fetching leaves
        # enough candidates after a source or category constraint is applied.
        candidate_limit=(
            top_k * SOURCE_VECTOR_OVERFETCH_FACTOR if source_paths else top_k
        ),
        query_embedding=query_embedding,
        source_paths=source_paths or [],
    )
    return [document_from_record(record) for record in records]


def retrieve_sparse_chunks(run_read, route) -> list[Document]:
    terms = query_terms(
        "",
        initial_terms=route.search_terms,
        ignored_terms=set(SOURCE_ALIASES),
        min_length=2,
        strip_particles=True,
    )
    if not terms:
        return []

    records = run_read(
        f"""
        MATCH (chunk:Chunk)
        WITH chunk,
            reduce(score = 0, term IN $terms |
                score + CASE WHEN chunk.content CONTAINS term THEN 1 ELSE 0 END
            ) AS sparse_score
        WHERE sparse_score > 0
        RETURN
            {chunk_return_fields("chunk", "sparse_score * 1.0")}
        ORDER BY sparse_score DESC, chunk.chunk_number
        LIMIT $limit
        """,
        terms=terms,
        limit=SPARSE_CHUNK_MATCH_LIMIT,
    )
    return [document_from_record(record) for record in records]


def retrieve_legal_unit_matches(
    run_read, route, source_paths=None
) -> list[Document]:
    if not route.search_terms and not route.article_numbers:
        return []

    terms = query_terms(
        "",
        initial_terms=route.search_terms,
        min_length=2,
        strip_particles=True,
    )
    term_pairs = [
        {"left": terms[index], "right": terms[index + 1]}
        for index in range(len(terms) - 1)
        if terms[index] not in GENERIC_PAIR_TERMS
        and terms[index + 1] not in GENERIC_PAIR_TERMS
    ]
    records = run_read(
        f"""
        MATCH (article:LegalUnit {{kind: "article"}})
        WHERE $source_paths = [] OR article.source_path IN $source_paths
        OPTIONAL MATCH (article)-[:CONTAINS*0..3]->(unit:LegalUnit)
        OPTIONAL MATCH (unit)-[:SUPPORTED_BY]->(chunk:Chunk)
        WITH article, collect(DISTINCT chunk) AS chunks
        WITH article, chunks,
            CASE
                WHEN article.number IN $article_numbers THEN $legal_unit_article_match_score
                ELSE 0
            END
            + reduce(score = 0, term IN $terms |
                score
                + CASE WHEN article.title IS NOT NULL AND article.title CONTAINS term THEN $legal_unit_title_match_score ELSE 0 END
                + CASE WHEN article.number IS NOT NULL AND article.number CONTAINS term THEN $legal_unit_number_match_score ELSE 0 END
                + CASE WHEN article.document_name IS NOT NULL AND article.document_name CONTAINS term THEN $legal_unit_document_match_score ELSE 0 END
                + CASE WHEN any(chunk IN chunks WHERE chunk.content CONTAINS term) THEN $legal_unit_content_match_score ELSE 0 END
            ) AS lexical_score
        WHERE lexical_score >= $minimum_score
        ORDER BY lexical_score DESC, article.order
        LIMIT $article_limit
        UNWIND chunks AS chunk
        WITH article, chunk, lexical_score,
            reduce(score = 0, term IN $terms |
                score + CASE WHEN chunk.content CONTAINS term THEN 1 ELSE 0 END
            ) AS chunk_score,
            reduce(score = 0, pair IN $term_pairs |
                score
                + CASE
                    WHEN chunk.content CONTAINS pair.left
                        AND chunk.content CONTAINS pair.right
                    THEN 25
                    ELSE 0
                END
            ) AS chunk_pair_score
        WHERE chunk IS NOT NULL
            AND ($article_numbers <> [] OR chunk_score > 0 OR chunk_pair_score > 0)
        ORDER BY lexical_score DESC, chunk_pair_score DESC, chunk_score DESC, chunk.chunk_number
        WITH article, lexical_score,
            collect({{chunk: chunk, chunk_score: chunk_score, chunk_pair_score: chunk_pair_score}})[..$chunks_per_article]
            AS selected_chunks
        UNWIND selected_chunks AS selected_chunk
        WITH article, selected_chunk.chunk AS chunk, lexical_score,
            selected_chunk.chunk_score AS chunk_score,
            selected_chunk.chunk_pair_score AS chunk_pair_score
        RETURN DISTINCT
            {chunk_return_fields("chunk", "lexical_score / 100.0 + chunk_score / 100.0 + chunk_pair_score / 100.0")}
        ORDER BY score DESC, chunk.chunk_number
        """,
        terms=terms,
        term_pairs=term_pairs,
        article_numbers=route.article_numbers,
        source_paths=source_paths or [],
        minimum_score=20,
        article_limit=LEGAL_UNIT_MATCH_LIMIT if route.article_numbers else LEGAL_UNIT_BROAD_MATCH_LIMIT,
        chunks_per_article=4,
        legal_unit_article_match_score=LEGAL_UNIT_MATCH_SCORES["article"],
        legal_unit_title_match_score=LEGAL_UNIT_MATCH_SCORES["title"],
        legal_unit_number_match_score=LEGAL_UNIT_MATCH_SCORES["number"],
        legal_unit_document_match_score=LEGAL_UNIT_MATCH_SCORES["document"],
        legal_unit_content_match_score=LEGAL_UNIT_MATCH_SCORES["content"],
    )
    return [document_from_record(record) for record in records]


def retrieve_source_constrained_matches(run_read, query, route, source_paths) -> list[Document]:
    terms = source_query_terms(query, route)
    phrases = adjacent_query_phrases(query)
    records = run_read(
        f"""
        MATCH (chunk:Chunk)
        WHERE chunk.source_path IN $source_paths
        OPTIONAL MATCH (unit:LegalUnit)-[support:SUPPORTED_BY]->(chunk)
        OPTIONAL MATCH (article:LegalUnit {{kind: "article"}})-[:CONTAINS*0..3]->(unit)
        WITH chunk, article,
            CASE WHEN article.number IN $article_numbers THEN $source_article_match_score ELSE 0 END
            + CASE WHEN $prefer_start AND chunk.chunk_number <= 3 THEN $source_start_strong_score ELSE 0 END
            + CASE WHEN $prefer_start AND chunk.chunk_number <= 8 THEN $source_start_weak_score ELSE 0 END
            + reduce(score = 0, term IN $terms |
                score
                + CASE WHEN chunk.content CONTAINS term THEN $source_content_match_score ELSE 0 END
                + CASE WHEN article.title IS NOT NULL AND article.title CONTAINS term THEN $source_title_match_score ELSE 0 END
                + CASE WHEN article.number IS NOT NULL AND article.number CONTAINS term THEN $source_number_match_score ELSE 0 END
            )
            + reduce(score = 0, phrase IN $phrases |
                score + CASE WHEN chunk.content CONTAINS phrase THEN 24 ELSE 0 END
            )
            AS source_score
        WHERE source_score > 0
        RETURN
            {chunk_return_fields("chunk", "source_score / 100.0")}
        ORDER BY
            CASE WHEN source_score > 0 THEN 0 ELSE 1 END,
            source_score DESC,
            chunk.chunk_number
        LIMIT 12
        """,
        source_paths=source_paths,
        article_numbers=route.article_numbers,
        terms=terms,
        phrases=phrases,
        prefer_start=any(term in query for term in START_OF_DOCUMENT_TERMS),
        source_article_match_score=SOURCE_MATCH_SCORES["article"],
        source_start_strong_score=SOURCE_MATCH_SCORES["start_strong"],
        source_start_weak_score=SOURCE_MATCH_SCORES["start_weak"],
        source_content_match_score=SOURCE_MATCH_SCORES["content"],
        source_title_match_score=SOURCE_MATCH_SCORES["title"],
        source_number_match_score=SOURCE_MATCH_SCORES["number"],
    )
    return [document_from_record(record) for record in records]


def retrieve_document_summary_matches(run_read, query, source_paths) -> list[Document]:
    terms = source_query_terms(query, classify_query(query))
    records = run_read(
        f"""
        MATCH (chunk:Chunk)
        WHERE chunk.source_path IN $source_paths
        WITH chunk,
            CASE WHEN chunk.chunk_number <= 3 THEN 100 ELSE 0 END
            + CASE WHEN chunk.chunk_number <= 8 THEN 40 ELSE 0 END
            + reduce(score = 0, term IN $terms |
                score + CASE WHEN chunk.content CONTAINS term THEN 10 ELSE 0 END
            ) AS summary_score
        RETURN
            {chunk_return_fields("chunk", "summary_score / 100.0")}
        ORDER BY summary_score DESC, chunk.chunk_number
        LIMIT 10
        """,
        source_paths=source_paths,
        terms=terms,
    )
    return [document_from_record(record) for record in records]


def retrieve_contract_matches(run_read, query, source_paths) -> list[Document]:
    party_terms = [
        term
        for term in CONTRACT_TERMS
        if term in query or term in {"임대인", "임차인", "원상회복", "목적물"}
    ]
    records = run_read(
        f"""
        MATCH (chunk:Chunk)
        WHERE chunk.source_path IN $source_paths
        WITH chunk,
            reduce(score = 0, term IN $party_terms |
                score + CASE WHEN chunk.content CONTAINS term THEN 25 ELSE 0 END
            )
            + CASE WHEN chunk.content CONTAINS "의무" THEN 20 ELSE 0 END
            + CASE WHEN chunk.content CONTAINS "하여야" THEN 20 ELSE 0 END
            + CASE WHEN chunk.content CONTAINS "사용" THEN 10 ELSE 0 END
            + CASE WHEN chunk.content CONTAINS "수익" THEN 10 ELSE 0 END
            + CASE WHEN chunk.content CONTAINS "인도" THEN 15 ELSE 0 END
            + CASE WHEN chunk.content CONTAINS "목적대로" THEN 20 ELSE 0 END
            + CASE WHEN chunk.content CONTAINS "상태로" THEN 10 ELSE 0 END
            + CASE WHEN chunk.content CONTAINS "반환" THEN 10 ELSE 0 END
            AS contract_score
        WHERE contract_score > 0
        RETURN
            {chunk_return_fields("chunk", "contract_score / 100.0")}
        ORDER BY contract_score DESC, chunk.chunk_number
        LIMIT 8
        """,
        source_paths=source_paths,
        party_terms=party_terms,
    )
    return [document_from_record(record) for record in records]


def retrieve_case_source_matches(run_read, query, source_paths) -> list[Document]:
    case_terms = ["판단", "이유", "주문", "결론", "원칙", "해석", "법리"]
    case_terms.extend(source_query_terms(query, classify_query(query)))
    records = run_read(
        f"""
        MATCH (chunk:Chunk)
        WHERE chunk.source_path IN $source_paths
        WITH chunk,
            reduce(score = 0, term IN $case_terms |
                score + CASE WHEN chunk.content CONTAINS term THEN 15 ELSE 0 END
            )
            + CASE WHEN chunk.chunk_number <= 5 THEN 10 ELSE 0 END
            AS case_score
        RETURN
            {chunk_return_fields("chunk", "case_score / 100.0")}
        ORDER BY
            CASE WHEN case_score > 0 THEN 0 ELSE 1 END,
            case_score DESC,
            chunk.chunk_number
        LIMIT 10
        """,
        source_paths=source_paths,
        case_terms=list(dict.fromkeys(case_terms)),
    )
    return [document_from_record(record) for record in records]


def attach_legal_context(run_read, documents) -> list[Document]:
    chunk_ids = [
        document.meta["id"]
        for document in documents
        if document.meta.get("id")
        and not str(document.meta.get("id")).startswith(GRAPH_FACT_ID_PREFIX)
    ]
    if not chunk_ids:
        return documents

    records = run_read(
        """
        MATCH (chunk:Chunk)
        WHERE chunk.id IN $chunk_ids
        OPTIONAL MATCH (unit:LegalUnit)-[:SUPPORTED_BY]->(chunk)
        OPTIONAL MATCH (article:LegalUnit {kind: "article"})-[:CONTAINS*0..3]->(unit)
        WITH chunk, unit, article, support
        ORDER BY
            CASE WHEN support.primary THEN 0 ELSE 1 END,
            CASE WHEN article IS NULL THEN 1 ELSE 0 END,
            CASE unit.kind
                WHEN "subitem" THEN 0
                WHEN "item" THEN 1
                WHEN "clause" THEN 2
                WHEN "article" THEN 3
                ELSE 4
            END
        RETURN
            chunk.id AS id,
            unit.id AS legal_unit_id,
            unit.kind AS legal_unit_kind,
            unit.number AS legal_unit_number,
            unit.title AS legal_unit_title,
            article.id AS article_id,
            article.number AS article_number,
            article.title AS article_title
        """,
        chunk_ids=chunk_ids,
    )
    context_by_id = {}
    for record in records:
        context_by_id.setdefault(record["id"], dict(record))

    enriched = []
    for document in documents:
        context = context_by_id.get(document.meta.get("id"))
        if not context:
            enriched.append(document)
            continue

        meta = dict(document.meta)
        for key, value in context.items():
            if key != "id" and value is not None:
                meta[key] = value
        specific_citation = specific_legal_citation(meta, document.content)
        if specific_citation:
            meta["specific_citation"] = specific_citation
        if not meta.get("document_type"):
            meta["document_type"] = document_type_from_source_path(
                meta.get("source_path", "")
            )
        enriched.append(Document(document.content, meta, document.score))
    return enriched


def attach_enumerated_items(run_read, documents) -> list[Document]:
    """Append numbered child items to a selected list-introducing clause."""
    seed_ids = [
        document.meta.get("id")
        for document in documents
        if document.meta.get("id")
        and ENUMERATED_LEAD_IN_PATTERN.search(document.content)
    ]
    if not seed_ids:
        return documents

    records = run_read(
        f"""
        UNWIND $seed_ids AS seed_id
        MATCH (seed:Chunk {{id: seed_id}})
        MATCH (unit:LegalUnit)-[:SUPPORTED_BY]->(seed)
        MATCH (unit)-[:CONTAINS*1..2]->(child_unit:LegalUnit)-[:SUPPORTED_BY]->(child:Chunk)
        WHERE child_unit.kind IN ["item", "subitem"]
        RETURN
            seed_id,
            {chunk_return_fields("child", "1.0", "child_unit.order")}
        ORDER BY seed_id, rank, chunk_number
        """,
        seed_ids=seed_ids,
    )
    children_by_seed = {}
    for record in records:
        contents = children_by_seed.setdefault(record["seed_id"], [])
        if record["content"] not in contents:
            contents.append(record["content"])

    def append_children(document: Document) -> Document:
        child_contents = children_by_seed.get(document.meta.get("id"), [])
        if not child_contents:
            return document
        content = "\n".join([document.content, *child_contents])
        return replace(
            document.with_meta(enumerated_item_count=len(child_contents)),
            content=content,
        )

    return [append_children(document) for document in documents]


def attach_parent_leads(run_read, documents) -> list[Document]:
    """Prepend the governing clause when retrieval selects a numbered child."""
    child_ids = [
        document.meta.get("id")
        for document in documents
        if document.meta.get("id")
        and document.meta.get("legal_unit_kind") in {"item", "subitem"}
    ]
    if not child_ids:
        return documents

    records = run_read(
        """
        UNWIND $child_ids AS child_id
        MATCH (child:Chunk {id: child_id})
        MATCH (child_unit:LegalUnit)-[:SUPPORTED_BY]->(child)
        MATCH (parent:LegalUnit)-[:CONTAINS*1..2]->(child_unit)
        WHERE parent.kind IN ["clause", "article"]
        MATCH (parent)-[:SUPPORTED_BY]->(lead:Chunk)
        WHERE lead.id <> child.id
        WITH child_id, parent, lead
        ORDER BY child_id,
            CASE parent.kind WHEN "clause" THEN 0 ELSE 1 END,
            lead.chunk_number
        WITH child_id, collect(lead)[0] AS lead
        RETURN child_id, lead.id AS lead_id, lead.content AS lead_content
        """,
        child_ids=child_ids,
    )
    leads = {record["child_id"]: record for record in records}
    enriched = []
    for document in documents:
        lead = leads.get(document.meta.get("id"))
        if not lead or lead["lead_content"] in document.content:
            enriched.append(document)
            continue
        enriched.append(replace(
            document.with_meta(parent_lead_id=lead["lead_id"]),
            content=f"{lead['lead_content']}\n{document.content}",
        ))
    return enriched


def specific_legal_citation(meta: dict, content: str) -> str | None:
    article_number = meta.get("article_number")
    unit_kind = meta.get("legal_unit_kind")
    unit_number = meta.get("legal_unit_number")
    if not article_number:
        return None
    first_circled = re.search(r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]", content[:80])
    if first_circled:
        clause_number = CIRCLED_CLAUSE_MARKS.index(first_circled.group(0)) + 1
        return f"{article_number}제{clause_number}항"
    if not unit_kind or not unit_number:
        return article_number
    if unit_kind == "article":
        return article_number
    if unit_kind == "clause" and unit_number in CIRCLED_CLAUSE_MARKS:
        clause_number = CIRCLED_CLAUSE_MARKS.index(unit_number) + 1
        return f"{article_number}제{clause_number}항"
    if unit_kind == "item" and unit_number.endswith("."):
        return f"{article_number}제{unit_number[:-1]}호"
    if unit_kind == "subitem" and unit_number.endswith("."):
        return f"{article_number}{unit_number[:-1]}목"
    return article_number


def fuse_ranked_results(
    ranked_results: list[tuple[str, list[Document]]],
) -> list[Document]:
    """Combine retriever ranks without comparing incompatible raw scores.

    Rank fusion gives each document a small contribution based on its position
    in each retriever. Term-only channels receive lower configured weights, so
    repeated generic words cannot dominate semantic or exact-phrase evidence.
    """
    documents_by_key = {}
    scores_by_key = {}
    channels_by_key = {}
    first_seen = {}
    seen_counter = 0

    for channel, documents in ranked_results:
        weight = RETRIEVAL_CHANNEL_WEIGHTS[channel]
        seen_in_channel = set()
        for rank, document in enumerate(documents, start=1):
            key = document.evidence_key
            if key in seen_in_channel:
                continue
            seen_in_channel.add(key)
            if key not in documents_by_key:
                documents_by_key[key] = document
                first_seen[key] = seen_counter
                seen_counter += 1
            scores_by_key[key] = scores_by_key.get(key, 0.0) + (
                weight / (RANK_FUSION_OFFSET + rank)
            )
            channels_by_key.setdefault(key, []).append(channel)

    ordered_keys = sorted(
        documents_by_key,
        key=lambda key: (-scores_by_key[key], first_seen[key]),
    )
    fused = []
    for key in ordered_keys:
        document = documents_by_key[key]
        fused.append(replace(
            document.with_meta(
                retrieval_channels=channels_by_key[key],
                fusion_score=scores_by_key[key],
            ),
            score=scores_by_key[key],
        ))
    return fused


def select_evidence(
    documents: list[Document],
    top_k: int,
    max_words: int = MAX_CONTEXT_WORDS,
) -> list[Document]:
    """Keep a shared, ordered evidence budget for every retrieval mode.

    Words are a simple prompt-size proxy because chunks may come from models
    with different tokenizers. A first oversized chunk is kept so an answer is
    never generated with no evidence.
    """
    ranked_documents = sorted(
        enumerate(documents),
        key=lambda item: (
            -item[1].meta.get(
                "fusion_score",
                item[1].score if item[1].score is not None else 0.0,
            ),
            item[0],
        ),
    )
    selected = []
    article_counts = {}
    word_count = 0
    for _original_position, document in ranked_documents:
        if len(selected) >= top_k:
            break
        article_key = (
            document.meta.get("source_path"),
            document.meta.get("article_number"),
        )
        if article_key[1] and article_counts.get(article_key, 0) >= 3:
            continue
        document_words = len(document.content.split())
        if selected and word_count + document_words > max_words:
            continue
        selected.append(document)
        if article_key[1]:
            article_counts[article_key] = article_counts.get(article_key, 0) + 1
        word_count += document_words
    return selected


def expand_legal_units(run_read, retrieved_chunks, route, source_paths) -> list[Document]:
    unit_windows = [
        {
            "rank": rank,
            "chunk_id": chunk.meta.get("id"),
        }
        for rank, chunk in enumerate(retrieved_chunks)
        if chunk.meta.get("id")
    ]
    if not unit_windows:
        return retrieved_chunks

    records = run_read(
        f"""
        UNWIND $unit_windows AS window
        MATCH (seed:Chunk {{id: window.chunk_id}})
        MATCH (legal_unit:LegalUnit)-[:SUPPORTED_BY]->(seed)
        OPTIONAL MATCH (article:LegalUnit {{kind: "article"}})-[:CONTAINS*0..3]->(legal_unit)
        WITH window, coalesce(article, legal_unit) AS expansion_unit
        WHERE
            ($source_paths = [] OR expansion_unit.source_path IN $source_paths)
            AND (
                $article_numbers = []
                OR expansion_unit.number IN $article_numbers
            )
        MATCH (expansion_unit)-[:CONTAINS*0..3]->(descendant:LegalUnit)
        MATCH (descendant)-[:SUPPORTED_BY]->(chunk:Chunk)
        WITH window, chunk
        ORDER BY window.rank, chunk.chunk_number
        WITH window.rank AS rank, collect(chunk)[..$expansion_limit] AS chunks
        UNWIND chunks AS chunk
        RETURN
            {chunk_return_fields("chunk", "1.0", "rank")}
        ORDER BY rank, chunk_number
        """,
        unit_windows=unit_windows,
        source_paths=source_paths,
        article_numbers=route.article_numbers,
        expansion_limit=100 if route.article_numbers else LEGAL_UNIT_EXPANSION_LIMIT,
    )
    return merge_ranked_expansion(retrieved_chunks, records)


def merge_ranked_expansion(retrieved_chunks, records) -> list[Document]:
    retrieved_by_id = {chunk.meta["id"]: chunk for chunk in retrieved_chunks}
    records_by_rank = {}
    for record in records:
        records_by_rank.setdefault(record["rank"], []).append(record)

    context_chunks = []
    seen_ids = set()
    for rank, chunk in enumerate(retrieved_chunks):
        if chunk.meta["id"] not in seen_ids:
            seen_ids.add(chunk.meta["id"])
            context_chunks.append(chunk)

        for record in records_by_rank.get(rank, []):
            if record["id"] in seen_ids:
                continue
            seen_ids.add(record["id"])
            expanded = retrieved_by_id.get(record["id"])
            if expanded is None:
                expanded = document_from_record(record)
                seed_score = chunk.meta.get(
                    "fusion_score",
                    chunk.score if chunk.score is not None else 0.0,
                )
                expanded = replace(
                    expanded.with_meta(
                        fusion_score=seed_score * 0.75,
                        retrieval_channels=["legal_unit_expansion"],
                        expanded_from=chunk.meta.get("id"),
                    ),
                    score=seed_score * 0.75,
                )
            context_chunks.append(expanded)
    return context_chunks


def prioritize_symbolic_evidence(
    query: str,
    evidence: list[Document],
    graph_fact_list: list[GraphFact],
) -> tuple[list[Document], dict | None]:
    """Order evidence using a rule trace and return the trace separately."""
    if not evidence:
        return evidence, None
    trace_evidence = evidence + [fact.to_document() for fact in graph_fact_list]
    trace = build_symbolic_trace(query, trace_evidence)
    important_keys = {
        (item.get("document"), item.get("chunk"))
        for item in trace.get("important_evidence", [])
    }
    evidence = sorted(
        enumerate(evidence),
        key=lambda item: (
            (
                item[1].meta.get("document_name"),
                item[1].meta.get("chunk_number"),
            ) not in important_keys,
            item[0],
        ),
    )
    evidence = [document for _index, document in evidence]
    return evidence, trace


def select_graph_facts(
    graph_facts_result: list[GraphFact], top_k: int,
) -> list[GraphFact]:
    """Deduplicate and rank graph facts without hiding them in evidence metadata."""
    seen_fact_keys = set()
    ranked_facts = []
    for graph_fact in sorted(
        graph_facts_result, key=lambda fact: fact.score or 0, reverse=True,
    ):
        key = (graph_fact.source_path, graph_fact.path, graph_fact.sentence)
        if key in seen_fact_keys:
            continue
        seen_fact_keys.add(key)
        ranked_facts.append(graph_fact)

    return ranked_facts[:max(1, top_k)]


def graph_facts_for_route(
    graph_fact_list: list[GraphFact], route: QueryRoute,
) -> list[GraphFact]:
    """Keep relation types that can answer a specifically routed question.

    A relation type is the legal meaning of a graph edge, for example
    ``PROHIBITS`` or ``HAS_DUTY``.  General and article questions retain all
    relations because their required answer shape is not known in advance.
    """
    tasks = route.tasks or [route.kind]
    relation_sets = [
        GRAPH_RELATIONS_BY_QUERY_TYPE[task]
        for task in tasks
        if task in GRAPH_RELATIONS_BY_QUERY_TYPE
    ]
    if not relation_sets:
        return graph_fact_list
    allowed_relations = set().union(*relation_sets)
    return [
        fact for fact in graph_fact_list if fact.relation in allowed_relations
    ]


def retrieve_unconstrained(
    run_read, embedding_model, top_k, query, route, prefer_primary_sources=False,
) -> list[Document]:
    exact_chunks = retrieve_exact_phrase_matches(run_read, query)
    vector_chunks = retrieve_vector_chunks(run_read, embedding_model, top_k, query)
    sparse_chunks = retrieve_sparse_chunks(run_read, route)
    lexical_chunks = retrieve_legal_unit_matches(run_read, route)
    authority_chunks = (
        [
            chunk for chunk in vector_chunks + lexical_chunks
            if (
                chunk.meta.get("document_type")
                or document_type_from_source_path(chunk.meta.get("source_path", ""))
            ) in {"statute", "regulation"}
        ]
        if prefer_primary_sources
        else []
    )
    return fuse_ranked_results([
        ("exact_phrase", exact_chunks),
        ("vector", vector_chunks),
        ("legal_unit", lexical_chunks),
        ("sparse_terms", sparse_chunks),
        ("authority", authority_chunks),
    ])


def retrieve_source_first_evidence(
    run_read, embedding_model, top_k, query, route, source_paths,
    is_summary_query, is_contract_query, is_case_query,
) -> list[Document]:
    focused_query = source_question_focus(query)
    focused_route = replace(
        route,
        search_terms=query_terms(
            focused_query,
            ignored_terms=QUERY_STOP_TERMS,
            max_terms=10,
            strip_particles=True,
        ),
    )
    exact_chunks = retrieve_exact_phrase_matches(
        run_read, focused_query, source_paths, include_adjacent=True,
    )
    vector_chunks = retrieve_vector_chunks(
        run_read, embedding_model, top_k, query, source_paths=source_paths,
    )
    summary_chunks = (
        retrieve_document_summary_matches(run_read, query, source_paths)
        if is_summary_query
        else []
    )
    contract_chunks = (
        retrieve_contract_matches(run_read, query, source_paths)
        if is_contract_query
        else []
    )
    case_chunks = (
        retrieve_case_source_matches(run_read, query, source_paths)
        if is_case_query
        else []
    )
    source_chunks = retrieve_source_constrained_matches(
        run_read, focused_query, focused_route, source_paths,
    )
    lexical_chunks = retrieve_legal_unit_matches(
        run_read, focused_route, source_paths=source_paths,
    )
    return fuse_ranked_results([
        ("exact_phrase", exact_chunks),
        ("vector", vector_chunks),
        ("summary", summary_chunks),
        ("contract", contract_chunks),
        ("case", case_chunks),
        ("provision_focus", lexical_chunks[:2]),
        ("legal_unit", lexical_chunks),
        ("source_terms", source_chunks),
    ])


def build_retrieval_context(
    query: str, source_matcher, category: str | None = None,
) -> RetrievalContext:
    route = classify_query(query, category)
    named_source_paths = source_matcher.constraints(query)
    category_source_paths = source_matcher.category_constraints(category)
    source_constraint_mismatch = False
    if named_source_paths and category_source_paths:
        source_paths = [
            path for path in named_source_paths if path in category_source_paths
        ]
        source_constraint_mismatch = not source_paths
    else:
        source_paths = named_source_paths or category_source_paths
    return RetrievalContext(
        query=query,
        route=route,
        source_paths=source_paths,
        is_summary_query=is_document_summary_query(query, source_paths),
        is_contract_query=is_contract_query(query, source_paths),
        is_case_query=is_case_query(query, source_paths),
        source_constraint_mismatch=source_constraint_mismatch,
    )


def retrieve_baseline(
    run_read, embedding_model, top_k, query, source_matcher, category=None,
) -> tuple[RetrievalContext, list[Document]]:
    """Build the common text evidence supplied to all three retrieval modes."""
    context = build_retrieval_context(query, source_matcher, category)
    if context.source_constraint_mismatch:
        return context, []
    if context.source_paths:
        chunks = retrieve_source_first_evidence(
            run_read, embedding_model, top_k, query, context.route,
            context.source_paths, context.is_summary_query,
            context.is_contract_query, context.is_case_query,
        )
    else:
        chunks = retrieve_unconstrained(
            run_read,
            embedding_model,
            top_k,
            query,
            context.route,
            prefer_primary_sources=(
                not context.is_summary_query
                and not context.is_contract_query
                and not context.is_case_query
            ),
        )
    chunks = attach_legal_context(run_read, chunks)
    chunks = attach_parent_leads(run_read, chunks)
    if not context.is_summary_query and should_expand_legal_units(query):
        chunks = expand_legal_units(
            run_read, chunks, context.route, context.source_paths,
        )
        chunks = attach_legal_context(run_read, chunks)
        chunks = attach_parent_leads(run_read, chunks)
    chunks = attach_enumerated_items(run_read, chunks)
    return context, select_evidence(chunks, min(top_k, MAX_EVIDENCE_CHUNKS))


def retrieve_graph_facts(
    run_read, chunks, query, route, source_paths, is_summary_query, top_k
) -> list[GraphFact]:
    graph_seed_chunks = chunks[:8]
    graph_source_paths = source_paths or list(
        dict.fromkeys(
            chunk.meta["source_path"]
            for chunk in graph_seed_chunks[:4]
            if chunk.meta.get("source_path")
        )
    )
    graph_fact_list = (
        graph_facts(
            run_read,
            query,
            graph_seed_chunks,
            chunk_ids=[chunk.meta["id"] for chunk in graph_seed_chunks if chunk.meta.get("id")],
            source_paths=graph_source_paths,
            terms=route.search_terms,
        )
        if not is_summary_query
        else []
    )
    graph_fact_list = graph_facts_for_route(graph_fact_list, route)
    return select_graph_facts(graph_fact_list, top_k)


def retrieve_vector_evidence(
    run_read, embedding_model, top_k, query, source_matcher, category,
) -> tuple[RetrievalContext, list[Document]]:
    """Retrieve dense-vector evidence without legal rules or graph expansion."""
    context = build_retrieval_context(query, source_matcher, category)
    if context.source_constraint_mismatch:
        return context, []
    chunks = retrieve_vector_chunks(
        run_read,
        embedding_model,
        top_k,
        query,
        source_paths=context.source_paths,
    )
    chunks = attach_legal_context(run_read, chunks)
    return context, select_evidence(chunks, min(top_k, MAX_EVIDENCE_CHUNKS))


def retrieve_chunks(
    run_read, embedding_model, top_k, query, source_matcher, strategy="graph",
    category=None,
) -> RetrievalResult:
    """Run the retrieval pipeline selected by ``strategy``.

    Vector uses dense similarity only. Graph adds routed lexical/legal retrieval
    and graph relations. Neuro-symbolic additionally orders that graph evidence
    using its rule trace. All modes share the same final evidence budget.
    """
    strategy = normalize_retrieval_strategy(strategy)
    if strategy not in {"vector", "graph", "neuro_symbolic"}:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")
    if strategy == "vector":
        context, chunks = retrieve_vector_evidence(
            run_read, embedding_model, top_k, query, source_matcher, category,
        )
        return RetrievalResult(strategy, context, chunks)

    context, chunks = retrieve_baseline(
        run_read, embedding_model, top_k, query, source_matcher, category,
    )
    graph_fact_list = retrieve_graph_facts(
        run_read, chunks, query, context.route, context.source_paths,
        context.is_summary_query, top_k,
    )
    trace = None
    if strategy == "neuro_symbolic":
        chunks, trace = prioritize_symbolic_evidence(
            query, chunks, graph_fact_list,
        )
    return RetrievalResult(strategy, context, chunks, graph_fact_list, trace)
