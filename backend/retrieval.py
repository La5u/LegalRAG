import re
from dataclasses import dataclass

from backend.config import NEO4J_VECTOR_INDEX, normalize_retrieval_strategy
from backend.constants import CIRCLED_CLAUSE_MARKS, GRAPH_FACT_ID_PREFIX
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
RULE_APPLICATION_KINDS = {"condition", "exception", "prohibition", "obligation"}
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


@dataclass(frozen=True)
class RetrievalContext:
    query: str
    route: QueryRoute
    source_paths: list[str]
    is_summary_query: bool
    is_contract_query: bool
    is_case_query: bool


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
    terms = query_terms(
        query,
        initial_terms=route.search_terms,
        ignored_terms=set(SOURCE_ALIASES) | {"대법원", "헌재", "헌법재판소"},
        max_terms=12,
        min_length=2,
        strip_particles=True,
    )
    expanded_terms = []
    for cleaned in terms:
        expanded_terms.append(cleaned)
        if cleaned == "표시광고":
            expanded_terms.extend(["표시", "광고"])
    if "비교" in query and "광고" in query:
        expanded_terms.extend(["부당하게", "비교하는"])
    return list(dict.fromkeys(expanded_terms))[:12]


def retrieve_exact_phrase_matches(
    run_read, query, source_paths=None
) -> list[Document]:
    phrases = quoted_terms(query)
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
        meta = dict(document.meta)
        meta["exact_phrase_match"] = True
        meta["matched_phrases"] = phrases
        documents.append(Document(document.content, meta, document.score))
    return documents


def retrieve_vector_chunks(run_read, embedding_model, top_k, query) -> list[Document]:
    if not vector_index_exists(run_read, NEO4J_VECTOR_INDEX):
        raise RetrievalUnavailable("No vector index exists. Run indexing first.")

    query_embedding = embedding_model.encode(
        query, normalize_embeddings=True,
    ).tolist()
    records = run_read(
        f"""
        CALL db.index.vector.queryNodes(
            $index_name,
            $top_k,
            $query_embedding
        )
        YIELD node, score
        RETURN
            {chunk_return_fields("node", "score")}
        ORDER BY score DESC
        """,
        index_name=NEO4J_VECTOR_INDEX,
        top_k=top_k,
        query_embedding=query_embedding,
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
    records = run_read(
        f"""
        MATCH (chunk:Chunk)
        WHERE chunk.source_path IN $source_paths
        OPTIONAL MATCH (unit:LegalUnit)-[:SUPPORTED_BY]->(chunk)
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
            ) AS source_score
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
        WITH chunk, unit, article
        ORDER BY
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


def merge_documents(documents: list[Document]) -> list[Document]:
    merged = []
    seen_ids = set()
    for document in documents:
        document_id = document.meta.get("id")
        if document_id is None:
            merged.append(document)
            continue
        if document_id in seen_ids:
            continue
        seen_ids.add(document_id)
        merged.append(document)
    return merged


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
        expansion_limit=100 if route.kind == "article_lookup" else LEGAL_UNIT_EXPANSION_LIMIT,
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
            context_chunks.append(
                retrieved_by_id.get(record["id"], document_from_record(record))
            )
    return context_chunks


def attach_symbolic_trace(query, evidence) -> list[Document]:
    if not evidence:
        return evidence
    trace_evidence = evidence + graph_facts_from_meta(evidence)
    trace = build_symbolic_trace(query, trace_evidence)
    first_document = evidence[0]
    meta = dict(first_document.meta)
    meta["symbolic_trace"] = trace
    return [Document(first_document.content, meta, first_document.score)] + evidence[1:]


def attach_graph_facts(chunks, graph_facts_result: list[GraphFact], top_k) -> list[Document]:
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

    if not chunks or not ranked_facts:
        return chunks

    meta = dict(chunks[0].meta)
    meta["graph_facts"] = ranked_facts[:max(1, top_k)]
    return [Document(chunks[0].content, meta, chunks[0].score)] + chunks[1:]


def graph_facts_from_meta(chunks: list[Document]) -> list[Document]:
    graph_fact_list = next(
        (chunk.meta.get("graph_facts") for chunk in chunks if chunk.meta.get("graph_facts")),
        [],
    )
    return [fact.to_document() for fact in graph_fact_list]


def filter_by_category(chunks: list[Document], category: str) -> list[Document]:
    if not category or category == "all":
        return chunks
    return [
        c for c in chunks
        if document_type_from_source_path(c.meta.get("source_path", "")) == category
    ]


def retrieve_unconstrained(run_read, embedding_model, top_k, query, route) -> list[Document]:
    exact_chunks = retrieve_exact_phrase_matches(run_read, query)
    vector_chunks = retrieve_vector_chunks(run_read, embedding_model, top_k, query)
    sparse_chunks = retrieve_sparse_chunks(run_read, route)
    lexical_chunks = retrieve_legal_unit_matches(run_read, route)
    if route.article_numbers or route.kind in RULE_APPLICATION_KINDS:
        return merge_documents(exact_chunks + lexical_chunks + vector_chunks + sparse_chunks)
    return merge_documents(exact_chunks + vector_chunks + lexical_chunks + sparse_chunks)


def retrieve_vectorrag(run_read, embedding_model, top_k, query) -> list[Document]:
    exact_chunks = retrieve_exact_phrase_matches(run_read, query)
    vector_chunks = retrieve_vector_chunks(run_read, embedding_model, top_k, query)
    return attach_legal_context(run_read, merge_documents(exact_chunks + vector_chunks))


def retrieve_source_first_evidence(
    run_read, query, route, source_paths, is_summary_query, is_contract_query, is_case_query
) -> list[Document]:
    exact_chunks = retrieve_exact_phrase_matches(run_read, query, source_paths)
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
    source_chunks = retrieve_source_constrained_matches(run_read, query, route, source_paths)
    lexical_chunks = retrieve_legal_unit_matches(run_read, route, source_paths=source_paths)
    if route.article_numbers:
        return merge_documents(
            exact_chunks
            + summary_chunks
            + contract_chunks
            + lexical_chunks
            + case_chunks
            + source_chunks
        )
    return merge_documents(
        exact_chunks
        + summary_chunks
        + contract_chunks
        + case_chunks
        + source_chunks
        + lexical_chunks
    )


def build_retrieval_context(query: str, source_matcher) -> RetrievalContext:
    route = classify_query(query)
    source_paths = source_matcher.constraints(query)
    return RetrievalContext(
        query=query,
        route=route,
        source_paths=source_paths,
        is_summary_query=is_document_summary_query(query, source_paths),
        is_contract_query=is_contract_query(query, source_paths),
        is_case_query=is_case_query(query, source_paths),
    )


def retrieve_initial_evidence(
    run_read,
    embedding_model,
    top_k,
    context: RetrievalContext,
) -> list[Document]:
    if context.source_paths:
        return retrieve_source_first_evidence(
            run_read,
            context.query,
            context.route,
            context.source_paths,
            context.is_summary_query,
            context.is_contract_query,
            context.is_case_query,
        )
    return retrieve_unconstrained(
        run_read,
        embedding_model,
        top_k,
        context.query,
        context.route,
    )


def enrich_chunks(
    run_read, chunks, query, route, source_paths, is_summary_query
) -> list[Document]:
    chunks = attach_legal_context(run_read, chunks)
    if not is_summary_query and should_expand_legal_units(query):
        chunks = expand_legal_units(run_read, chunks, route, source_paths)
        chunks = attach_legal_context(run_read, chunks)
    if route.kind == "article_lookup" and route.article_numbers:
        chunks = [chunk for chunk in chunks if chunk.meta.get("article_number") in route.article_numbers]
    return chunks


def add_graph_facts(
    run_read, chunks, query, route, source_paths, is_summary_query, is_contract_query, top_k
) -> list[Document]:
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
        if route.include_graph_facts and not is_summary_query and not is_contract_query
        else []
    )
    return attach_graph_facts(chunks, graph_fact_list, top_k)


def retrieve_graphrag(run_read, embedding_model, top_k, query, source_matcher) -> list[Document]:
    context = build_retrieval_context(query, source_matcher)
    chunks = retrieve_initial_evidence(run_read, embedding_model, top_k, context)
    chunks = enrich_chunks(
        run_read,
        chunks,
        context.query,
        context.route,
        context.source_paths,
        context.is_summary_query,
    )
    chunks = add_graph_facts(
        run_read,
        chunks,
        context.query,
        context.route,
        context.source_paths,
        context.is_summary_query,
        context.is_contract_query,
        top_k,
    )
    return chunks


def retrieve_chunks(
    run_read, embedding_model, top_k, query, source_matcher, strategy="graph"
) -> list[Document]:
    """Run the retrieval pipeline selected by ``strategy``.

    - ``vector``: exact-phrase + vector search only (no graph facts, no expansion).
    - ``graph``: full GraphRAG pipeline (routing, source matching, expansion, graph facts).
    - ``neuro_symbolic``: the full pipeline with a symbolic trace attached to the first chunk.
    """
    strategy = normalize_retrieval_strategy(strategy)
    if strategy == "vector":
        return retrieve_vectorrag(run_read, embedding_model, top_k, query)
    if strategy == "graph":
        return retrieve_graphrag(run_read, embedding_model, top_k, query, source_matcher)
    if strategy == "neuro_symbolic":
        return attach_symbolic_trace(
            query,
            retrieve_graphrag(run_read, embedding_model, top_k, query, source_matcher),
        )
    raise ValueError(f"Unknown retrieval strategy: {strategy}")
