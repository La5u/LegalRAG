import re
from dataclasses import dataclass

from backend.constants import (
    CONCEPT_STOP_TERMS,
    CONCEPT_STOP_TERMS_LIST,
    GRAPH_FACT_ID_PREFIX,
    QUERY_STOP_TERMS,
)
from backend.document import Document
from backend.records import chunk_meta
from backend.text import extract_tokens
from backend.triples import CONCEPT_RELATION_TYPES, LEGAL_RELATIONS





@dataclass(frozen=True)
class GraphFact:
    id: str
    document: str
    source_path: str
    chunk: str | int
    relation: str
    path: str
    sentence: str
    score: float
    article: str | None = None
    legal_unit_id: str | None = None
    case_number: str | None = None

    def to_document(self) -> Document:
        return Document(
            content=self.content,
            meta=chunk_meta(
                id=self.id,
                chunk_number=self.chunk,
                document_name=self.document,
                group_id=self.chunk,
                part=0,
                source_path=self.source_path,
                legal_unit_id=self.legal_unit_id,
                article_number=self.article,
                graph_relation=self.relation,
                case_number=self.case_number,
            ),
            score=self.score,
        )

    @property
    def content(self) -> str:
        return f"그래프 사실:\n{self.path}\n근거 문장: {self.sentence}"

    def to_response(self) -> dict:
        return {
            "document": self.document,
            "raw_document": self.document,
            "chunk": self.chunk,
            "article": self.article,
            "relation": self.relation,
            "path": self.path,
            "sentence": self.sentence,
            "score": self.score,
        }


def graph_facts(
    run_read,
    query: str,
    chunks: list[Document],
    chunk_ids: list[str] | None = None,
    source_paths: list[str] | None = None,
    terms: list[str] | None = None,
) -> list[GraphFact]:
    chunk_ids = chunk_ids or [chunk.meta["id"] for chunk in chunks if chunk.meta.get("id")]
    source_paths = source_paths or list(
        dict.fromkeys(
            chunk.meta["source_path"]
            for chunk in chunks
            if chunk.meta.get("source_path")
        )
    )
    if terms is None:
        terms = query_terms(query)
    focus_article_numbers = list(
        dict.fromkeys(
            article_terms(query)
            + [
                chunk.meta["article_number"]
                for chunk in chunks
                if chunk.meta.get("article_number")
            ]
        )
    )
    return (
        _case_holding_facts(run_read, query, source_paths, terms)
        + _graph_relation_facts(
            run_read,
            query,
            chunk_ids,
            source_paths,
            terms,
            focus_article_numbers,
        )
        + _graph_definition_facts(run_read, query, chunk_ids, source_paths, terms)
    )


def graph_fact_documents(*args, **kwargs) -> list[Document]:
    return [fact.to_document() for fact in graph_facts(*args, **kwargs)]


def _case_holding_facts(
    run_read,
    query: str,
    source_paths: list[str],
    terms: list[str],
) -> list[GraphFact]:
    if not wants_case_fact(query):
        return []

    case_count = run_read("MATCH (case_node:Case) RETURN count(case_node) AS count")
    if not case_count or case_count[0]["count"] == 0:
        return []

    case_terms = terms + case_terms_from_query(query)
    case_records = run_read(
        """
        MATCH (case_node:Case)-[:HAS_HOLDING]->(holding:Holding)-[:SUPPORTED_BY]->(chunk:Chunk)
        WHERE (
            ($source_paths <> [] AND case_node.source_path IN $source_paths)
            OR (
                $source_paths = []
                AND (
                    any(term IN $case_terms WHERE case_node.case_number CONTAINS term)
                    OR any(term IN $case_terms WHERE case_node.document_name CONTAINS term)
                    OR any(term IN $case_terms WHERE holding.text CONTAINS term)
                )
            )
        )
        RETURN
            case_node.case_number AS case_number,
            holding.text AS text,
            holding.document_name AS document_name,
            holding.source_path AS source_path,
            chunk.id AS chunk_id,
            holding.chunk_number AS chunk_number
        ORDER BY
            CASE
                WHEN any(term IN $case_terms WHERE case_node.case_number CONTAINS term) THEN 0
                WHEN any(term IN $case_terms WHERE holding.text CONTAINS term) THEN 1
                ELSE 2
            END,
            holding.chunk_number
        LIMIT 8
        """,
        source_paths=list(dict.fromkeys(source_paths)),
        case_terms=case_terms,
    )
    return [
        GraphFact(
            id=f"{GRAPH_FACT_ID_PREFIX}HAS_HOLDING::{record['chunk_id']}::{index}",
            document=record["document_name"],
            source_path=record["source_path"],
            chunk=f"graph-case-{index}",
            relation="HAS_HOLDING",
            path=f"{record['case_number']} -HAS_HOLDING-> Holding",
            sentence=record["text"],
            score=0.94 - index * 0.001,
            case_number=record["case_number"],
        )
        for index, record in enumerate(case_records, start=1)
    ]


def _graph_relation_facts(
    run_read,
    query: str,
    chunk_ids: list[str],
    source_paths: list[str],
    _terms: list[str],
    focus_article_numbers: list[str],
) -> list[GraphFact]:
    relation_records = run_read(
        """
        MATCH (source:Concept)-[relation]->(target:Concept)
        WHERE type(relation) IN $relations
            AND ($has_source_paths = false OR relation.source_path IN $source_paths)
            AND NOT source.name IN $concept_stop_terms
            AND NOT target.name IN $concept_stop_terms
            AND size(source.name) > 1
            AND size(target.name) > 1
        OPTIONAL MATCH (chunk:Chunk {id: relation.chunk_id})
        OPTIONAL MATCH (unit:LegalUnit)-[:SUPPORTED_BY]->(chunk)
        OPTIONAL MATCH (article:LegalUnit {kind: "article"})-[:CONTAINS*0..3]->(unit)
        WITH source, relation, target, chunk, unit, article
        WHERE (
            relation.chunk_id IN $chunk_ids
            OR article.number IN $article_numbers
        )
        AND (
                $allow_case_sources
                OR (
                    NOT relation.source_path CONTAINS "/precedents/"
                    AND NOT relation.source_path CONTAINS "/constitutional_decisions/"
                )
            )
        WITH source, relation, target, chunk, unit, article,
            CASE WHEN type(relation) IN $preferred_relations THEN 0 ELSE 1 END
                AS relation_priority,
            CASE
                WHEN source.name IN $article_terms THEN 0
                WHEN target.name IN $article_terms THEN 0
                WHEN article.number IN $article_terms THEN 0
                ELSE 1
            END AS article_priority
        ORDER BY relation_priority, article_priority, relation.chunk_number
        RETURN
            source.name AS source,
            type(relation) AS relation,
            target.name AS target,
            relation_priority AS relation_priority,
            article_priority AS article_priority,
            relation.sentence AS sentence,
            relation.document_name AS document_name,
            relation.source_path AS source_path,
            relation.chunk_id AS chunk_id,
            relation.chunk_number AS chunk_number,
            unit.id AS legal_unit_id,
            article.number AS article_number
        LIMIT 20
        """,
        relations=CONCEPT_RELATION_TYPES,
        preferred_relations=preferred_relations(query),
        concept_stop_terms=CONCEPT_STOP_TERMS_LIST,
        article_terms=article_terms(query),
        chunk_ids=chunk_ids,
        source_paths=list(dict.fromkeys(source_paths)),
        article_numbers=focus_article_numbers,
        has_source_paths=bool(source_paths),
        allow_case_sources=wants_case_fact(query),
    )
    return [
        GraphFact(
            id=f"{GRAPH_FACT_ID_PREFIX}{record['relation']}::{record['chunk_id']}::{index}",
            document=record["document_name"],
            source_path=record["source_path"],
            chunk=f"graph-{index}",
            relation=record["relation"],
            path=(
                (f"[{record['article_number']}] " if record.get("article_number") else "")
                + f"{record['source']} -{record['relation']}-> {record['target']}"
            ),
            sentence=record["sentence"],
            score=(
                0.95
                - record["relation_priority"] * 0.1
                - record["article_priority"] * 0.05
                - index * 0.001
            ),
            article=record.get("article_number"),
            legal_unit_id=record.get("legal_unit_id"),
        )
        for index, record in enumerate(relation_records, start=1)
    ]


def _graph_definition_facts(
    run_read,
    query: str,
    chunk_ids: list[str],
    source_paths: list[str],
    terms: list[str],
) -> list[GraphFact]:
    if not wants_definition(query):
        return []

    definition_records = run_read(
        """
        MATCH (term:Term)-[:DEFINED_AS]->(definition:Definition)
        OPTIONAL MATCH (definition)-[:DEFINED_IN]->(article:LegalUnit)
        WITH term, definition, article
        WHERE (
            ($source_paths <> [] AND definition.source_path IN $source_paths)
            OR (
                $source_paths = []
                AND (
                    definition.chunk_id IN $chunk_ids
                    OR article.number IN $article_numbers
                    OR any(query_term IN $query_terms WHERE term.name CONTAINS query_term)
                )
            )
        )
        RETURN
            term.name AS term,
            definition.text AS text,
            definition.document_name AS document_name,
            definition.source_path AS source_path,
            definition.chunk_id AS chunk_id,
            definition.chunk_number AS chunk_number,
            article.id AS article_id,
            article.number AS article_number
        ORDER BY definition.chunk_number
        LIMIT 10
        """,
        chunk_ids=chunk_ids,
        source_paths=list(dict.fromkeys(source_paths)),
        article_numbers=article_terms(query),
        query_terms=terms,
    )
    return [
        GraphFact(
            id=f"{GRAPH_FACT_ID_PREFIX}DEFINED_AS::{record['chunk_id']}::{index}",
            document=record["document_name"],
            source_path=record["source_path"],
            chunk=f"graph-definition-{index}",
            relation="DEFINED_AS",
            path=(
                (f"[{record['article_number']}] " if record.get("article_number") else "")
                + f"{record['term']} -DEFINED_AS-> Definition"
            ),
            sentence=record["text"],
            score=0.9 - index * 0.001,
            article=record.get("article_number"),
            legal_unit_id=record.get("article_id"),
        )
        for index, record in enumerate(definition_records, start=1)
    ]


def query_terms(query: str) -> list[str]:
    return [
        term
        for term in extract_tokens(query)
        if term.casefold() not in QUERY_STOP_TERMS
        and term not in CONCEPT_STOP_TERMS
    ][:8]


def wants_definition(query: str) -> bool:
    lowered = query.casefold()
    return any(
        term in lowered
        for term in ["정의", "이란", "란 무엇", "define", "defined", "definition", "mean"]
    )


def wants_case_fact(query: str) -> bool:
    return any(
        term in query
        for term in ["판례", "판결", "결정", "사건번호", "대법원", "헌재", "헌법재판소", "lbox"]
    ) or bool(re.search(r"\d{4}[가-힣]{1,3}\d+", query))


def case_terms_from_query(query: str) -> list[str]:
    terms = re.findall(r"\d{4}[가-힣]{1,3}\d+(?:,\s*\d+)?|lbox_precedent_\d+", query)
    return [term.replace(" ", "") for term in terms]


def article_terms(query: str) -> list[str]:
    return re.findall(r"제\d+조(?:의\d+)?", query)


def preferred_relations(query: str) -> list[str]:
    preferred = []
    if "예외" in query:
        preferred.append("EXCEPTION_TO")
    if "통지" in query or "알려" in query:
        preferred.append("MUST_NOTIFY")
    if "요구" in query or "청구" in query:
        preferred.append("MAY_REQUEST")
    if "금지" in query:
        preferred.append("PROHIBITS")
    if "의무" in query:
        preferred.append("HAS_DUTY")
    if "할 수" in query or "권리" in query:
        preferred.append("MAY")
    return preferred
