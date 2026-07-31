import logging

from neo4j import GraphDatabase

from backend.config import (
    CHUNKING_VERSION,
    EMBEDDING_MODEL,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_VECTOR_INDEX,
    TOP_K,
    configure_runtime_environment,
)
from backend.graph_index import (
    delete_orphan_graph_nodes,
    delete_source_paths,
    write_index_transaction,
)
from backend.indexing import (
    document_is_current,
    embedding_hash,
    index_document,
    prune_missing_documents,
)
from backend.document import Document
from backend.llm import stream_answer
from backend.retrieval import RetrievalResult, retrieve_chunks
from backend.schema import setup_schema, setup_vector_index
from backend.sources import SourceMatcher


logger = logging.getLogger(__name__)


SYMBOLIC_ROLE = {
    "definition": ("definitions_checked", "정의"),
    "obligation": ("obligations_checked", "의무"),
    "prohibition": ("prohibitions_checked", "금지"),
    "permission": ("permissions_checked", "허용"),
    "exception": ("exceptions_checked", "예외"),
    "condition": ("conditions_checked", "조건"),
    "scenario_apply": ("conditions_checked", "조건"),
}


def grounded_symbolic_context(
    chunks: list[Document], trace: dict | None,
) -> str | None:
    """Label one retrieved rule passage without predicting a conclusion."""
    if not trace:
        return None
    citation_by_chunk = {
        (chunk.meta.get("document_name"), chunk.meta.get("chunk_number")): number
        for number, chunk in enumerate(chunks, start=1)
    }
    if trace.get("workflow") == "privacy_compliance":
        lines = []
        for check in trace.get("privacy_checklist", []):
            evidence = check.get("evidence") or {}
            citation = citation_by_chunk.get((evidence.get("document"), evidence.get("chunk")))
            rule_status = f"규정 근거 [{citation}]" if citation else "규정 근거 미검색"
            fact_status = "사실 진술됨" if check["fact_status"] == "stated" else "사실 확인 필요"
            lines.append(f"- {check['issue']}: {rule_status}; {fact_status}")
        return (
            "개인정보 컴플라이언스 체크리스트(근거 발견은 적법 결론이 아님):\n"
            + "\n".join(lines)
            + "\n누락 사실이 있으면 적법 여부를 단정하지 마세요."
        )
    role = SYMBOLIC_ROLE.get(trace.get("query_type"))
    if not role:
        return None
    trace_key, label = role
    for item in trace.get(trace_key, []):
        citation = citation_by_chunk.get((item.get("document"), item.get("chunk")))
        if citation:
            return f"근거 조각의 규칙 역할(결론이 아님):\n[{citation}] {label}"
    return None


class LegalRAG:
    def __init__(
        self,
        top_k: int = TOP_K,
        embedding_model=None,
        embedding_dimensions: int | None = None,
    ) -> None:
        configure_runtime_environment()
        self.top_k = top_k
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._vector_index_ready = False
        self._source_matcher = SourceMatcher(self._run_read)
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        try:
            setup_schema(self._run_write)
        except Exception:
            self.driver.close()
            raise

    def close(self) -> None:
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            try:
                self._embedding_model = SentenceTransformer(
                    model_name_or_path=EMBEDDING_MODEL,
                    local_files_only=True,
                )
            except OSError as error:
                logger.exception("Embedding model is not available locally.")
                raise RuntimeError(
                    "Embedding model is not downloaded locally. "
                    f"Run: HF_HOME=.cache/huggingface python -c "
                    f"\"from sentence_transformers import SentenceTransformer; "
                    f"SentenceTransformer('{EMBEDDING_MODEL}')\""
                ) from error
        return self._embedding_model

    @property
    def embedding_dimensions(self) -> int:
        if self._embedding_dimensions is None:
            get_dimensions = getattr(
                self.embedding_model,
                "get_embedding_dimension",
                self.embedding_model.get_sentence_embedding_dimension,
            )
            self._embedding_dimensions = get_dimensions()
        return self._embedding_dimensions

    def _run_write(self, query: str, **parameters) -> None:
        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.run(query, **parameters).consume()

    def _run_read(self, query: str, **parameters) -> list:
        with self.driver.session(database=NEO4J_DATABASE) as session:
            return list(session.run(query, **parameters))

    def document_is_current(self, source_path: str, source_mtime_ns: int) -> bool:
        return document_is_current(self._run_read, source_path, source_mtime_ns)

    def prune_missing_documents(
        self, source_paths: list[str], cleanup_orphans: bool = False,
    ) -> bool:
        records = self._run_read(
            """
            MATCH (document:Document)
            WHERE NOT document.source_path IN $source_paths
            RETURN count(document) > 0 AS removed_documents
            """,
            source_paths=source_paths,
        )
        removed_documents = bool(records and records[0]["removed_documents"])
        prune_missing_documents(self._run_read, self._run_write, source_paths)
        if cleanup_orphans or removed_documents:
            delete_orphan_graph_nodes(self._run_write)
        self._source_matcher.invalidate()
        return removed_documents

    def delete_document(self, source_path: str) -> None:
        delete_source_paths(self._run_write, [source_path])
        delete_orphan_graph_nodes(self._run_write)
        self._source_matcher.invalidate()

    def index(
        self,
        chunks: list[Document],
        source_path: str,
        source_mtime_ns: int,
    ) -> None:
        cached_embeddings = self._cached_embeddings(
            source_path, source_mtime_ns,
        )
        if not self._vector_index_ready:
            dimensions = (
                len(next(iter(cached_embeddings.values())))
                if cached_embeddings
                else self.embedding_dimensions
            )
            setup_vector_index(
                self._run_write,
                self._run_read,
                NEO4J_VECTOR_INDEX,
                dimensions,
            )
            self._vector_index_ready = True
        required_hashes = {embedding_hash(chunk) for chunk in chunks}
        embedding_model = (
            None
            if required_hashes.issubset(cached_embeddings)
            else self.embedding_model
        )
        batch = index_document(
            embedding_model,
            chunks,
            source_path,
            source_mtime_ns,
            cached_embeddings,
        )

        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.execute_write(
                write_index_transaction,
                source_path,
                source_mtime_ns,
                batch,
            )
        self._source_matcher.invalidate()
        logger.info("Finished indexing %s.", source_path)

    def _cached_embeddings(
        self, source_path: str, source_mtime_ns: int,
    ) -> dict[str, list[float]]:
        records = self._run_read(
            """
            MATCH (document:Document {source_path: $source_path})-[:CONTAINS]->(chunk:Chunk)
            WHERE document.mtime_ns = $source_mtime_ns
                AND document.embedding_model = $embedding_model
                AND document.chunking_version = $chunking_version
                AND chunk.embedding_hash IS NOT NULL
                AND chunk.embedding IS NOT NULL
            RETURN chunk.embedding_hash AS embedding_hash,
                chunk.embedding AS embedding
            """,
            source_path=source_path,
            source_mtime_ns=source_mtime_ns,
            embedding_model=EMBEDDING_MODEL,
            chunking_version=CHUNKING_VERSION,
        )
        return {
            record["embedding_hash"]: record["embedding"]
            for record in records
        }

    def list_documents(self) -> list[dict]:
        records = self._run_read(
            """
            MATCH (document:Document)
            OPTIONAL MATCH (chunk:Chunk)
            WHERE chunk.source_path = document.source_path
            RETURN document.name AS name, count(chunk) AS chunks
            ORDER BY name
            """
        )
        return [
            {
                "name": record["name"],
                "chunks": record["chunks"],
            }
            for record in records
        ]

    def retrieve(
        self, query: str, strategy: str = "graph", category: str = "auto",
    ) -> RetrievalResult:
        return retrieve_chunks(
            self._run_read,
            self.embedding_model,
            self.top_k,
            query,
            self._source_matcher,
            strategy,
            category,
        )

    async def generate_async(
        self,
        query: str,
        result: RetrievalResult,
        callback,
        hosted_model: str | None = None,
        api_key: str | None = None,
        on_status=None,
    ) -> None:
        if not result.evidence:
            message = (
                "검색된 문서에서 질문을 뒷받침할 근거를 찾지 못했습니다."
                if any("가" <= character <= "힣" for character in query)
                else "No supporting evidence was found in the indexed documents."
            )
            await callback(message)
            return
        context = format_evidence_context(result)
        symbolic_context = grounded_symbolic_context(
            result.evidence, result.symbolic_trace,
        )
        generation_options = {"symbolic_context": symbolic_context, "on_status": on_status}
        if hosted_model:
            generation_options.update(
                hosted_model=hosted_model,
                api_key=api_key,
            )
        await stream_answer(query, context, callback, **generation_options)


def format_evidence_context(result: RetrievalResult) -> str:
    """Format retrieved evidence as compact, numbered answer context."""
    facts_by_chunk = {}
    for fact in result.graph_facts:
        facts_by_chunk.setdefault((fact.document, fact.chunk), []).append(fact)

    entries = []
    for citation_number, chunk in enumerate(result.evidence, start=1):
        label = ", ".join(
            value
            for value in (
                chunk.meta.get("document_title")
                or chunk.meta.get("document_name"),
                chunk.meta.get("version_label"),
                chunk.meta.get("specific_citation")
                or chunk.meta.get("article_number"),
            )
            if value
        )
        parts = [
            part for part in (chunk.meta.get("heading_path"), chunk.content) if part
        ]
        matching_facts = facts_by_chunk.get(
            (
                chunk.meta.get("document_name"),
                chunk.meta.get("chunk_number"),
            ),
            [],
        )
        parts.extend(
            f"구조화된 관계: {fact.path}\n근거 문장: {fact.sentence}"
            for fact in matching_facts
        )
        entries.append(f"[{citation_number}] {label}\n" + "\n".join(parts))
    return "\n\n".join(entries)
