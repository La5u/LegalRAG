import logging

from neo4j import GraphDatabase

from backend.config import (
    EMBEDDING_MODEL,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_VECTOR_INDEX,
    TOP_K,
    configure_runtime_environment,
    normalize_retrieval_strategy,
)
from backend.indexing import (
    delete_source_paths,
    document_is_current,
    index_document,
    prune_missing_documents,
    write_index_transaction,
)
from backend.ingest import Document
from backend.llm import stream_answer
from backend.reasoning import prioritize_symbolic_evidence, symbolic_prompt
from backend.retrieval import attach_symbolic_trace, filter_by_category, retrieve_graphrag, retrieve_vectorrag
from backend.schema import setup_schema, setup_vector_index
from backend.sources import SourceMatcher


logger = logging.getLogger(__name__)


def _compact_chunk(chunk: Document) -> str:
    content = chunk.content
    document_name = chunk.meta.get("document_name", "")
    if document_name and content.startswith(document_name):
        content = content[len(document_name):].lstrip("\n")
    return content


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
            self._embedding_dimensions = (
                self.embedding_model.get_sentence_embedding_dimension()
            )
        return self._embedding_dimensions

    def _run_write(self, query: str, **parameters) -> None:
        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.run(query, **parameters).consume()

    def _run_read(self, query: str, **parameters) -> list:
        with self.driver.session(database=NEO4J_DATABASE) as session:
            return list(session.run(query, **parameters))

    def document_is_current(self, source_path: str, source_mtime_ns: int) -> bool:
        return document_is_current(self._run_read, source_path, source_mtime_ns)

    def prune_missing_documents(self, source_paths: list[str]) -> None:
        prune_missing_documents(self._run_read, self._run_write, source_paths)

    def delete_document(self, source_path: str) -> None:
        delete_source_paths(self._run_write, [source_path])

    def index(
        self,
        chunks: list[Document],
        source_path: str,
        source_mtime_ns: int,
    ) -> None:
        logger.info("Indexing %s with %s chunks.", source_path, len(chunks))
        document_name = (
            chunks[0].meta.get("document_name", source_path) if chunks else source_path
        )
        setup_vector_index(
            self._run_write,
            self._run_read,
            NEO4J_VECTOR_INDEX,
            self.embedding_dimensions,
        )

        index_rows = index_document(
            self._run_write,
            self._run_read,
            self.embedding_model,
            chunks,
            source_path,
            source_mtime_ns,
            self.embedding_dimensions,
        )

        logger.info(
            "Prepared graph rows for %s: %s legal units, %s definitions, "
            "%s triples, %s connections, %s holdings.",
            source_path,
            len(index_rows["legal_unit_rows"]),
            len(index_rows["definition_rows"]),
            len(index_rows["triple_rows"]),
            len(index_rows["connection_rows"]),
            len(index_rows["holding_rows"]),
        )

        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.execute_write(
                write_index_transaction,
                source_path,
                document_name,
                source_mtime_ns,
                index_rows["chunk_rows"],
                index_rows["legal_unit_rows"],
                index_rows["definition_rows"],
                index_rows["triple_rows"],
                index_rows["connection_rows"],
                index_rows["case_row"],
                index_rows["holding_rows"],
            )
        logger.info("Finished indexing %s.", source_path)

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

    def retrieve(self, query: str, strategy: str = "graph", category: str = "auto") -> list[Document]:
        strategy = normalize_retrieval_strategy(strategy)
        if strategy == "vector":
            chunks = retrieve_vectorrag(self._run_read, self.embedding_model, self.top_k, query)
        elif strategy == "graph":
            chunks = retrieve_graphrag(
                self._run_read, self.embedding_model, self.top_k, query, self._source_matcher
            )
        elif strategy == "neuro_symbolic":
            chunks = attach_symbolic_trace(
                query,
                retrieve_graphrag(
                    self._run_read, self.embedding_model, self.top_k, query, self._source_matcher
                ),
            )
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
        resolved = category if category not in ("all", "auto") else None
        return filter_by_category(chunks, resolved)

    async def generate_async(
        self,
        query: str,
        chunks: list[Document],
        callback,
    ) -> None:
        trace = next(
            (
                chunk.meta.get("symbolic_trace")
                for chunk in chunks
                if chunk.meta.get("symbolic_trace")
            ),
            None,
        )
        chunks = prioritize_symbolic_evidence(chunks, trace)
        context = "\n\n".join(
            f"[{citation_number}]\n{_compact_chunk(chunk)}"
            for citation_number, chunk in enumerate(chunks, start=1)
        )
        await stream_answer(
            query,
            context,
            callback,
            symbolic_context=symbolic_prompt(trace),
        )
