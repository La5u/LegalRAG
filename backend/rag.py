import logging
import re

from neo4j import GraphDatabase

from backend.config import (
    CHUNKING_VERSION,
    DEFAULT_MAX_WORDS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_VECTOR_INDEX,
    PARAGRAPH_NEIGHBORS,
    TOP_K,
    configure_runtime_environment,
)
from backend.connections import extract_connections
from backend.definitions import extract_definitions
from backend.graph import (
    build_legal_unit_rows,
    graph_fact_documents,
    write_connections,
    write_triples,
)
from backend.ingest import Document
from backend.llm import stream_answer
from backend.records import chunk_return_fields, document_from_record
from backend.schema import setup_schema, setup_vector_index, vector_index_exists
from backend.triples import extract_triples


logger = logging.getLogger(__name__)


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


class RetrievalUnavailable(RuntimeError):
    """Raised when retrieval cannot run because the index is not ready."""


def should_expand_legal_units(query: str) -> bool:
    """Return True when a question explicitly asks for legal-unit context."""
    normalized_query = query.casefold()
    return bool(LEGAL_UNIT_QUERY_PATTERN.search(query)) or any(
        term in normalized_query
        for term in LEGAL_UNIT_QUERY_TERMS
    )


class LegalRAG:
    def __init__(self, top_k: int = TOP_K) -> None:
        configure_runtime_environment()
        self.top_k = top_k
        self._embedding_model = None
        self._embedding_dimensions = None
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        setup_schema(self._run_write)

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
            if hasattr(self.embedding_model, "get_embedding_dimension"):
                self._embedding_dimensions = (
                    self.embedding_model.get_embedding_dimension()
                )
            else:
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
        records = self._run_read(
            """
            MATCH (document:Document {source_path: $source_path})
            WITH document
            WHERE document.mtime_ns IS NOT NULL
                AND document.embedding_model IS NOT NULL
                AND document.chunking_version IS NOT NULL
                AND document.max_words IS NOT NULL
            RETURN document.mtime_ns = $source_mtime_ns
                AND document.embedding_model = $embedding_model
                AND document.chunking_version = $chunking_version
                AND document.max_words = $max_words
                AS is_current
            """,
            source_path=source_path,
            source_mtime_ns=source_mtime_ns,
            embedding_model=EMBEDDING_MODEL,
            chunking_version=CHUNKING_VERSION,
            max_words=DEFAULT_MAX_WORDS,
        )
        return bool(records and records[0]["is_current"])

    def prune_missing_documents(self, source_paths: list[str]) -> None:
        self._run_write(
            """
            MATCH (document:Document)
            WHERE NOT document.source_path IN $source_paths
            WITH collect(document) AS documents,
                collect(document.source_path) AS deleted_source_paths
            OPTIONAL MATCH (:Concept)-[legal_relation]->(:Concept)
            WHERE legal_relation.source_path IN deleted_source_paths
            WITH documents, deleted_source_paths, collect(legal_relation) AS legal_relations
            FOREACH (legal_relation IN legal_relations | DELETE legal_relation)
            WITH documents, deleted_source_paths
            OPTIONAL MATCH (chunk:Chunk)
            WHERE chunk.source_path IN deleted_source_paths
            WITH documents, deleted_source_paths, collect(chunk) AS chunks
            OPTIONAL MATCH (definition:Definition)-[:SUPPORTED_BY]->(chunk:Chunk)
            WHERE chunk.source_path IN deleted_source_paths
            WITH documents, deleted_source_paths, chunks, collect(definition) AS definitions
            OPTIONAL MATCH (legal_unit:LegalUnit)
            WHERE legal_unit.source_path IN deleted_source_paths
            WITH documents, chunks, definitions, collect(legal_unit) AS legal_units
            FOREACH (definition IN definitions | DETACH DELETE definition)
            FOREACH (chunk IN chunks | DETACH DELETE chunk)
            FOREACH (legal_unit IN legal_units | DETACH DELETE legal_unit)
            FOREACH (document IN documents | DETACH DELETE document)
            WITH 1 AS _
            MATCH (term:Term)
            WHERE NOT (term)--()
            DETACH DELETE term
            WITH 1 AS _
            MATCH (concept:Concept)
            WHERE NOT (concept)--()
            DETACH DELETE concept
            """,
            source_paths=source_paths,
        )

    def delete_document(self, source_path: str) -> None:
        self._run_write(
            """
            MATCH (document:Document {source_path: $source_path})
            OPTIONAL MATCH (:Concept)-[legal_relation]->(:Concept)
            WHERE legal_relation.source_path = document.source_path
            WITH document, collect(legal_relation) AS legal_relations
            FOREACH (legal_relation IN legal_relations | DELETE legal_relation)
            WITH document
            OPTIONAL MATCH (chunk:Chunk)
            WHERE chunk.source_path = document.source_path
            WITH document, collect(chunk) AS chunks
            OPTIONAL MATCH (definition:Definition)-[:SUPPORTED_BY]->(chunk:Chunk)
            WHERE chunk.source_path = document.source_path
            WITH document, chunks, collect(definition) AS definitions
            OPTIONAL MATCH (legal_unit:LegalUnit)
            WHERE legal_unit.source_path = document.source_path
            WITH document, chunks, definitions, collect(legal_unit) AS legal_units
            FOREACH (definition IN definitions | DETACH DELETE definition)
            FOREACH (chunk IN chunks | DETACH DELETE chunk)
            FOREACH (legal_unit IN legal_units | DETACH DELETE legal_unit)
            DETACH DELETE document
            WITH 1 AS _
            MATCH (term:Term)
            WHERE NOT (term)--()
            DETACH DELETE term
            WITH 1 AS _
            MATCH (concept:Concept)
            WHERE NOT (concept)--()
            DETACH DELETE concept
            """,
            source_path=source_path,
        )

    def index(
        self,
        chunks: list[Document],
        source_path: str,
        source_mtime_ns: int,
    ) -> None:
        """Embed documents and write chunks to Neo4j."""
        logger.info("Indexing %s with %s chunks.", source_path, len(chunks))
        self.delete_document(source_path)
        document_name = chunks[0].meta["document_name"] if chunks else source_path
        setup_vector_index(
            self._run_write,
            self._run_read,
            NEO4J_VECTOR_INDEX,
            self.embedding_dimensions,
        )

        chunk_texts = [chunk.content for chunk in chunks]
        if chunk_texts:
            chunk_embeddings = self.embedding_model.encode(
                chunk_texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            chunk_embeddings = []

        self._run_write(
            """
            MERGE (document:Document {source_path: $source_path})
            SET
                document.name = $document_name,
                document.mtime_ns = $source_mtime_ns,
                document.embedding_model = $embedding_model,
                document.chunking_version = $chunking_version,
                document.max_words = $max_words,
                document.chunk_count = $chunk_count,
                document.indexed_at = datetime()
            """,
            source_path=source_path,
            document_name=document_name,
            source_mtime_ns=source_mtime_ns,
            embedding_model=EMBEDDING_MODEL,
            chunking_version=CHUNKING_VERSION,
            max_words=DEFAULT_MAX_WORDS,
            chunk_count=len(chunks),
        )

        if not chunks:
            logger.info("Finished indexing %s with no chunks.", source_path)
            return

        # Deterministic structure graph:
        # 법령/document -> LegalUnit nodes -> 문서조각/chunk.
        # "Deterministic" means we build it from article headings like
        # "제15조(개인정보의 수집ㆍ이용)", not from an LLM guess.
        legal_unit_rows = []
        chunk_rows = []
        chunk_article_ids = {}
        for chunk, embedding in zip(chunks, chunk_embeddings, strict=True):
            legal_units = build_legal_unit_rows(
                chunk,
                source_path,
                document_name,
                legal_unit_rows,
            )
            chunk_article_ids[chunk.meta["chunk_number"]] = (
                legal_units["ids"].get("article")
            )
            chunk_rows.append(
                {
                    "id": f"{source_path}::chunk-{chunk.meta['chunk_number']}",
                    "content": chunk.content,
                    "embedding": (
                        embedding.tolist()
                        if hasattr(embedding, "tolist")
                        else list(embedding)
                    ),
                    "chunk_number": chunk.meta["chunk_number"],
                    "document_name": chunk.meta["document_name"],
                    "group_id": chunk.meta["group_id"],
                    "part": chunk.meta["part"],
                    "source_path": source_path,
                    "legal_unit_id": legal_units["deepest_id"],
                    "chapter_id": legal_units["ids"].get("chapter"),
                    "chapter_number": chunk.meta.get("chapter_number"),
                    "chapter_title": chunk.meta.get("chapter_title"),
                    "chapter_order": chunk.meta.get("chapter_order"),
                    "section_id": legal_units["ids"].get("section"),
                    "section_number": chunk.meta.get("section_number"),
                    "section_title": chunk.meta.get("section_title"),
                    "section_order": chunk.meta.get("section_order"),
                    "article_id": legal_units["ids"].get("article"),
                    "article_number": chunk.meta.get("article_number"),
                    "article_title": chunk.meta.get("article_title"),
                    "article_order": chunk.meta.get("article_order"),
                }
            )
        definition_rows = [
            {
                "id": (
                    f"{source_path}::definition-"
                    f"{definition.chunk_number}-{definition.term}"
                ),
                "term": definition.term,
                "term_id": f"{source_path}::term-{definition.term}",
                "text": definition.text,
                "chunk_id": f"{source_path}::chunk-{definition.chunk_number}",
                "chunk_number": definition.chunk_number,
                "article_id": chunk_article_ids.get(definition.chunk_number),
                "article_number": definition.article_number,
            }
            for definition in extract_definitions(chunks)
        ]
        triple_rows = extract_triples(
            [Document(row["content"], row) for row in chunk_rows]
        )
        connection_rows = extract_connections(
            [Document(row["content"], row) for row in chunk_rows]
        )
        logger.info(
            "Prepared graph rows for %s: %s legal units, %s definitions, "
            "%s triples, %s connections.",
            source_path,
            len(legal_unit_rows),
            len(definition_rows),
            len(triple_rows),
            len(connection_rows),
        )

        self._run_write(
            """
            UNWIND $chunk_rows AS row
            CREATE (chunk:Chunk {
                id: row.id,
                content: row.content,
                embedding: row.embedding,
                chunk_number: row.chunk_number,
                document_name: row.document_name,
                group_id: row.group_id,
                part: row.part,
                source_path: row.source_path,
                legal_unit_id: row.legal_unit_id,
                chapter_id: row.chapter_id,
                chapter_number: row.chapter_number,
                chapter_title: row.chapter_title,
                chapter_order: row.chapter_order,
                section_id: row.section_id,
                section_number: row.section_number,
                section_title: row.section_title,
                section_order: row.section_order,
                article_id: row.article_id,
                article_number: row.article_number,
                article_title: row.article_title,
                article_order: row.article_order
            })
            """,
            chunk_rows=chunk_rows,
        )
        self._run_write(
            """
            // Cypher is Neo4j's graph query language.
            // This query turns parsed headings into generic LegalUnit nodes.
            MATCH (document:Document {source_path: $source_path})
            UNWIND $legal_unit_rows AS row
            MERGE (legal_unit:LegalUnit {id: row.id})
            SET
                legal_unit.kind = row.kind,
                legal_unit.number = row.number,
                legal_unit.title = row.title,
                legal_unit.order = row.order,
                legal_unit.source_path = row.source_path,
                legal_unit.document_name = row.document_name
            WITH document, collect(row) AS rows
            UNWIND rows AS row
            MATCH (legal_unit:LegalUnit {id: row.id})
            FOREACH (_ IN CASE WHEN row.parent_id IS NULL THEN [1] ELSE [] END |
                MERGE (document)-[:CONTAINS]->(legal_unit)
            )
            WITH row, legal_unit
            OPTIONAL MATCH (parent:LegalUnit {id: row.parent_id})
            FOREACH (_ IN CASE WHEN parent IS NULL THEN [] ELSE [1] END |
                MERGE (parent)-[:CONTAINS]->(legal_unit)
            )
            """,
            source_path=source_path,
            legal_unit_rows=legal_unit_rows,
        )
        self._run_write(
            """
            // This query connects each chunk to the smallest detected LegalUnit.
            UNWIND $chunk_rows AS row
            MATCH (chunk:Chunk {id: row.id})
            OPTIONAL MATCH (legal_unit:LegalUnit {id: row.legal_unit_id})
            FOREACH (_ IN CASE WHEN legal_unit IS NULL THEN [] ELSE [1] END |
                MERGE (legal_unit)-[:HAS_CHUNK]->(chunk)
            )
            """,
            chunk_rows=chunk_rows,
        )
        self._run_write(
            """
            // This query turns deterministic legal terms and definitions into graph nodes.
            UNWIND $definition_rows AS row
            MERGE (term:Term {id: row.term_id})
            SET
                term.name = row.term
            MERGE (definition:Definition {id: row.id})
            SET
                definition.text = row.text,
                definition.chunk_number = row.chunk_number,
                definition.article_number = row.article_number
            MERGE (term)-[:DEFINED_AS]->(definition)
            WITH row, definition
            MATCH (chunk:Chunk {id: row.chunk_id})
            MERGE (definition)-[:SUPPORTED_BY]->(chunk)
            WITH row, definition
            OPTIONAL MATCH (article:LegalUnit {id: row.article_id, kind: "article"})
            FOREACH (_ IN CASE WHEN article IS NULL THEN [] ELSE [1] END |
                MERGE (definition)-[:DEFINED_IN]->(article)
            )
            """,
            definition_rows=definition_rows,
        )
        write_triples(self._run_write, triple_rows)
        write_connections(self._run_write, connection_rows)
        logger.info("Finished indexing %s.", source_path)

    def list_documents(self) -> list[dict]:
        """Return indexed document names and their chunk counts."""
        records = self._run_read(
            f"""
            MATCH (document:Document)
            OPTIONAL MATCH (chunk:Chunk {source_path: document.source_path})
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

    def retrieve(self, query: str, strategy: str = "hybrid") -> list[Document]:
        """Retrieve relevant chunks with the selected retrieval strategy."""
        if strategy == "vector":
            return self._retrieve_vector_chunks(query)
        if strategy == "hybrid":
            return self._retrieve_graphrag(query)
        raise ValueError(f"Unknown retrieval strategy: {strategy}")

    def _retrieve_graphrag(self, query: str) -> list[Document]:
        chunks = self._add_paragraph_neighbors(self._retrieve_vector_chunks(query))
        if should_expand_legal_units(query):
            chunks = self._expand_legal_units(chunks)
        graph_facts = graph_fact_documents(self._run_read, query, chunks)
        return self._merge_graph_facts(chunks, graph_facts)

    def _merge_graph_facts(
        self,
        chunks: list[Document],
        graph_facts: list[Document],
    ) -> list[Document]:
        seen_fact_keys = set()
        ranked_facts = []
        for graph_fact in sorted(
            graph_facts,
            key=lambda document: document.score or 0,
            reverse=True,
        ):
            key = (
                graph_fact.meta.get("source_path"),
                graph_fact.content,
            )
            if key in seen_fact_keys:
                continue
            seen_fact_keys.add(key)
            ranked_facts.append(graph_fact)

        fact_limit = max(1, self.top_k)
        leading_chunks = min(2, len(chunks))
        return (
            chunks[:leading_chunks]
            + ranked_facts[:fact_limit]
            + chunks[leading_chunks:]
        )

    def _retrieve_vector_chunks(self, query: str) -> list[Document]:
        """Retrieve chunks by vector similarity only."""
        if not vector_index_exists(self._run_read, NEO4J_VECTOR_INDEX):
            raise RetrievalUnavailable("No vector index exists. Run indexing first.")

        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()
        records = self._run_read(
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
            top_k=self.top_k,
            query_embedding=query_embedding,
        )
        return [document_from_record(record) for record in records]

    def _add_paragraph_neighbors(
        self,
        retrieved_chunks: list[Document],
    ) -> list[Document]:
        """Add the previous and next chunk when they belong to the same block."""
        windows = [
            {
                "rank": rank,
                "document_name": chunk.meta["document_name"],
                "group_id": chunk.meta["group_id"],
                "first_part": max(0, chunk.meta["part"] - PARAGRAPH_NEIGHBORS),
                "last_part": chunk.meta["part"] + PARAGRAPH_NEIGHBORS,
            }
            for rank, chunk in enumerate(retrieved_chunks)
        ]
        records = self._run_read(
            f"""
            UNWIND $windows AS window
            MATCH (part:Chunk {{
                document_name: window.document_name,
                group_id: window.group_id
            }})
            WHERE part.part >= window.first_part AND part.part <= window.last_part
            RETURN DISTINCT
                {chunk_return_fields("part", "null", "window.rank")}
            ORDER BY rank, part
            """,
            windows=windows,
        )
        return self._merge_ranked_expansion(retrieved_chunks, records)

    def _expand_legal_units(self, retrieved_chunks: list[Document]) -> list[Document]:
        """Expand vector hits to chunks from the closest available LegalUnit."""
        unit_windows = [
            {
                "rank": rank,
                "legal_unit_id": (
                    chunk.meta.get("article_id")
                    or chunk.meta.get("section_id")
                    or chunk.meta.get("chapter_id")
                ),
            }
            for rank, chunk in enumerate(retrieved_chunks)
            if (
                chunk.meta.get("article_id")
                or chunk.meta.get("section_id")
                or chunk.meta.get("chapter_id")
            )
        ]
        if not unit_windows:
            return retrieved_chunks

        records = self._run_read(
            f"""
            UNWIND $unit_windows AS window
            MATCH (legal_unit:LegalUnit)-[:HAS_CHUNK]->(chunk:Chunk)
            WHERE legal_unit.id = window.legal_unit_id
            RETURN
                {chunk_return_fields("chunk", "null", "window.rank")}
            ORDER BY rank, chunk_number
            """,
            unit_windows=unit_windows,
        )
        return self._merge_ranked_expansion(retrieved_chunks, records)

    def _merge_ranked_expansion(
        self,
        retrieved_chunks: list[Document],
        records: list,
    ) -> list[Document]:
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

    async def generate_async(
        self,
        query: str,
        chunks: list[Document],
        callback,
    ) -> None:
        context = "\n\n".join(
            f"[{citation_number}]\n{chunk.content}"
            for citation_number, chunk in enumerate(chunks, start=1)
        )
        await stream_answer(query, context, callback)
