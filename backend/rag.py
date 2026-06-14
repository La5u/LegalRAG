from dataclasses import replace

from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.generators.ollama import (
    OllamaChatGenerator,
)
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "gemma4:e4b-it-qat"
TOP_K = 3
PARAGRAPH_NEIGHBORS = 1
TEMPERATURE = 0.1
CONTEXT_WINDOW = 2048
MAX_OUTPUT_TOKENS = 150


class LegalRAG:
    def __init__(self, top_k: int = TOP_K) -> None:
        self.top_k = top_k
        self.document_store = InMemoryDocumentStore(
            embedding_similarity_function="cosine"
        )
        self.embedding_model = SentenceTransformer(
            model_name_or_path=EMBEDDING_MODEL,
            local_files_only=True,
        )
        self.retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store,
            top_k=self.top_k,
        )
        self.groups: dict[tuple[str, int], list[Document]] = {}
        self.generator = OllamaChatGenerator(
            model=GENERATION_MODEL,
            url="http://127.0.0.1:11434",
            generation_kwargs={
                "temperature": TEMPERATURE,
                "num_ctx": CONTEXT_WINDOW,
                "num_predict": MAX_OUTPUT_TOKENS,
            },
            keep_alive="1h",
            timeout=600,
        )

    def index(self, chunks: list[Document]) -> None:
        """Embed documents and write them to the in-memory store."""
        chunk_texts = [chunk.content for chunk in chunks]
        chunk_embeddings = self.embedding_model.encode(
            chunk_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embedded_chunks = [
            replace(chunk, embedding=embedding.tolist())
            for chunk, embedding in zip(
                chunks,
                chunk_embeddings,
                strict=True,
            )
        ]

        self.document_store.write_documents(embedded_chunks)
        for chunk in embedded_chunks:
            group = (chunk.meta["document_name"], chunk.meta["group_id"])
            self.groups.setdefault(group, []).append(chunk)

        for parts in self.groups.values():
            parts.sort(key=lambda part: part.meta["part"])

    def save_document_store(self, path: str) -> None:
        """Save indexed documents and embeddings to a JSON file."""
        self.document_store.save_to_disk(path)

    def load_document_store(self, path: str) -> None:
        """Load indexed documents and rebuild the in-memory lookup data."""
        self.document_store = InMemoryDocumentStore.load_from_disk(path)
        self.retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store,
            top_k=self.top_k,
        )
        self.groups = {}

        for chunk in self.document_store.storage.values():
            group = (chunk.meta["document_name"], chunk.meta["group_id"])
            self.groups.setdefault(group, []).append(chunk)

        for parts in self.groups.values():
            parts.sort(key=lambda part: part.meta["part"])

    def list_documents(self) -> list[dict]:
        """Return indexed document names and their chunk counts."""
        counts = {}
        for chunk in self.document_store.storage.values():
            name = chunk.meta["document_name"]
            counts[name] = counts.get(name, 0) + 1

        return [
            {
                "name": name,
                "chunks": chunks,
            }
            for name, chunks in sorted(counts.items())
        ]

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve relevant chunks and include their immediate paragraph neighbors."""
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()
        retrieval_result = self.retriever.run(query_embedding=query_embedding)
        retrieved_chunks = retrieval_result["documents"]
        return self._add_paragraph_neighbors(retrieved_chunks)

    def _add_paragraph_neighbors(
        self,
        retrieved_chunks: list[Document],
    ) -> list[Document]:
        """Add the previous and next chunk when they belong to the same block."""
        context_chunks = []
        added_chunk_numbers = set()
        retrieved_by_number = {
            chunk.meta["chunk_number"]: chunk for chunk in retrieved_chunks
        }

        for chunk in retrieved_chunks:
            group = (chunk.meta["document_name"], chunk.meta["group_id"])
            parts = self.groups[group]
            matched_index = next(
                index
                for index, part in enumerate(parts)
                if part.meta["chunk_number"] == chunk.meta["chunk_number"]
            )
            first_index = max(0, matched_index - PARAGRAPH_NEIGHBORS)
            last_index = min(len(parts), matched_index + PARAGRAPH_NEIGHBORS + 1)

            for part in parts[first_index:last_index]:
                chunk_number = part.meta["chunk_number"]
                if chunk_number in added_chunk_numbers:
                    continue

                added_chunk_numbers.add(chunk_number)
                context_chunks.append(retrieved_by_number.get(chunk_number, part))

        return context_chunks

    def _messages(self, query: str, chunks: list[Document]) -> list[ChatMessage]:
        context = "\n\n".join(
            f"[{citation_number}]\n{chunk.content}"
            for citation_number, chunk in enumerate(chunks, start=1)
        )
        return [
            ChatMessage.from_system(
                "Answer only from the supplied legal-document context. "
                "If the context does not contain the answer, say that you "
                "cannot answer from the retrieved text. Cite each chunk you "
                "use with the exact format [N]. Put the citation directly "
                "after each supported claim. Format multiple citations as "
                "[1][2][3]."
            ),
            ChatMessage.from_user(
                f"Context:\n{context}\n\nQuestion: {query}"
            ),
        ]

    async def generate_async(
        self,
        query: str,
        chunks: list[Document],
        callback,
    ) -> None:
        await self.generator.run_async(
            messages=self._messages(query, chunks),
            streaming_callback=callback,
        )
