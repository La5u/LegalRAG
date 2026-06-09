from dataclasses import replace

from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever # retrieve the closes chunk vector to query vector
from haystack.dataclasses import ChatMessage, Document                          # 
from haystack.document_stores.in_memory import InMemoryDocumentStore            # holds all vectors in RAM (only for prototype)
from haystack_integrations.components.generators.ollama import (
    OllamaChatGenerator,                  # sends messages to ollama and receives answer
)
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "gemma4:12b-it-q4_K_M"


class LegalRAG:
    def __init__(self, top_k: int = 3) -> None:
        self.document_store = InMemoryDocumentStore(
            embedding_similarity_function="cosine"
        )
        self.embedding_model = SentenceTransformer(
            model_name_or_path=EMBEDDING_MODEL,
            local_files_only=True,
        )
        self.retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store,
            top_k=top_k,
        )
        self.generator = OllamaChatGenerator(
            model=GENERATION_MODEL,
            generation_kwargs={"temperature": 0.1, "num_predict": 200},
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
            for chunk, embedding in zip(chunks, chunk_embeddings, strict=True)
        ]

        self.document_store.write_documents(embedded_chunks)

    def retrieve(self, query: str) -> list[Document]:
        """Return the chunks with the highest cosine similarity."""
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()
        retrieval_result = self.retriever.run(query_embedding=query_embedding)
        return retrieval_result["documents"]

    def answer(self, query: str) -> tuple[str, list[Document]]:
        chunks = self.retrieve(query)
        context = "\n\n".join(
            f"[Chunk {chunk.meta['chunk_number']}]\n{chunk.content}"
            for chunk in chunks
        )
        messages = [
            ChatMessage.from_system(
                "Answer only from the supplied U.S. Constitution context. "
                "If the context does not contain the answer, say that you "
                "cannot answer from the retrieved text. Cite chunk numbers."
            ),
            ChatMessage.from_user(
                f"Context:\n{context}\n\nQuestion: {query}"
            ),
        ]
        generation_result = self.generator.run(messages=messages)
        replies = generation_result["replies"]
        answer = replies[0].text or ""
        return answer, chunks
