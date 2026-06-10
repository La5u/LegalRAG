import re
from dataclasses import replace

from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever # retrieve the closes chunk vector to query vector
from haystack.dataclasses import ChatMessage, Document                          # 
from haystack.document_stores.in_memory import InMemoryDocumentStore            # holds all vectors in RAM (only for prototype)
from haystack_integrations.components.generators.ollama import (
    OllamaChatGenerator,                  # sends messages to ollama and receives answer
)
from sentence_transformers import SentenceTransformer

from backend.ingest import split_sentences

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "gemma4:e4b-it-qat"


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
        self.groups: dict[tuple[str, int], list[Document]] = {}
        self.generator = OllamaChatGenerator(
            model=GENERATION_MODEL,
            generation_kwargs={
                "temperature": 0.1,
                "num_ctx": 4096,
                "num_predict": 150,
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
            for chunk, embedding in zip(chunks, chunk_embeddings, strict=True)
        ]

        self.document_store.write_documents(embedded_chunks)
        for chunk in embedded_chunks:
            group = (chunk.meta["document_name"], chunk.meta["group_id"])
            self.groups.setdefault(group, []).append(chunk)

        for parts in self.groups.values():
            parts.sort(key=lambda part: part.meta["part"])

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve relevant chunks and reconstruct any split blocks."""
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()
        retrieval_result = self.retriever.run(query_embedding=query_embedding)
        retrieved_chunks = retrieval_result["documents"]
        complete_chunks = []
        added_blocks = set()

        for chunk in retrieved_chunks:
            group = (chunk.meta["document_name"], chunk.meta["group_id"])
            if group in added_blocks:
                continue

            added_blocks.add(group)
            parts = self.groups[group]
            if len(parts) == 1:
                complete_chunks.append(chunk)
                continue

            complete_chunks.append(
                replace(
                    chunk,
                    content=" ".join(part.content for part in parts),
                    meta={**chunk.meta, "part": 0},
                )
            )

        return complete_chunks

    def answer(self, query: str) -> tuple[str, list[Document]]:
        chunks = self.retrieve(query)
        context = "\n\n".join(
            f"[{chunk.meta['chunk_number']}]\n{chunk.content}"
            for chunk in chunks
        )
        messages = [
            ChatMessage.from_system(
                "Answer only from the supplied legal-document context. "
                "If the context does not contain the answer, say that you "
                "cannot answer from the retrieved text. Cite each chunk you "
                "use with the exact format [N]. Put the citation directly "
                "after each sentence or claim that the chunk supports."
            ),
            ChatMessage.from_user(
                f"Context:\n{context}\n\nQuestion: {query}"
            ),
        ]
        generation_result = self.generator.run(messages=messages)
        replies = generation_result["replies"]
        answer = replies[0].text or ""
        return answer, chunks

    def supporting_sentence_indexes(
        self,
        answer: str,
        chunk: Document,
    ) -> set[int]:
        """Find chunk sentences that most closely match claims citing the chunk."""
        chunk_number = chunk.meta["chunk_number"]
        citation = f"[{chunk_number}]"
        claims = [
            sentence
            for sentence in split_sentences(answer)
            if citation in sentence
        ]
        sentences = split_sentences(chunk.content)
        if not claims or not sentences:
            return set()

        sentence_words = [
            set(re.findall(r"[A-Za-z]+", sentence.lower()))
            for sentence in sentences
        ]
        highlighted = set()
        for claim in claims:
            claim_words = set(re.findall(r"[A-Za-z]+", claim.lower()))
            highlighted.add(
                max(
                    range(len(sentences)),
                    key=lambda index: len(claim_words & sentence_words[index]),
                )
            )
        return highlighted
