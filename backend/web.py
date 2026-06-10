import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.ingest import chunk_text, load_text, split_sentences
from backend.rag import LegalRAG

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENT_PATH = PROJECT_ROOT / "documents" / "us_constitution.txt"
rag = LegalRAG()
rag.index(
    chunk_text(
        load_text(DOCUMENT_PATH),
        document_name=DOCUMENT_PATH.name,
    )
)
app = FastAPI()


class Question(BaseModel):
    question: str


@app.get("/")
def home() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend" / "index.html")


@app.post("/ask")
def ask(request: Question) -> dict:
    answer, chunks = rag.answer(request.question)
    cited_numbers = {
        int(number) for number in re.findall(r"\[(\d+)\]", answer)
    }
    sources = []

    for chunk in chunks:
        used = chunk.meta["chunk_number"] in cited_numbers
        supporting_indexes = (
            rag.supporting_sentence_indexes(answer, chunk) if used else set()
        )
        sentences = [
            {
                "text": sentence,
                "highlighted": sentence_index in supporting_indexes,
            }
            for sentence_index, sentence in enumerate(
                split_sentences(chunk.content)
            )
        ]
        sources.append(
            {
                "document": chunk.meta["document_name"],
                "number": chunk.meta["chunk_number"],
                "score": chunk.score,
                "used": used,
                "sentences": sentences,
            }
        )

    sources.sort(key=lambda source: not source["used"])
    return {"answer": answer, "sources": sources}
