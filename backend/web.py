import asyncio
import json
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from backend.rag import LegalRAG, RetrievalUnavailable

PROJECT_ROOT = Path(__file__).resolve().parent.parent


rag: LegalRAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = LegalRAG()
    yield


app = FastAPI(lifespan=lifespan)


class Question(BaseModel):
    question: str
    strategy: Literal["vector", "hybrid"] = "hybrid"


def get_rag() -> LegalRAG:
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system is starting.")
    return rag


@app.get("/")
def home() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend" / "index.html")


@app.get("/documents")
def list_documents(rag_system: Annotated[LegalRAG, Depends(get_rag)]) -> dict:
    return {"documents": rag_system.list_documents()}


@app.post("/ask")
async def ask(
    request: Question,
    rag_system: Annotated[LegalRAG, Depends(get_rag)],
) -> StreamingResponse:
    try:
        chunks = rag_system.retrieve(request.question, strategy=request.strategy)
    except RetrievalUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    async def stream():
        sources = [
            {
                "document": chunk.meta["document_name"],
                "citation": citation_number,
                "chunk": chunk.meta["chunk_number"],
                "score": chunk.score,
                "text": chunk.content,
            }
            for citation_number, chunk in enumerate(chunks, start=1)
        ]
        yield json.dumps({"type": "sources", "sources": sources}) + "\n"

        tokens = asyncio.Queue()

        async def send_token(token: str) -> None:
            await tokens.put(token)

        async def generate() -> None:
            try:
                await rag_system.generate_async(
                    request.question,
                    chunks,
                    send_token,
                )
            except Exception as error:
                await tokens.put(error)
            finally:
                await tokens.put(None)

        generation = asyncio.create_task(generate())

        try:
            while (token := await tokens.get()) is not None:
                if isinstance(token, Exception):
                    yield json.dumps({"type": "error"}) + "\n"
                    await generation
                    return
                yield json.dumps({"type": "token", "text": token}) + "\n"

            await generation
            yield json.dumps({"type": "done"}) + "\n"
        finally:
            if not generation.done():
                generation.cancel()
                with suppress(asyncio.CancelledError):
                    await generation

    return StreamingResponse(stream(), media_type="application/x-ndjson")
