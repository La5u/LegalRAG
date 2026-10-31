import asyncio
import hmac
import json
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.rag import LegalRAG
from backend.config import ALLOW_HOSTED_MODELS, LLM_PROVIDER, PROXY_SECRET
from backend.llm import close_http_client
from backend.retrieval import RetrievalUnavailable
from backend.routing import route_reason
from backend.slurm_vllm import SlurmGpuBusy, SlurmStartupTimeout, StatusUpdate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DOCUMENTS_PATH = PROJECT_ROOT / "documents" / "raw"

CATEGORIES = [
    "all",
    "statute",
    "regulation",
    "precedent",
    "constitutional_decision",
    "contract",
    "guidance",
    "qna",
]


def require_proxy_secret(request: Request) -> None:
    if not PROXY_SECRET:
        return
    supplied = request.headers.get("X-Internal-Proxy-Secret", "")
    if not hmac.compare_digest(supplied, PROXY_SECRET):
        raise HTTPException(status_code=403, detail="Invalid proxy credentials.")


def generation_error_code(error: Exception) -> str:
    if isinstance(error, SlurmGpuBusy):
        return "gpu_busy"
    if isinstance(error, SlurmStartupTimeout):
        return "provider_unavailable"
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status in {401, 403}:
            return "invalid_api_key"
        if status == 429:
            return "rate_limit"
        if status == 400:
            return "provider_rejected"
    if isinstance(error, httpx.TransportError):
        return "provider_unavailable"
    return "generation_failed"


def _safe_raw_document_path(source_path: str) -> Path | None:
    path = (PROJECT_ROOT / source_path).resolve()
    try:
        path.relative_to(RAW_DOCUMENTS_PATH)
    except ValueError:
        return None
    return path


def document_pdf_url(
    source_path: str | None, pdf_path: str | None = None,
) -> str | None:
    selected_path = pdf_path
    if not selected_path and source_path and source_path.casefold().endswith(".pdf"):
        selected_path = source_path
    if not selected_path:
        return None

    path = _safe_raw_document_path(selected_path)
    if path is None:
        return None

    if path.suffix.casefold() != ".pdf" or not path.is_file():
        return None

    relative_path = path.relative_to(RAW_DOCUMENTS_PATH).as_posix()
    # A relative URL works both at the local root and behind a /legal-rag proxy.
    return f"documents/raw/{quote(relative_path, safe='/')}"


def graph_fact_rows(facts) -> list[dict]:
    return [
        {
            "document": fact.document,
            "raw_document": fact.source_path,
            "chunk": fact.chunk,
            "article": fact.article,
            "relation": fact.relation,
            "path": fact.path,
            "sentence": fact.sentence,
            "score": fact.score,
        }
        for fact in facts
    ]


def source_rows(chunks) -> list[dict]:
    return [
        {
            "document": chunk.meta.get("document_name", ""),
            "raw_document": chunk.meta.get("source_path", ""),
            "document_title": chunk.meta.get("document_title"),
            "version_label": chunk.meta.get("version_label"),
            "document_type": chunk.meta.get("document_type"),
            "source_url": chunk.meta.get("source_url"),
            "page_number": chunk.meta.get("page_number"),
            "pdf_url": document_pdf_url(
                chunk.meta.get("source_path"), chunk.meta.get("pdf_path")
            ),
            "citation": citation_number,
            "chunk": chunk.meta.get("chunk_number"),
            "score": chunk.score,
            "text": chunk.content,
            "heading_path": chunk.meta.get("heading_path"),
            "article": chunk.meta.get("article_number"),
            "article_title": chunk.meta.get("article_title"),
            "legal_unit_kind": chunk.meta.get("legal_unit_kind"),
            "legal_unit_number": chunk.meta.get("legal_unit_number"),
            "legal_unit_title": chunk.meta.get("legal_unit_title"),
            "specific_citation": chunk.meta.get("specific_citation"),
        }
        for citation_number, chunk in enumerate(chunks, start=1)
    ]


def stream_event(event_type: str, **payload) -> str:
    return json.dumps({"type": event_type, **payload}) + "\n"


def add_trace_citations(
    trace_payload: dict | None,
    sources: list[dict],
    category: str,
) -> dict | None:
    if not trace_payload:
        return None

    citation_by_source = {
        (source["document"], source["chunk"]): source["citation"]
        for source in sources
    }
    trace = dict(trace_payload)
    trace["important_evidence"] = [
        {
            **item,
            "citation": citation_by_source.get((item.get("document"), item.get("chunk"))),
        }
        for item in trace.get("important_evidence", [])
    ]
    trace["privacy_checklist"] = [
        {
            **check,
            "citation": citation_by_source.get(
                (
                    (check.get("evidence") or {}).get("document"),
                    (check.get("evidence") or {}).get("chunk"),
                )
            ),
        }
        for check in trace.get("privacy_checklist", [])
    ]
    return compact_trace(trace, category)


def compact_trace(trace: dict, category: str) -> dict:
    query_type = trace.get("query_type")
    compact = {
        "query_type": query_type,
        "category": category,
        "route_reason": route_reason(query_type),
        "important_evidence": [
            {"citation": item.get("citation")}
            for item in trace.get("important_evidence", [])
            if item.get("citation")
        ],
        "conclusion": {
            "answer_type": trace.get("conclusion", {}).get("answer_type"),
        },
    }
    if trace.get("workflow") == "privacy_compliance":
        compact["workflow"] = trace["workflow"]
        compact["privacy_checklist"] = [
            {
                "issue": check.get("issue"),
                "rule_status": check.get("rule_status"),
                "fact_status": check.get("fact_status"),
                "citation": check.get("citation"),
            }
            for check in trace.get("privacy_checklist", [])
        ]
        compact["missing_facts"] = trace.get("missing_facts", [])
    return compact


@asynccontextmanager
async def lifespan(app: FastAPI):
    if LLM_PROVIDER == "openai_compatible":
        logging.getLogger(__name__).warning(
            "LEGALRAG_LLM_PROVIDER=openai_compatible: questions and evidence "
            "leave the machine and are sent to the configured LLM endpoint."
        )
    app.state.rag = LegalRAG()
    try:
        yield
    finally:
        app.state.rag.close()
        await close_http_client()


app = FastAPI(
    lifespan=lifespan,
    dependencies=[Depends(require_proxy_secret)],
)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    generation_model: Literal[
        "local", "gemini-3.1-flash-lite", "qwen/qwen3.6-27b",
    ] = "local"
    api_key: str | None = Field(default=None, max_length=4096)
    strategy: Literal["vector", "graph", "neuro_symbolic"] = "graph"
    category: Literal[
        "auto", "all", "statute", "regulation", "precedent",
        "constitutional_decision", "contract", "guidance", "qna",
    ] = "auto"


def get_rag(request: Request) -> LegalRAG:
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system is starting.")
    return rag


@app.get("/")
def home() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend" / "index.html")


@app.get("/config")
def public_config() -> dict:
    return {"allow_hosted_models": ALLOW_HOSTED_MODELS}


@app.get("/documents")
def list_documents(rag_system: Annotated[LegalRAG, Depends(get_rag)]) -> dict:
    return {"documents": rag_system.list_documents()}


@app.get("/documents/raw/{document_path:path}")
def raw_document(document_path: str, download: bool = False) -> FileResponse:
    path = (RAW_DOCUMENTS_PATH / document_path).resolve()
    try:
        path.relative_to(RAW_DOCUMENTS_PATH)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="Document not found.") from error

    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="Document not found.")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=path.name,
        content_disposition_type="attachment" if download else "inline",
    )


@app.post("/ask")
async def ask(
    request: Question,
    rag_system: Annotated[LegalRAG, Depends(get_rag)],
) -> StreamingResponse:
    if request.generation_model != "local" and not ALLOW_HOSTED_MODELS:
        raise HTTPException(
            status_code=403,
            detail="Hosted models are disabled. Use local generation or ask an "
            "administrator to enable LEGALRAG_ALLOW_HOSTED_MODELS.",
        )
    if request.generation_model != "local" and not (
        request.api_key and request.api_key.strip()
    ):
        raise HTTPException(
            status_code=422,
            detail="An API key is required for hosted models.",
        )
    try:
        result = await asyncio.to_thread(
            rag_system.retrieve,
            request.question,
            strategy=request.strategy,
            category=request.category,
        )
    except RetrievalUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    async def stream():
        trace_payload = result.symbolic_trace
        source_chunks = result.evidence
        sources = source_rows(source_chunks)
        graph_facts = graph_fact_rows(result.graph_facts)
        trace_payload = add_trace_citations(trace_payload, sources, request.category)
        yield stream_event("sources", sources=sources)
        if graph_facts:
            yield stream_event("graph_facts", graph_facts=graph_facts)
        if trace_payload:
            yield stream_event("trace", trace=trace_payload)

        tokens = asyncio.Queue(maxsize=32)

        async def send_token(token: str) -> None:
            await tokens.put(token)

        async def send_status(status: StatusUpdate) -> None:
            await tokens.put(status)

        async def generate() -> None:
            try:
                generation_options = {}
                if request.generation_model != "local":
                    generation_options = {
                        "hosted_model": request.generation_model,
                        "api_key": request.api_key.strip(),
                    }
                await rag_system.generate_async(
                    request.question,
                    result,
                    send_token,
                    on_status=send_status,
                    **generation_options,
                )
            except Exception as error:
                await tokens.put(error)
            finally:
                await tokens.put(None)

        generation = asyncio.create_task(generate())

        try:
            while True:
                token = await tokens.get()
                if token is None:
                    break
                if isinstance(token, StatusUpdate):
                    yield stream_event(
                        "status",
                        phase=token.phase,
                        elapsed=round(token.elapsed, 1),
                        estimated_total=token.estimated_total,
                    )
                    continue
                if isinstance(token, Exception):
                    yield stream_event(
                        "error",
                        code=generation_error_code(token),
                        message="Generation failed. Review the cited sources.",
                    )
                    await generation
                    return
                yield stream_event("token", text=token)

            await generation
            yield stream_event("done")
        finally:
            if not generation.done():
                generation.cancel()
                with suppress(asyncio.CancelledError):
                    await generation

    return StreamingResponse(stream(), media_type="application/x-ndjson")
