import json
import logging

import httpx

from backend.config import (
    CONTEXT_WINDOW,
    GENERATION_MODEL,
    MAX_OUTPUT_TOKENS,
    OLLAMA_THINK,
    OLLAMA_URL,
    TEMPERATURE,
)


logger = logging.getLogger(__name__)


def messages(query: str, context: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "Answer only from the supplied legal-document context. "
                "If the context does not contain the answer, say that you "
                "cannot answer from the retrieved text. Cite each chunk you "
                "use with the exact format [N]. Put the citation directly "
                "after each supported claim. Format multiple citations as "
                "[1][2][3]."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]


def payload(query: str, context: str, include_think: bool) -> dict:
    request = {
        "model": GENERATION_MODEL,
        "messages": messages(query, context),
        "stream": True,
        "keep_alive": "1h",
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": CONTEXT_WINDOW,
            "num_predict": MAX_OUTPUT_TOKENS,
        },
    }
    if include_think:
        request["think"] = OLLAMA_THINK
    return request


async def stream_answer(query: str, context: str, callback) -> None:
    try:
        await stream_payload(payload(query, context, include_think=True), callback)
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 400:
            logger.warning("Ollama rejected the think option; retrying without it.")
            await stream_payload(payload(query, context, include_think=False), callback)
            return
        raise


async def stream_payload(request: dict, callback) -> None:
    async with httpx.AsyncClient(timeout=600) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_URL}/api/chat",
            json=request,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue

                event = json.loads(line)
                message = event.get("message", {})
                token = message.get("content")
                if token:
                    await callback(token)
