import json
import logging

import httpx

from backend.config import (
    CONTEXT_WINDOW,
    GENERATION_MODEL,
    LLM_PROVIDER,
    MAX_OUTPUT_TOKENS,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_THINK,
    OLLAMA_URL,
    TEMPERATURE,
    VLLM_URL,
)


logger = logging.getLogger(__name__)


def messages(
    query: str,
    context: str,
    symbolic_context: str | None = None,
) -> list[dict]:
    system_content = (
        "제공된 법률 문서 문맥만 근거로 답하세요. "
        "질문이 한국어이면 반드시 전체 답변을 한국어로 작성하세요. "
        "한국어 질문에 영어로 답하지 마세요. "
        "문맥에 답이 없으면 검색된 문서만으로는 답할 수 없다고 말하세요. "
        "사용한 조각은 정확히 [N] 형식으로 인용하세요. "
        "한국어 답변의 모든 문장에도 근거 조각 바로 뒤에 인용을 붙이세요. "
        "인용이 여러 개이면 [1][2][3]처럼 쓰세요. "
        "[N]을 인용하기 전에 해당 조각이 주장을 실제로 뒷받침하는지 확인하세요. "
        "예/아니요 질문은 '예.' 또는 '아니요.'로 시작한 뒤 법적 조건을 한두 문장으로 설명하세요."
    )
    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": (
                f"{symbolic_context}\n\n" if symbolic_context else ""
            )
            + f"문맥:\n{context}\n\n질문: {query}",
        },
    ]


def payload(
    query: str,
    context: str,
    include_think: bool,
    symbolic_context: str | None = None,
) -> dict:
    request = {
        "model": GENERATION_MODEL,
        "messages": messages(query, context, symbolic_context),
        "stream": True,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": CONTEXT_WINDOW,
            "num_predict": MAX_OUTPUT_TOKENS,
        },
    }
    if include_think:
        request["think"] = OLLAMA_THINK
    return request


def vllm_payload(
    query: str,
    context: str,
    symbolic_context: str | None = None,
) -> dict:
    return {
        "model": GENERATION_MODEL,
        "messages": messages(query, context, symbolic_context),
        "stream": True,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }


async def stream_answer(
    query: str,
    context: str,
    callback,
    symbolic_context: str | None = None,
) -> None:
    if LLM_PROVIDER == "vllm":
        try:
            await stream_vllm_payload(
                vllm_payload(query, context, symbolic_context=symbolic_context),
                callback,
            )
        except httpx.HTTPError as error:
            logger.exception("vLLM request failed.")
            await callback(f"\n[Generation error: {error}]\n")
        return

    if LLM_PROVIDER != "ollama":
        await callback(f"\n[Generation error: unknown LLM provider {LLM_PROVIDER!r}]\n")
        return

    try:
        await stream_payload(
            payload(query, context, include_think=True, symbolic_context=symbolic_context),
            callback,
        )
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 400:
            logger.warning("Ollama rejected the think option; retrying without it.")
            try:
                await stream_payload(
                    payload(
                        query,
                        context,
                        include_think=False,
                        symbolic_context=symbolic_context,
                    ),
                    callback,
                )
            except httpx.HTTPError as retry_error:
                logger.exception("Ollama retry failed.")
                await callback(f"\n[Generation error: {retry_error}]\n")
            return
        logger.exception("Ollama request failed.")
        await callback(f"\n[Generation error: {error}]\n")
    except httpx.HTTPError as error:
        logger.exception("Ollama request failed.")
        await callback(f"\n[Generation error: {error}]\n")


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

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping non-JSON Ollama stream line: %r", line)
                    continue
                message = event.get("message", {})
                token = message.get("content")
                if token:
                    await callback(token)


async def stream_vllm_payload(request: dict, callback) -> None:
    async with httpx.AsyncClient(timeout=600) as client:
        async with client.stream(
            "POST",
            f"{VLLM_URL}/v1/chat/completions",
            json=request,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break

                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("Skipping non-JSON vLLM stream line: %r", line)
                    continue

                choices = event.get("choices", [])
                if not choices:
                    continue
                token = choices[0].get("delta", {}).get("content")
                if token:
                    await callback(token)
