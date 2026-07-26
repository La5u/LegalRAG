import asyncio
import json
import logging

import httpx

from backend.config import (
    CONTEXT_WINDOW,
    GENERATION_SEED,
    GENERATION_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_PROVIDER,
    MAX_OUTPUT_TOKENS,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_THINK,
    OLLAMA_URL,
    TEMPERATURE,
    VLLM_URL,
)


logger = logging.getLogger(__name__)
_http_client: httpx.AsyncClient | None = None
HOSTED_MODEL_URLS = {
    "gemini-3.1-flash-lite": (
        "https://generativelanguage.googleapis.com/v1beta/openai"
    ),
    "qwen/qwen3.6-27b": "https://api.groq.com/openai/v1",
}


def http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(600, connect=15),
        )
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
    _http_client = None


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
            "seed": GENERATION_SEED,
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
    max_tokens: int = MAX_OUTPUT_TOKENS,
    model: str = GENERATION_MODEL,
) -> dict:
    return {
        "model": model,
        "messages": messages(query, context, symbolic_context),
        "stream": True,
        "temperature": TEMPERATURE,
        "seed": GENERATION_SEED,
        "max_tokens": max_tokens,
    }


async def stream_answer(
    query: str,
    context: str,
    callback,
    symbolic_context: str | None = None,
    hosted_model: str | None = None,
    api_key: str | None = None,
) -> None:
    if hosted_model or LLM_PROVIDER in {"vllm", "openai_compatible"}:
        if hosted_model:
            base_url = HOSTED_MODEL_URLS.get(hosted_model)
            if base_url is None:
                raise ValueError(f"Unsupported hosted model: {hosted_model!r}")
            if not api_key:
                raise ValueError("An API key is required for hosted models")
            selected_model = hosted_model
            selected_api_key = api_key
        else:
            base_url = (
                f"{VLLM_URL.rstrip('/')}/v1"
                if LLM_PROVIDER == "vllm"
                else LLM_BASE_URL
            )
            selected_model = GENERATION_MODEL
            selected_api_key = LLM_API_KEY
        if not base_url:
            raise ValueError("LEGALRAG_LLM_BASE_URL is required")
        headers = (
            {"Authorization": f"Bearer {selected_api_key}"}
            if selected_api_key
            else None
        )
        final_text_seen = False

        async def track_final_text(token):
            nonlocal final_text_seen
            final_text_seen = True
            await callback(token)

        request = vllm_payload(
            query,
            context,
            symbolic_context=symbolic_context,
            model=selected_model,
        )
        if hosted_model and selected_model.startswith("gemini-"):
            request.pop("seed", None)
        if hosted_model == "qwen/qwen3.6-27b":
            request["reasoning_effort"] = "none"
        try:
            await stream_openai_compatible_payload(
                request, track_final_text, base_url, headers,
            )
        except (httpx.TransportError, httpx.HTTPStatusError) as error:
            if final_text_seen:
                raise
            if (
                isinstance(error, httpx.HTTPStatusError)
                and error.response.status_code not in {429, 502, 503, 504}
            ):
                raise
            logger.warning(
                "Hosted model stream failed before returning text; retrying.",
                exc_info=True,
            )
            retry_after = (
                error.response.headers.get("retry-after")
                if isinstance(error, httpx.HTTPStatusError)
                else None
            )
            await asyncio.sleep(min(float(retry_after or 0), 2.0))
        if not final_text_seen and (
            hosted_model or LLM_PROVIDER == "openai_compatible"
        ):
            logger.warning(
                "Hosted model emitted no final text; retrying with a larger output budget."
            )
            retry_request = dict(request)
            retry_request["max_tokens"] = MAX_OUTPUT_TOKENS * 2
            await stream_openai_compatible_payload(
                retry_request, track_final_text, base_url, headers,
            )
            if not final_text_seen:
                await callback(
                    "호스팅된 언어 모델이 최종 답변을 생성하지 못했습니다. "
                    "같은 검색 결과로 다시 시도해 주세요."
                )
        return

    if LLM_PROVIDER != "ollama":
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER!r}")

    try:
        await stream_payload(
            payload(query, context, include_think=True, symbolic_context=symbolic_context),
            callback,
        )
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 400:
            logger.warning("Ollama rejected the think option; retrying without it.")
            await stream_payload(
                payload(
                    query,
                    context,
                    include_think=False,
                    symbolic_context=symbolic_context,
                ),
                callback,
            )
            return
        raise


async def stream_payload(request: dict, callback) -> None:
    await stream_json_tokens(
        f"{OLLAMA_URL}/api/chat",
        request,
        callback,
        parse_ollama_line,
        token_from_ollama_event,
    )


async def stream_model_payload(request: dict, callback) -> None:
    """Stream an arbitrary chat request through the configured provider.

    Unlike :func:`stream_answer`, this accepts a complete request so callers
    such as the evaluator can select a separate judge model.
    """
    if LLM_PROVIDER == "ollama":
        await stream_payload(request, callback)
        return
    if LLM_PROVIDER not in {"vllm", "openai_compatible"}:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER!r}")

    hosted_request = dict(request)
    options = hosted_request.pop("options", {})
    hosted_request.pop("keep_alive", None)
    hosted_request.pop("think", None)
    hosted_request["max_tokens"] = options.get("num_predict", MAX_OUTPUT_TOKENS)
    hosted_request["temperature"] = options.get("temperature", TEMPERATURE)
    base_url = VLLM_URL if LLM_PROVIDER == "vllm" else LLM_BASE_URL
    if not base_url:
        raise ValueError("LEGALRAG_LLM_BASE_URL is required")
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else None
    await stream_openai_compatible_payload(
        hosted_request, callback, base_url, headers,
    )


async def stream_openai_compatible_payload(
    request: dict,
    callback,
    base_url: str,
    headers: dict | None = None,
) -> None:
    await stream_json_tokens(
        f"{base_url}/chat/completions",
        request,
        callback,
        parse_vllm_line,
        token_from_vllm_event,
        headers=headers,
    )


async def stream_json_tokens(
    url: str,
    request: dict,
    callback,
    parse_line,
    token_from_event,
    headers: dict | None = None,
) -> None:
    async with http_client().stream(
        "POST",
        url,
        json=request,
        headers=headers,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            event = parse_line(line)
            if event is None:
                continue
            if event == "[DONE]":
                break
            token = token_from_event(event)
            if token:
                await callback(token)


def parse_ollama_line(line: str):
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        logger.warning("Skipping non-JSON Ollama stream line: %r", line)
        return None


def parse_vllm_line(line: str):
    if not line.startswith("data: "):
        return None
    data = line.removeprefix("data: ").strip()
    if data == "[DONE]":
        return "[DONE]"
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        logger.warning("Skipping non-JSON vLLM stream line: %r", line)
        return None


def token_from_ollama_event(event: dict) -> str | None:
    return event.get("message", {}).get("content")


def token_from_vllm_event(event: dict) -> str | None:
    choices = event.get("choices", [])
    if not choices:
        return None
    return choices[0].get("delta", {}).get("content")
