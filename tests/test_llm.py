import pytest
import httpx

from backend import llm


def test_browser_selected_gemini_uses_request_key_and_model(monkeypatch):
    captured = {}

    async def fake_stream(
        url, request, callback, parse_line, token_from_event, headers=None,
    ):
        captured.update(url=url, request=request, headers=headers)
        await callback("answer")

    monkeypatch.setattr(llm, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm, "stream_json_tokens", fake_stream)

    async def callback(_token):
        pass

    import asyncio
    asyncio.run(
        llm.stream_answer(
            "질문",
            "문맥",
            callback,
            hosted_model="gemini-3.1-flash-lite",
            api_key="browser-secret",
        )
    )

    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert captured["headers"] == {"Authorization": "Bearer browser-secret"}
    assert captured["request"]["model"] == "gemini-3.1-flash-lite"
    assert "seed" not in captured["request"]


def test_browser_selected_groq_model_requires_key(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "ollama")

    async def callback(_token):
        pass

    import asyncio
    with pytest.raises(ValueError, match="API key"):
        asyncio.run(
            llm.stream_answer(
                "질문",
                "문맥",
                callback,
                hosted_model="qwen/qwen3.6-27b",
            )
        )


def test_browser_selected_groq_model_disables_reasoning(monkeypatch):
    captured = {}

    async def fake_stream(
        url, request, callback, parse_line, token_from_event, headers=None,
    ):
        captured.update(url=url, request=request, headers=headers)
        await callback("answer")

    monkeypatch.setattr(llm, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm, "stream_json_tokens", fake_stream)

    async def callback(_token):
        pass

    import asyncio
    asyncio.run(
        llm.stream_answer(
            "질문",
            "문맥",
            callback,
            hosted_model="qwen/qwen3.6-27b",
            api_key="browser-secret",
        )
    )

    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["request"]["reasoning_effort"] == "none"


def test_openai_compatible_stream_uses_base_url_and_auth(monkeypatch):
    captured = {}

    async def fake_stream(url, request, callback, parse_line, token_from_event, headers=None):
        captured.update(url=url, request=request, headers=headers)
        await callback("answer")

    monkeypatch.setattr(llm, "LLM_PROVIDER", "openai_compatible")
    monkeypatch.setattr(llm, "LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr(llm, "LLM_API_KEY", "secret")
    monkeypatch.setattr(llm, "stream_json_tokens", fake_stream)

    async def callback(_token):
        pass

    import asyncio
    asyncio.run(llm.stream_answer("질문", "문맥", callback))

    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["request"]["model"] == llm.GENERATION_MODEL


def test_openai_compatible_retries_empty_final_output(monkeypatch):
    requests = []

    async def fake_stream(url, request, callback, parse_line, token_from_event, headers=None):
        requests.append(request)
        if len(requests) == 2:
            await callback("answer")

    monkeypatch.setattr(llm, "LLM_PROVIDER", "openai_compatible")
    monkeypatch.setattr(llm, "LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr(llm, "stream_json_tokens", fake_stream)

    tokens = []

    async def callback(token):
        tokens.append(token)

    import asyncio
    asyncio.run(llm.stream_answer("질문", "문맥", callback))

    assert len(requests) == 2
    assert requests[1]["max_tokens"] == llm.MAX_OUTPUT_TOKENS * 2
    assert tokens == ["answer"]


def test_openai_compatible_reports_two_empty_attempts(monkeypatch):
    async def fake_stream(url, request, callback, parse_line, token_from_event, headers=None):
        pass

    monkeypatch.setattr(llm, "LLM_PROVIDER", "openai_compatible")
    monkeypatch.setattr(llm, "LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr(llm, "stream_json_tokens", fake_stream)
    tokens = []

    async def callback(token):
        tokens.append(token)

    import asyncio
    asyncio.run(llm.stream_answer("질문", "문맥", callback))

    assert len(tokens) == 1
    assert "최종 답변을 생성하지 못했습니다" in tokens[0]


def test_openai_compatible_retries_transport_failure_before_text(monkeypatch):
    attempts = 0

    async def fake_stream(url, request, callback, parse_line, token_from_event, headers=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadError("input stream closed")
        await callback("answer")

    monkeypatch.setattr(llm, "LLM_PROVIDER", "openai_compatible")
    monkeypatch.setattr(llm, "LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr(llm, "stream_json_tokens", fake_stream)
    tokens = []

    async def callback(token):
        tokens.append(token)

    import asyncio
    asyncio.run(llm.stream_answer("질문", "문맥", callback))

    assert attempts == 2
    assert tokens == ["answer"]


def test_openai_compatible_requires_base_url(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "openai_compatible")
    monkeypatch.setattr(llm, "LLM_BASE_URL", "")

    async def callback(_token):
        pass

    import asyncio
    with pytest.raises(ValueError, match="LEGALRAG_LLM_BASE_URL"):
        asyncio.run(llm.stream_answer("질문", "문맥", callback))


def test_model_payload_routes_judge_to_hosted_provider(monkeypatch):
    captured = {}

    async def fake_stream(request, callback, base_url, headers=None):
        captured.update(request=request, base_url=base_url, headers=headers)
        await callback("1.0")

    monkeypatch.setattr(llm, "LLM_PROVIDER", "openai_compatible")
    monkeypatch.setattr(llm, "LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr(llm, "LLM_API_KEY", "secret")
    monkeypatch.setattr(llm, "stream_openai_compatible_payload", fake_stream)
    tokens = []

    async def callback(token):
        tokens.append(token)

    request = {
        "model": "judge-model",
        "messages": [],
        "stream": True,
        "keep_alive": "1h",
        "options": {"temperature": 0, "num_predict": 512},
    }
    import asyncio
    asyncio.run(llm.stream_model_payload(request, callback))

    assert captured["request"]["model"] == "judge-model"
    assert captured["request"]["max_tokens"] == 512
    assert "options" not in captured["request"]
    assert "keep_alive" not in captured["request"]
    assert captured["base_url"] == "https://example.test/v1"
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert tokens == ["1.0"]
