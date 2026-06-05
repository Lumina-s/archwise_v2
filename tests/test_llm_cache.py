import asyncio

import httpx
import pytest
from langchain_core.globals import get_llm_cache
from langchain_core.outputs import Generation

from app.services.llm_client import LLMClient


class CountingAsyncClient:
    calls = 0

    def __init__(self, timeout):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, url, headers, json):
        CountingAsyncClient.calls += 1
        request = httpx.Request("POST", url)
        if url.endswith("/embeddings"):
            return httpx.Response(
                200,
                request=request,
                json={
                    "data": [
                        {"index": index, "embedding": [float(index), 1.0]}
                        for index, _ in enumerate(json["input"])
                    ]
                },
            )
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": "cached answer"}}]},
        )


def test_chat_cache_reuses_successful_response(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr("app.services.llm_client.httpx.AsyncClient", CountingAsyncClient)
    CountingAsyncClient.calls = 0

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    cache_path = tmp_path / "llm_cache.json"

    first_client = LLMClient(cache_path=cache_path)
    second_client = LLMClient(cache_path=cache_path)

    assert asyncio.run(first_client._chat(payload)) == "cached answer"
    assert asyncio.run(second_client._chat(payload)) == "cached answer"
    assert CountingAsyncClient.calls == 1


def test_embedding_cache_reuses_successful_response(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding.example/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "embedding-model")
    monkeypatch.setattr("app.services.llm_client.httpx.AsyncClient", CountingAsyncClient)
    CountingAsyncClient.calls = 0
    cache_path = tmp_path / "llm_cache.json"

    first_client = LLMClient(cache_path=cache_path)
    second_client = LLMClient(cache_path=cache_path)

    assert asyncio.run(first_client.embed_texts(["alpha", "beta"])) == [[0.0, 1.0], [1.0, 1.0]]
    assert asyncio.run(second_client.embed_texts(["alpha", "beta"])) == [[0.0, 1.0], [1.0, 1.0]]
    assert CountingAsyncClient.calls == 1


def test_invalid_cache_file_fails_fast(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    cache_path = tmp_path / "llm_cache.json"
    cache_path.write_text("[]", encoding="utf-8")

    client = LLMClient(cache_path=cache_path)

    with pytest.raises(ValueError, match="LLM cache must be a JSON object"):
        asyncio.run(client._chat({"model": "test-model", "messages": [], "stream": False}))


def test_llm_cache_registers_langchain_base_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    cache_path = tmp_path / "llm_cache.json"

    client = LLMClient(cache_path=cache_path)
    langchain_cache = get_llm_cache()

    assert langchain_cache is client.cache
    langchain_cache.update("prompt", "llm-string", [Generation(text="answer")])
    cached = langchain_cache.lookup("prompt", "llm-string")
    assert cached is not None
    assert cached[0].text == "answer"


def test_chat_reports_exception_class_when_exception_has_empty_message(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    class EmptyMessageAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, headers, json):
            raise TimeoutError()

    monkeypatch.setattr("app.services.llm_client.httpx.AsyncClient", EmptyMessageAsyncClient)
    client = LLMClient(cache_path=tmp_path / "llm_cache.json")

    assert asyncio.run(client._chat({"model": "test-model", "messages": [], "stream": False})) is None
    assert client.last_error == "TimeoutError"
