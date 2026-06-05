import asyncio

import pytest

from app.services.langchain_embeddings import LLMClientEmbeddings


class FakeLLMClient:
    def __init__(self, vectors):
        self.vectors = vectors
        self.last_error = None
        self.requests = []

    async def embed_texts(self, texts):
        self.requests.append(texts)
        return self.vectors


def test_langchain_embeddings_async_documents():
    llm_client = FakeLLMClient([[1.0, 0.0], [0.0, 1.0]])
    embeddings = LLMClientEmbeddings(llm_client)

    vectors = asyncio.run(embeddings.aembed_documents(["alpha", "beta"]))

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert llm_client.requests == [["alpha", "beta"]]


def test_langchain_embeddings_async_query():
    llm_client = FakeLLMClient([[0.5, 0.5]])
    embeddings = LLMClientEmbeddings(llm_client)

    vector = asyncio.run(embeddings.aembed_query("alpha"))

    assert vector == [0.5, 0.5]
    assert llm_client.requests == [["alpha"]]


def test_langchain_embeddings_sync_methods_fail_fast():
    embeddings = LLMClientEmbeddings(FakeLLMClient([[1.0]]))

    with pytest.raises(RuntimeError, match="async embedding methods"):
        embeddings.embed_documents(["alpha"])
    with pytest.raises(RuntimeError, match="async embedding methods"):
        embeddings.embed_query("alpha")


def test_langchain_embeddings_reports_empty_embedding_response():
    llm_client = FakeLLMClient(None)
    llm_client.last_error = "embedding unavailable"
    embeddings = LLMClientEmbeddings(llm_client)

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        asyncio.run(embeddings.aembed_documents(["alpha"]))
