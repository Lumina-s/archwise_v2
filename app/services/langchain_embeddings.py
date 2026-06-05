from __future__ import annotations

from langchain_core.embeddings import Embeddings


class LLMClientEmbeddings(Embeddings):
    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("LLMClientEmbeddings must be used through async embedding methods.")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("LLMClientEmbeddings must be used through async embedding methods.")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = await self.llm_client.embed_texts(texts)
        if vectors is None:
            raise RuntimeError(getattr(self.llm_client, "last_error", None) or "Embedding service returned no document vectors.")
        return vectors

    async def aembed_query(self, text: str) -> list[float]:
        vectors = await self.aembed_documents([text])
        if len(vectors) != 1:
            raise RuntimeError("Embedding service returned an invalid query vector count.")
        return vectors[0]
