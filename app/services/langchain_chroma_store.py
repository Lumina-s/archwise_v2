from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma

from app.services.langchain_embeddings import LLMClientEmbeddings


class LangChainChromaStore:
    def __init__(
        self,
        *,
        collection_name: str,
        persist_directory: Path,
        embeddings: LLMClientEmbeddings,
        collection_metadata: dict[str, Any] | None = None,
    ) -> None:
        persist_directory.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=None,
            persist_directory=str(persist_directory),
            collection_metadata=collection_metadata,
        )

    async def aupsert_texts(
        self,
        *,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            raise ValueError("Chroma upsert requires at least one id.")
        if len(ids) != len(texts) or len(ids) != len(metadatas):
            raise ValueError("Chroma upsert ids, texts, and metadatas must have equal lengths.")
        vectors = await self.embeddings.aembed_documents(texts)
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding vector count does not match Chroma documents.")
        await asyncio.to_thread(
            self.vector_store._collection.upsert,
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=vectors,
        )

    async def asimilarity_search_by_text(
        self,
        *,
        query: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ):
        query_vector = await self.embeddings.aembed_query(query)
        return await asyncio.to_thread(
            self.vector_store.similarity_search_by_vector_with_relevance_scores,
            query_vector,
            k,
            metadata_filter,
        )
