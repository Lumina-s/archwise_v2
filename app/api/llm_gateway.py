from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import EmbeddingRequest, EmbeddingResponse
from app.services.llm_client import LLMClient


def build_llm_gateway_router(llm_client: LLMClient) -> APIRouter:
    router = APIRouter(prefix="/api/llm", tags=["llm-gateway"])

    @router.get("/status")
    async def llm_status():
        return await llm_client.ping()

    @router.post("/embeddings", response_model=EmbeddingResponse)
    async def embeddings(payload: EmbeddingRequest) -> EmbeddingResponse:
        vectors = await llm_client.embed_texts(payload.texts)
        if vectors is None:
            raise HTTPException(status_code=503, detail=llm_client.last_error or "Embedding 服务调用失败。")
        return EmbeddingResponse(vectors=vectors)

    return router
