from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from app.models.schemas import EmbeddingRequest, EmbeddingResponse
from app.services.llm_client import LLMClient
from app.services.service_proxy import filtered_headers


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


def build_openai_compatible_router(llm_client: LLMClient) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["llm-gateway"])

    @router.post("/chat/completions")
    async def chat_completions(payload: dict[str, Any]):
        if not llm_client.api_key:
            raise HTTPException(status_code=503, detail="LLM service is not configured.")
        if payload.get("stream"):
            return await _stream_upstream(llm_client.chat_url, llm_client.api_key, payload, llm_client.timeout)
        return await _post_upstream(llm_client.chat_url, llm_client.api_key, payload, llm_client.timeout)

    @router.post("/embeddings")
    async def embeddings_proxy(payload: dict[str, Any]):
        if not (llm_client.embedding_api_key and llm_client.embedding_url):
            raise HTTPException(status_code=503, detail="Embedding service is not configured.")
        return await _post_upstream(
            llm_client.embedding_url,
            llm_client.embedding_api_key,
            payload,
            llm_client.timeout,
        )

    return router


async def _post_upstream(url: str, api_key: str, payload: dict[str, Any], timeout: float) -> Response:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=payload)
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=filtered_headers(response.headers),
        media_type=response.headers.get("content-type"),
    )


async def _stream_upstream(url: str, api_key: str, payload: dict[str, Any], timeout: float) -> StreamingResponse:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    client = httpx.AsyncClient(timeout=timeout)
    try:
        request = client.build_request("POST", url, headers=headers, json=payload)
        response = await client.send(request, stream=True)
    except httpx.RequestError as exc:
        await client.aclose()
        raise HTTPException(status_code=503, detail=f"LLM upstream unavailable: {exc}") from exc

    async def chunks() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(
        chunks(),
        status_code=response.status_code,
        headers=filtered_headers(response.headers),
        media_type=response.headers.get("content-type", "text/event-stream"),
    )
