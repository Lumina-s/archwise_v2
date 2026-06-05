from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse


HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "content-encoding",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def filtered_headers(headers) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() not in HOP_BY_HOP_HEADERS}


def target_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


async def proxy_request(request: Request, base_url: str, path: str, stream: bool = False):
    url = target_url(base_url, path)
    body = await request.body()
    headers = filtered_headers(request.headers)
    headers.pop("host", None)

    if stream:
        return await _proxy_stream(request, url, body, headers)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.request(
                request.method,
                url,
                params=request.query_params,
                content=body,
                headers=headers,
            )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Upstream service unavailable: {exc}") from exc

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=filtered_headers(response.headers),
        media_type=response.headers.get("content-type"),
    )


async def _proxy_stream(request: Request, url: str, body: bytes, headers: dict[str, str]) -> StreamingResponse:
    client = httpx.AsyncClient(timeout=None)
    try:
        upstream_request = client.build_request(
            request.method,
            url,
            params=request.query_params,
            content=body,
            headers=headers,
        )
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.RequestError as exc:
        await client.aclose()
        raise HTTPException(status_code=503, detail=f"Upstream service unavailable: {exc}") from exc

    async def chunks() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream_response.aiter_bytes():
                yield chunk
        finally:
            await upstream_response.aclose()
            await client.aclose()

    return StreamingResponse(
        chunks(),
        status_code=upstream_response.status_code,
        headers=filtered_headers(upstream_response.headers),
        media_type=upstream_response.headers.get("content-type"),
    )
