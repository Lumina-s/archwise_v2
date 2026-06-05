from __future__ import annotations

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.system import build_system_router
from app.services.service_proxy import proxy_request

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
REASONING_SERVICE_URL = os.getenv("REASONING_SERVICE_URL", "http://127.0.0.1:8011").rstrip("/")
KNOWLEDGE_SERVICE_URL = os.getenv("KNOWLEDGE_SERVICE_URL", "http://127.0.0.1:8012").rstrip("/")
LLM_GATEWAY_SERVICE_URL = os.getenv("LLM_GATEWAY_SERVICE_URL", "http://127.0.0.1:8013").rstrip("/")

app = FastAPI(
    title="ArchWise API Gateway",
    description="Frontend entrypoint and HTTP gateway for ArchWise services.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.include_router(build_system_router(templates))


def _select_upstream(path: str) -> str:
    if path in {"recommend", "recommend/stream", "topology/stream"}:
        return REASONING_SERVICE_URL
    if path.startswith("llm/"):
        return LLM_GATEWAY_SERVICE_URL
    if path == "styles" or path == "cases" or path.startswith("knowledge/"):
        return KNOWLEDGE_SERVICE_URL
    raise HTTPException(status_code=404, detail=f"No gateway route for /api/{path}")


def _is_stream(path: str) -> bool:
    return path.endswith("/stream")


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def api_gateway(path: str, request: Request):
    return await proxy_request(request, _select_upstream(path), f"/api/{path}", stream=_is_stream(path))


@app.get("/health/services")
async def service_health() -> dict[str, object]:
    services = {
        "reasoning": f"{REASONING_SERVICE_URL}/health",
        "knowledge": f"{KNOWLEDGE_SERVICE_URL}/health",
        "llm_gateway": f"{LLM_GATEWAY_SERVICE_URL}/health",
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        results = {}
        for name, url in services.items():
            response = await client.get(url)
            response.raise_for_status()
            results[name] = response.json()
    return {"status": "ok", "services": results}
