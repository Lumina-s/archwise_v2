from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.llm_gateway import build_llm_gateway_router, build_openai_compatible_router
from app.services.llm_client import LLMClient

load_dotenv()

llm_client = LLMClient()

app = FastAPI(
    title="ArchWise LLM 网关服务",
    description="聊天模型与 embedding 的统一出口。",
    version="1.0.0",
)
app.include_router(build_llm_gateway_router(llm_client))
app.include_router(build_openai_compatible_router(llm_client))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "llm-gateway"}
