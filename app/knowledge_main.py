from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.knowledge import build_knowledge_router
from app.services.knowledge_service import KnowledgeService
from app.services.llm_client import LLMClient

load_dotenv()

llm_client = LLMClient()
knowledge_service = KnowledgeService(llm_client)

app = FastAPI(
    title="ArchWise 知识服务",
    description="架构风格、图谱知识和案例记忆的持久化服务边界。",
    version="1.0.0",
)
app.include_router(build_knowledge_router(knowledge_service))
