from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

from app.api.knowledge import build_knowledge_router
from app.api.llm_gateway import build_llm_gateway_router
from app.api.recommendation import router as recommendation_router
from app.api.runtime import knowledge_service, recommendation_service
from app.api.system import build_system_router

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="ArchWise 软件体系结构风格智能助手",
    description="LLM + 知识图谱 + 多 Agent + 规则引擎的架构推荐演示系统",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.include_router(build_system_router(templates))
app.include_router(build_llm_gateway_router(recommendation_service.llm_client))
app.include_router(recommendation_router)
app.include_router(build_knowledge_router(knowledge_service))
