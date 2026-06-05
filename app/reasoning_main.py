from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from app.api.recommendation import router as recommendation_router

app = FastAPI(
    title="ArchWise Reasoning Service",
    description="Recommendation orchestration service for requirement analysis, matching, report and topology generation.",
    version="1.0.0",
)
app.include_router(recommendation_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "reasoning"}
