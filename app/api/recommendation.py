from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.runtime import recommendation_service
from app.models.schemas import RequirementRequest, TopologyRequest
from app.services.exceptions import DeepSeekServiceError, RequirementParsingError

router = APIRouter(prefix="/api", tags=["recommendation"])


@router.post("/recommend")
async def recommend(payload: RequirementRequest):
    try:
        return await recommendation_service.recommend(
            payload.requirement,
            payload.top_k,
            topology_options={
                "fast_mode": payload.topology_fast_mode,
                "llm_timeout_seconds": payload.topology_llm_timeout_seconds,
                "repair_max_rounds": payload.topology_repair_max_rounds,
            },
        )
    except (RequirementParsingError, DeepSeekServiceError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/recommend/stream")
async def recommend_stream(payload: RequirementRequest):
    return StreamingResponse(
        recommendation_service.recommend_stream(
            payload.requirement,
            payload.top_k,
            topology_options={
                "fast_mode": payload.topology_fast_mode,
                "llm_timeout_seconds": payload.topology_llm_timeout_seconds,
                "repair_max_rounds": payload.topology_repair_max_rounds,
            },
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/topology/stream")
async def topology_stream(payload: TopologyRequest):
    return StreamingResponse(
        recommendation_service.topology_stream(
            requirement=payload.requirement,
            features=payload.features,
            final_recommendation=payload.final_recommendation,
            composition_recommendation=payload.composition_recommendation,
            decision_trace=payload.decision_trace,
            topology_options={
                "fast_mode": payload.topology_fast_mode,
                "llm_timeout_seconds": payload.topology_llm_timeout_seconds,
                "repair_max_rounds": payload.topology_repair_max_rounds,
            },
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
