import asyncio

import pytest
from langchain_chroma import Chroma

from app.models.schemas import ArchitectureStyle, CandidateEvaluation, ExtractedFeatures, RecommendationResponse
from app.services.case_memory import CaseMemoryService
from app.services.recommendation_cache import RecommendationCacheService


class FakeLLMClient:
    def __init__(self):
        self.last_error = None

    async def embed_texts(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


def style() -> ArchitectureStyle:
    return ArchitectureStyle(
        id="event_driven",
        name="事件驱动架构",
        category="分布式",
        description="事件异步协作",
        suitable_for=["即时通信"],
        quality_scores={"scalability": 0.9, "performance": 0.8},
        strengths=["削峰"],
        weaknesses=["一致性复杂"],
        topology="事件总线连接服务",
        rules={},
    )


def response(requirement: str) -> RecommendationResponse:
    features = ExtractedFeatures(
        domain="即时通信",
        keywords=["聊天", "实时"],
        business_capabilities=["消息通信"],
        architecture_drivers=["实时性"],
        topology_expectations={},
        quality_attributes={"realtime": 0.9},
        constraints={},
        data_flow="event_stream",
        ambiguity_notes=[],
    )
    candidate = CandidateEvaluation(
        style_id="event_driven",
        name="事件驱动架构",
        score=93,
        raw_score=93,
        recommendation_role="核心推荐",
        confidence="高",
        matched_reasons=["实时事件流"],
        risks=[],
        deductions=[],
        quality_scores={"scalability": 0.9, "performance": 0.8},
    )
    return RecommendationResponse(
        requirement=requirement,
        features=features,
        candidates=[candidate],
        final_recommendation=candidate,
        report="推荐事件驱动架构。",
        comparison_matrix=[],
        topology_diagrams={},
        topology_graphs={},
        trace=["生成完成"],
        decision_trace={"rule_evidence": {"enabled": True}},
        composition_recommendation={},
    )


def test_case_memory_uses_langchain_chroma(tmp_path):
    service = CaseMemoryService(
        FakeLLMClient(),
        data_dir=tmp_path,
        chroma_dir=tmp_path / "case_chroma",
    )

    assert isinstance(service.vector_store, Chroma)


def test_recommendation_cache_exact_and_semantic_hit(tmp_path):
    cache = RecommendationCacheService(
        FakeLLMClient(),
        data_dir=tmp_path,
        chroma_dir=tmp_path / "recommendation_chroma",
    )
    requirement = "开发跨平台即时通讯系统，支持万人在线和实时消息"
    styles = [style()]

    asyncio.run(
        cache.store(
            response=response(requirement),
            requirement=requirement,
            top_k=3,
            topology_options={"fast_mode": True},
            styles=styles,
        )
    )

    exact = asyncio.run(
        cache.lookup(
            requirement=requirement,
            top_k=3,
            topology_options={"fast_mode": True},
            styles=styles,
        )
    )
    assert exact is not None
    assert exact.decision_trace["recommendation_cache_evidence"]["hit_type"] == "exact"

    semantic = asyncio.run(
        cache.lookup(
            requirement="开发跨平台即时通讯系统，支持万人同时在线和实时消息",
            top_k=3,
            topology_options={"fast_mode": True},
            styles=styles,
        )
    )
    assert semantic is not None
    assert semantic.decision_trace["recommendation_cache_evidence"]["hit_type"] == "semantic"


def test_invalid_recommendation_cache_file_fails_fast(tmp_path):
    (tmp_path / "recommendation_cache.json").write_text("[]", encoding="utf-8")
    cache = RecommendationCacheService(
        FakeLLMClient(),
        data_dir=tmp_path,
        chroma_dir=tmp_path / "recommendation_chroma",
    )

    with pytest.raises(ValueError, match="Recommendation cache must be a JSON object"):
        asyncio.run(
            cache.lookup(
                requirement="开发跨平台即时通讯系统",
                top_k=3,
                topology_options={},
                styles=[style()],
            )
        )
