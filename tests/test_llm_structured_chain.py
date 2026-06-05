import asyncio
import json

from app.models.schemas import ExtractedFeatures
from app.services.llm_client import LLMClient


def test_extract_features_uses_langchain_structured_chain(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    async def fake_chat(self, payload, request_timeout=None):
        return json.dumps(
            {
                "domain": "在线教育",
                "keywords": ["直播", "课程", "作业"],
                "business_capabilities": ["课程管理", "直播授课", "作业批改", "学习统计"],
                "architecture_drivers": ["高可用", "可扩展"],
                "topology_expectations": {
                    "must_have_components": ["课程服务", "直播服务"],
                    "must_have_relations": ["课程服务->直播服务"],
                    "quality_infrastructure": ["监控服务"],
                },
                "quality_attributes": {
                    "concurrency": 0.7,
                    "realtime": 0.8,
                    "reliability": 0.8,
                    "scalability": 0.75,
                    "data_intensity": 0.5,
                    "ai_reasoning": 0.0,
                },
                "constraints": {
                    "scale_mentions": ["万人观看"],
                    "deployment": ["云部署"],
                    "requires_high_availability": True,
                    "requires_future_extension": True,
                },
                "data_flow": "request_response",
                "ambiguity_notes": [],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr("app.services.llm_client.LLMClient._chat", fake_chat)
    monkeypatch.setattr(
        "app.services.llm_client.LLMClient._extract_json",
        staticmethod(lambda content: (_ for _ in ()).throw(AssertionError("_extract_json should not be used"))),
    )

    client = LLMClient(cache_path=tmp_path / "llm_cache.json")
    features = asyncio.run(client.extract_features("在线教育平台，支持万人直播、课程管理、作业批改和学习统计"))

    assert isinstance(features, ExtractedFeatures)
    assert features.domain == "在线教育"
    assert features.data_flow == "request_response"
    assert "作业批改" in features.business_capabilities


def test_recommend_architectures_uses_langchain_structured_chain(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    async def fake_chat(self, payload, request_timeout=None):
        return json.dumps(
            {
                "candidates": [
                    {
                        "style_id": "layered",
                        "name": "分层架构",
                        "score": 91,
                        "raw_score": 91,
                        "recommendation_role": "核心推荐",
                        "confidence": "高",
                        "matched_reasons": ["业务清晰"],
                        "risks": ["高并发能力有限"],
                        "deductions": [],
                        "quality_scores": quality_scores,
                    },
                    {
                        "style_id": "mvc",
                        "name": "MVC 架构",
                        "score": 84,
                        "raw_score": 84,
                        "recommendation_role": "备选方案",
                        "confidence": "中高",
                        "matched_reasons": ["界面交互明确"],
                        "risks": [],
                        "deductions": [],
                        "quality_scores": quality_scores,
                    },
                    {
                        "style_id": "microservices",
                        "name": "微服务架构",
                        "score": 73,
                        "raw_score": 73,
                        "recommendation_role": "专项补充",
                        "confidence": "中",
                        "matched_reasons": ["后续扩展可用"],
                        "risks": ["运维成本较高"],
                        "deductions": [],
                        "quality_scores": quality_scores,
                    },
                ],
                "composition_recommendation": {
                    "composition_needed": False,
                    "primary_style": "分层架构",
                    "supporting_styles": [],
                    "reason": "当前需求可由主架构覆盖",
                    "triggers": [],
                    "overengineering_warnings": [],
                },
                "review_notes": ["排序合理"],
            },
            ensure_ascii=False,
        )

    quality_scores = {
        "scalability": 0.6,
        "performance": 0.7,
        "reliability": 0.75,
        "modifiability": 0.7,
        "complexity": 0.4,
        "realtime": 0.3,
    }
    styles = [
        {
            "id": "layered",
            "name": "分层架构",
            "category": "经典架构",
            "description": "按层组织职责",
            "suitable_for": ["管理系统"],
            "quality_scores": quality_scores,
        },
        {
            "id": "mvc",
            "name": "MVC 架构",
            "category": "经典架构",
            "description": "拆分模型视图控制器",
            "suitable_for": ["Web 应用"],
            "quality_scores": quality_scores,
        },
        {
            "id": "microservices",
            "name": "微服务架构",
            "category": "分布式架构",
            "description": "按业务能力划分服务",
            "suitable_for": ["复杂平台"],
            "quality_scores": quality_scores,
        },
    ]
    features = ExtractedFeatures(
        domain="教务管理",
        keywords=["课程", "作业", "统计"],
        business_capabilities=["课程管理", "作业批改", "学习统计"],
        architecture_drivers=["可维护"],
        topology_expectations={
            "must_have_components": ["课程服务", "作业服务"],
            "must_have_relations": ["课程服务->作业服务"],
            "quality_infrastructure": ["监控服务"],
        },
        quality_attributes={
            "concurrency": 0.4,
            "realtime": 0.2,
            "reliability": 0.7,
            "scalability": 0.6,
            "data_intensity": 0.5,
            "ai_reasoning": 0.0,
        },
        constraints={
            "scale_mentions": [],
            "deployment": [],
            "requires_high_availability": False,
            "requires_future_extension": True,
        },
        data_flow="request_response",
        ambiguity_notes=[],
    )

    monkeypatch.setattr("app.services.llm_client.LLMClient._chat", fake_chat)
    monkeypatch.setattr(
        "app.services.llm_client.LLMClient._extract_json",
        staticmethod(lambda content: (_ for _ in ()).throw(AssertionError("_extract_json should not be used"))),
    )

    client = LLMClient(cache_path=tmp_path / "llm_cache.json")
    result = asyncio.run(client.recommend_architectures("教务管理系统", features, styles, top_k=3))

    assert result is not None
    candidates, composition = result
    assert [item.style_id for item in candidates] == ["layered", "mvc", "microservices"]
    assert composition["primary_style"] == "分层架构"
    assert composition["review_notes"] == ["排序合理"]
