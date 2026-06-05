from __future__ import annotations

from typing import Any

from app.models.schemas import ExtractedFeatures
from app.services.exceptions import RequirementParsingError
from app.services.llm_client import LLMClient


class RequirementAnalysisAgent:
    """Agent 1: transform natural language requirements into validated features."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    async def analyze(self, requirement: str) -> ExtractedFeatures:
        features = await self.llm_client.extract_features(requirement)
        if not features:
            message = self.llm_client.last_error or "LLM 未返回符合 Schema 的需求特征。"
            raise RequirementParsingError(f"需求解析失败：{message}")
        return self._normalize_features(features)

    def _normalize_features(self, features: ExtractedFeatures) -> ExtractedFeatures:
        quality_attributes = {
            key: round(max(0.0, min(1.0, float(value))), 2)
            for key, value in features.quality_attributes.items()
        }
        for key in ["concurrency", "realtime", "reliability", "scalability", "data_intensity", "ai_reasoning"]:
            quality_attributes.setdefault(key, 0.0)

        return features.model_copy(
            update={
                "keywords": self._dedupe(features.keywords),
                "business_capabilities": self._dedupe(features.business_capabilities),
                "architecture_drivers": self._dedupe(features.architecture_drivers),
                "topology_expectations": self._normalize_topology_expectations(features.topology_expectations),
                "quality_attributes": quality_attributes,
                "ambiguity_notes": self._dedupe(features.ambiguity_notes),
            }
        )

    @staticmethod
    def _normalize_topology_expectations(expectations: dict[str, Any] | None) -> dict[str, Any]:
        expectations = expectations or {}
        normalized: dict[str, Any] = {}
        for key in ["must_have_components", "must_have_relations", "quality_infrastructure"]:
            value = expectations.get(key, [])
            if isinstance(value, list):
                normalized[key] = [str(item).strip() for item in value if str(item).strip()]
            elif value:
                normalized[key] = [str(value).strip()]
            else:
                normalized[key] = []
        return normalized

    @staticmethod
    def _dedupe(items: list[Any]) -> list[Any]:
        return list(dict.fromkeys(str(item).strip() for item in items if str(item).strip()))
