from __future__ import annotations

from typing import Any

import httpx

from app.models.schemas import ArchitectureStyle, CandidateEvaluation, CaseRecord, ExtractedFeatures


class RemoteKnowledgeGraphClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def retrieve_topology_knowledge(
        self,
        requirement: str,
        features: ExtractedFeatures,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/internal/topology/retrieve",
                json={"requirement": requirement, "features": features.model_dump()},
            )
            response.raise_for_status()
            return response.json()

    async def normalize_topology_patch(
        self,
        patch: dict[str, Any],
        requirement: str,
        features: ExtractedFeatures,
        graph_knowledge: dict[str, Any] | None = None,
        coverage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/internal/topology/normalize",
                json={
                    "patch": patch,
                    "requirement": requirement,
                    "features": features.model_dump(),
                    "graph_knowledge": graph_knowledge or {},
                    "coverage": coverage or {},
                },
            )
            response.raise_for_status()
            return response.json()

    def merge_topology_patch(
        self,
        requirement: str,
        features: ExtractedFeatures,
        patch: dict[str, Any],
    ) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/internal/topology/merge",
                json={"requirement": requirement, "features": features.model_dump(), "patch": patch},
            )
            response.raise_for_status()
            return response.json()


class KnowledgeServiceClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.graph_service = RemoteKnowledgeGraphClient(self.base_url, timeout)
        self.repository = None

    def list_styles(self) -> list[ArchitectureStyle]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/api/styles")
            response.raise_for_status()
            return [ArchitectureStyle.model_validate(item) for item in response.json()]

    async def retrieve_trusted_cases(
        self,
        requirement: str,
        features: ExtractedFeatures,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/internal/cases/retrieve",
                json={"requirement": requirement, "features": features.model_dump(), "top_k": top_k},
            )
            response.raise_for_status()
            return response.json()

    async def capture_candidate_case(
        self,
        requirement: str,
        features: ExtractedFeatures,
        candidates: list[CandidateEvaluation],
    ) -> CaseRecord:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/internal/cases/capture",
                json={
                    "requirement": requirement,
                    "features": features.model_dump(),
                    "candidates": [candidate.model_dump() for candidate in candidates],
                },
            )
            response.raise_for_status()
            return CaseRecord.model_validate(response.json())
