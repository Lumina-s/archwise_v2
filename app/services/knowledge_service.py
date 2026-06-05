from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from app.knowledge.repository import KnowledgeRepository
from app.models.schemas import ArchitectureStyle, CandidateEvaluation, CaseRecord, CaseRequest, ExtractedFeatures
from app.services.case_memory import CaseMemoryService
from app.services.knowledge_graph import KnowledgeGraphService
from app.services.llm_client import LLMClient


class KnowledgeService:
    """Boundary for persistent architecture knowledge and case memory."""

    def __init__(
        self,
        llm_client: LLMClient,
        repository: KnowledgeRepository | None = None,
        graph_service: KnowledgeGraphService | None = None,
        case_memory: CaseMemoryService | None = None,
    ) -> None:
        self.repository = repository or KnowledgeRepository()
        self.graph_service = graph_service or KnowledgeGraphService(llm_client)
        self.case_memory = case_memory or CaseMemoryService(llm_client)
        self._neo4j_sync_task: asyncio.Task | None = None
        self._neo4j_sync_status: dict[str, Any] = {
            "running": False,
            "ok": None,
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }

    def list_styles(self) -> list[ArchitectureStyle]:
        if self.graph_service.neo4j.configured:
            return self.graph_service.list_styles()
        return self.repository.list_styles()

    def add_style(self, style: ArchitectureStyle) -> ArchitectureStyle:
        if self.graph_service.neo4j.configured:
            return self.graph_service.add_style(style)
        return self.repository.add_style(style)

    def build_style_graph(self):
        return self.graph_service.build_graph(self.list_styles())

    def build_topology_graph(self):
        return self.graph_service.build_topology_graph()

    def neo4j_status(self):
        return self.graph_service.neo4j_status()

    async def sync_styles_to_neo4j(self):
        return await self.graph_service.sync_to_neo4j(self.repository.list_styles())

    def start_neo4j_sync(self) -> dict[str, Any]:
        if self._neo4j_sync_task and not self._neo4j_sync_task.done():
            return self.neo4j_sync_status()
        self._neo4j_sync_status = {
            "running": True,
            "ok": None,
            "started_at": datetime.now(UTC).isoformat(),
            "finished_at": None,
            "result": None,
            "error": None,
        }
        self._neo4j_sync_task = asyncio.create_task(self._run_neo4j_sync_job())
        return self.neo4j_sync_status()

    def neo4j_sync_status(self) -> dict[str, Any]:
        return dict(self._neo4j_sync_status)

    async def _run_neo4j_sync_job(self) -> None:
        try:
            result = await self.sync_styles_to_neo4j()
            self._neo4j_sync_status = {
                **self._neo4j_sync_status,
                "running": False,
                "ok": bool(result.get("ok")),
                "finished_at": datetime.now(UTC).isoformat(),
                "result": result,
                "error": None if result.get("ok") else str(result),
            }
        except Exception as exc:
            self._neo4j_sync_status = {
                **self._neo4j_sync_status,
                "running": False,
                "ok": False,
                "finished_at": datetime.now(UTC).isoformat(),
                "result": None,
                "error": f"{exc.__class__.__name__}: {exc}",
            }

    async def rebuild_domain_topology(self):
        return await self.graph_service.rebuild_domain_topology()

    async def reindex_topology_vectors(self):
        return await self.graph_service.reindex_topology_vectors()

    async def detect_duplicate_like_nodes(self):
        return await self.graph_service.detect_duplicate_like_nodes()

    def list_cases(self) -> list[CaseRecord]:
        return self.case_memory.list_records()

    async def add_trusted_case(self, case: CaseRequest) -> CaseRecord:
        return await self.case_memory.add_trusted_case(case)

    async def add_manual_case(self, case: CaseRequest, status: str = "candidate") -> CaseRecord:
        return await self.case_memory.add_manual_case(case, status)

    async def check_manual_case(self, case: CaseRequest) -> dict[str, Any]:
        return await self.case_memory.check_manual_case(case)

    async def delete_case(self, case_id: str) -> dict[str, Any]:
        return await self.case_memory.delete_case(case_id)

    async def trust_case(self, case_id: str) -> CaseRecord:
        return await self.case_memory.trust_case(case_id)

    async def retrieve_trusted_cases(
        self,
        requirement: str,
        features: ExtractedFeatures,
        top_k: int = 3,
    ) -> list[dict]:
        return await self.case_memory.retrieve_trusted_cases(requirement, features, top_k=top_k)

    async def capture_candidate_case(
        self,
        requirement: str,
        features: ExtractedFeatures,
        candidates: list[CandidateEvaluation],
    ) -> CaseRecord:
        return await self.case_memory.capture_candidate_case(requirement, features, candidates)
