from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import (
    CandidateCaseCaptureRequest,
    CaseRequest,
    CaseRetrievalRequest,
    CaseTrustRequest,
    KnowledgeStyleRequest,
    ManualCaseRequest,
    TopologyKnowledgeRequest,
    TopologyMergeRequest,
    TopologyNormalizeRequest,
)
from app.services.knowledge_service import KnowledgeService


def build_knowledge_router(knowledge_service: KnowledgeService) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["knowledge"])

    @router.get("/styles")
    async def styles():
        return knowledge_service.list_styles()

    @router.get("/knowledge/graph")
    async def graph():
        return knowledge_service.build_style_graph()

    @router.get("/knowledge/topology-graph")
    async def topology_graph():
        return knowledge_service.build_topology_graph()

    @router.get("/knowledge/neo4j/status")
    async def neo4j_status():
        return knowledge_service.neo4j_status()

    @router.post("/knowledge/neo4j/sync")
    async def sync_neo4j():
        return knowledge_service.start_neo4j_sync()

    @router.get("/knowledge/neo4j/sync/status")
    async def neo4j_sync_status():
        return knowledge_service.neo4j_sync_status()

    @router.post("/knowledge/neo4j/rebuild-topology")
    async def rebuild_neo4j_topology():
        return await knowledge_service.rebuild_domain_topology()

    @router.post("/knowledge/neo4j/reindex-vectors")
    async def reindex_neo4j_vectors():
        return await knowledge_service.reindex_topology_vectors()

    @router.get("/knowledge/neo4j/duplicates")
    async def neo4j_duplicate_like_nodes():
        return await knowledge_service.detect_duplicate_like_nodes()

    @router.post("/knowledge/styles")
    async def add_style(payload: KnowledgeStyleRequest):
        return knowledge_service.add_style(payload.style)

    @router.get("/cases")
    async def cases():
        return knowledge_service.list_cases()

    @router.post("/knowledge/cases")
    async def add_case(payload: ManualCaseRequest):
        case = CaseRequest(
            title=payload.title,
            requirement=payload.requirement,
            expected_styles=payload.expected_styles,
            notes=payload.notes,
        )
        return await knowledge_service.add_manual_case(case, status="trusted" if payload.as_trusted else "candidate")

    @router.post("/knowledge/cases/check")
    async def check_case(payload: CaseRequest):
        return await knowledge_service.check_manual_case(payload)

    @router.post("/knowledge/cases/delete")
    async def delete_case(payload: CaseTrustRequest):
        return await knowledge_service.delete_case(payload.case_id)

    @router.post("/knowledge/cases/trust")
    async def trust_case(payload: CaseTrustRequest):
        return await knowledge_service.trust_case(payload.case_id)

    @router.post("/internal/cases/retrieve")
    async def retrieve_internal_cases(payload: CaseRetrievalRequest):
        return await knowledge_service.retrieve_trusted_cases(payload.requirement, payload.features, payload.top_k)

    @router.post("/internal/cases/capture")
    async def capture_internal_case(payload: CandidateCaseCaptureRequest):
        return await knowledge_service.capture_candidate_case(
            payload.requirement,
            payload.features,
            payload.candidates,
        )

    @router.post("/internal/topology/retrieve")
    async def retrieve_internal_topology(payload: TopologyKnowledgeRequest):
        return await knowledge_service.graph_service.retrieve_topology_knowledge(payload.requirement, payload.features)

    @router.post("/internal/topology/normalize")
    async def normalize_internal_topology(payload: TopologyNormalizeRequest):
        return await knowledge_service.graph_service.normalize_topology_patch(
            payload.patch,
            payload.requirement,
            payload.features,
            payload.graph_knowledge,
            payload.coverage,
        )

    @router.post("/internal/topology/merge")
    async def merge_internal_topology(payload: TopologyMergeRequest):
        return knowledge_service.graph_service.merge_topology_patch(
            payload.requirement,
            payload.features,
            payload.patch,
        )

    return router
