from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import CaseRequest, CaseTrustRequest, KnowledgeStyleRequest
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
    async def add_case(payload: CaseRequest):
        return await knowledge_service.add_trusted_case(payload)

    @router.post("/knowledge/cases/trust")
    async def trust_case(payload: CaseTrustRequest):
        return await knowledge_service.trust_case(payload.case_id)

    return router
