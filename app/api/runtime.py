from __future__ import annotations

from app.knowledge.repository import KnowledgeRepository
from app.services.recommendation_service import RecommendationService

repository = KnowledgeRepository()
recommendation_service = RecommendationService(repository)
knowledge_service = recommendation_service.knowledge_service
