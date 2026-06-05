from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.models.schemas import ArchitectureStyle, RecommendationResponse
from app.services.langchain_chroma_store import LangChainChromaStore
from app.services.langchain_embeddings import LLMClientEmbeddings
from app.services.llm_client import LLMClient


class RecommendationCacheService:
    COLLECTION_NAME = "archwise_recommendation_cache"
    VECTOR_SIMILARITY_THRESHOLD = 0.96
    TEXT_SIMILARITY_THRESHOLD = 0.9

    def __init__(
        self,
        llm_client: LLMClient,
        data_dir: Path | None = None,
        chroma_dir: Path | None = None,
    ) -> None:
        self.data_dir = data_dir or Path("data")
        self.records_file = self.data_dir / "recommendation_cache.json"
        self.chroma_dir = chroma_dir or self.data_dir / "chroma_recommendation_cache"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = LLMClientEmbeddings(llm_client)
        self.chroma_store = LangChainChromaStore(
            collection_name=self.COLLECTION_NAME,
            persist_directory=self.chroma_dir,
            embeddings=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.vector_store = self.chroma_store.vector_store

    async def lookup(
        self,
        *,
        requirement: str,
        top_k: int,
        topology_options: dict[str, Any] | None,
        styles: list[ArchitectureStyle],
    ) -> RecommendationResponse | None:
        records = self._read_records()
        signature = self._signature(top_k, topology_options, styles)
        query_text = self._query_text(requirement, signature)
        exact_id = self._stable_id(query_text)
        exact_record = records.get(exact_id)
        if exact_record:
            return self._response_from_record(exact_record, "exact", 1.0)

        normalized_requirement = self._normalize_requirement(requirement)
        text_candidates = [
            record
            for record in records.values()
            if record.get("signature") == signature
            and SequenceMatcher(
                None,
                normalized_requirement,
                self._normalize_requirement(record["requirement"]),
            ).ratio()
            >= self.TEXT_SIMILARITY_THRESHOLD
        ]
        if not text_candidates:
            return None

        docs_and_scores = await self.chroma_store.asimilarity_search_by_text(
            query=query_text,
            k=1,
            metadata_filter={"signature": signature},
        )
        for document, distance in docs_and_scores:
            cache_id = document.metadata["cache_id"]
            record = records.get(cache_id)
            if not record:
                raise ValueError(f"Recommendation cache vector points to missing record: {cache_id}")
            vector_similarity = round(max(0.0, min(1.0, 1.0 - float(distance))), 4)
            text_similarity = SequenceMatcher(
                None,
                normalized_requirement,
                self._normalize_requirement(record["requirement"]),
            ).ratio()
            if (
                vector_similarity >= self.VECTOR_SIMILARITY_THRESHOLD
                and text_similarity >= self.TEXT_SIMILARITY_THRESHOLD
            ):
                return self._response_from_record(record, "semantic", vector_similarity)
        return None

    async def store(
        self,
        *,
        response: RecommendationResponse,
        requirement: str,
        top_k: int,
        topology_options: dict[str, Any] | None,
        styles: list[ArchitectureStyle],
    ) -> None:
        records = self._read_records()
        signature = self._signature(top_k, topology_options, styles)
        query_text = self._query_text(requirement, signature)
        cache_id = self._stable_id(query_text)
        now = datetime.now(UTC).isoformat()
        records[cache_id] = {
            "id": cache_id,
            "requirement": requirement,
            "signature": signature,
            "response": response.model_dump(),
            "updated_at": now,
        }
        self._write_records(records)
        await self.chroma_store.aupsert_texts(
            ids=[cache_id],
            texts=[query_text],
            metadatas=[
                {
                    "cache_id": cache_id,
                    "requirement": requirement,
                    "signature": signature,
                    "updated_at": now,
                }
            ],
        )

    def _response_from_record(self, record: dict[str, Any], hit_type: str, similarity: float) -> RecommendationResponse:
        response = RecommendationResponse.model_validate(record["response"]).model_copy(deep=True)
        response.trace = list(dict.fromkeys([*response.trace, f"推荐缓存命中：{hit_type}"]))
        response.decision_trace = dict(response.decision_trace)
        response.decision_trace["recommendation_cache_evidence"] = {
            "hit": True,
            "hit_type": hit_type,
            "similarity": similarity,
            "cache_id": record["id"],
            "policy": "普通推荐接口复用相同或高度相似需求的完整推荐响应。",
        }
        return response

    def _read_records(self) -> dict[str, Any]:
        if not self.records_file.exists():
            return {}
        payload = json.loads(self.records_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Recommendation cache must be a JSON object: {self.records_file}")
        return payload

    def _write_records(self, records: dict[str, Any]) -> None:
        temp_path = self.records_file.with_suffix(f"{self.records_file.suffix}.tmp")
        temp_path.write_text(json.dumps(records, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.records_file)

    @classmethod
    def _query_text(cls, requirement: str, signature: str) -> str:
        return "\n".join([f"需求: {cls._normalize_requirement(requirement)}", f"签名: {signature}"])

    @staticmethod
    def _signature(
        top_k: int,
        topology_options: dict[str, Any] | None,
        styles: list[ArchitectureStyle],
    ) -> str:
        style_payload = [
            {
                "id": style.id,
                "name": style.name,
                "category": style.category,
                "quality_scores": style.quality_scores,
            }
            for style in sorted(styles, key=lambda item: item.id)
        ]
        payload = {
            "top_k": top_k,
            "topology_options": topology_options or {},
            "styles": style_payload,
        }
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_requirement(requirement: str) -> str:
        return re.sub(r"\s+", "", requirement.strip().lower())

    @staticmethod
    def _stable_id(query_text: str) -> str:
        digest = hashlib.sha256(query_text.encode("utf-8")).hexdigest()[:16]
        return f"recommendation-{digest}"
