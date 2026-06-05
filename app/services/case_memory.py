from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.models.schemas import CandidateEvaluation, CaseRecord, CaseRequest, ExtractedFeatures
from app.services.langchain_chroma_store import LangChainChromaStore
from app.services.langchain_embeddings import LLMClientEmbeddings
from app.services.llm_client import LLMClient


class CaseMemoryService:
    """Trusted/candidate case memory backed by Chroma."""

    COLLECTION_NAME = "archwise_cases"
    SIMILARITY_THRESHOLD = 0.72

    def __init__(
        self,
        llm_client: LLMClient,
        data_dir: Path | None = None,
        chroma_dir: Path | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.data_dir = data_dir or Path("data")
        self.records_file = self.data_dir / "case_records.json"
        self.seed_file = self.data_dir / "test_cases.json"
        self.chroma_dir = chroma_dir or self.data_dir / "chroma_cases"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = LLMClientEmbeddings(llm_client)
        self.chroma_store = LangChainChromaStore(
            collection_name=self.COLLECTION_NAME,
            persist_directory=self.chroma_dir,
            embeddings=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.vector_store = self.chroma_store.vector_store

    def list_records(self) -> list[CaseRecord]:
        return self._load_records()

    async def add_trusted_case(self, case: CaseRequest) -> CaseRecord:
        now = self._now()
        record = CaseRecord(
            id=self._stable_id("manual", case.requirement),
            title=case.title,
            requirement=case.requirement,
            abstract_features=self._abstract_from_text(case.requirement, case.notes),
            expected_styles=case.expected_styles,
            recommended_styles=case.expected_styles,
            notes=case.notes,
            status="trusted",
            source="manual",
            confidence=1.0,
            created_at=now,
            updated_at=now,
        )
        await self._upsert_records([record])
        return record

    async def trust_case(self, case_id: str) -> CaseRecord:
        records = self._load_records()
        record_map = {record.id: record for record in records}
        if case_id not in record_map:
            raise ValueError(f"案例不存在：{case_id}")
        record = record_map[case_id].model_copy(update={"status": "trusted", "updated_at": self._now()})
        record_map[case_id] = record
        self._write_records(list(record_map.values()))
        await self._upsert_records([record])
        return record

    async def retrieve_trusted_cases(
        self,
        requirement: str,
        features: ExtractedFeatures,
        top_k: int = 3,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        await self.bootstrap_seed_cases()
        query_text = self._abstract_from_features(requirement, features)
        docs_and_scores = await self.chroma_store.asimilarity_search_by_text(
            query=query_text,
            k=top_k,
            metadata_filter={"status": "trusted"},
        )
        min_similarity = self.SIMILARITY_THRESHOLD if threshold is None else threshold
        matches: list[dict[str, Any]] = []
        for document, distance in docs_and_scores:
            metadata = document.metadata
            similarity = round(max(0.0, min(1.0, 1.0 - float(distance))), 4)
            if similarity < min_similarity:
                continue
            matches.append(
                {
                    "id": metadata["case_id"],
                    "title": metadata["title"],
                    "requirement": metadata["requirement"],
                    "abstract_features": document.page_content,
                    "expected_styles": json.loads(metadata["expected_styles"]),
                    "recommended_styles": json.loads(metadata["recommended_styles"]),
                    "notes": metadata["notes"],
                    "source": metadata["source"],
                    "similarity": similarity,
                }
            )
        return matches

    async def capture_candidate_case(
        self,
        requirement: str,
        features: ExtractedFeatures,
        candidates: list[CandidateEvaluation],
    ) -> CaseRecord:
        if not candidates:
            raise ValueError("候选架构为空，不能捕获案例。")
        now = self._now()
        record = CaseRecord(
            id=self._stable_id("runtime", requirement),
            title=f"{features.domain} 推荐案例",
            requirement=requirement,
            abstract_features=self._abstract_from_features(requirement, features),
            expected_styles=[],
            recommended_styles=[candidate.name for candidate in candidates[:3]],
            notes="运行时自动捕获，待人工确认后进入可信检索集合。",
            status="candidate",
            source="runtime",
            confidence=max(0.0, min(1.0, candidates[0].score / 100)),
            created_at=now,
            updated_at=now,
        )
        await self._upsert_records([record])
        return record

    async def bootstrap_seed_cases(self) -> None:
        existing_seed_ids = {
            record.id
            for record in self._load_records()
            if record.source == "seed"
        }
        payload = json.loads(self.seed_file.read_text(encoding="utf-8"))
        records: list[CaseRecord] = []
        now = self._now()
        for item in payload:
            case = CaseRequest(**item)
            case_id = self._stable_id("seed", case.requirement)
            if case_id in existing_seed_ids:
                continue
            records.append(
                CaseRecord(
                    id=case_id,
                    title=case.title,
                    requirement=case.requirement,
                    abstract_features=self._abstract_from_text(case.requirement, case.notes),
                    expected_styles=case.expected_styles,
                    recommended_styles=case.expected_styles,
                    notes=case.notes,
                    status="trusted",
                    source="seed",
                    confidence=1.0,
                    created_at=now,
                    updated_at=now,
                )
            )
        if records:
            await self._upsert_records(records)

    async def _upsert_records(self, records: list[CaseRecord]) -> None:
        await self.chroma_store.aupsert_texts(
            ids=[record.id for record in records],
            texts=[record.abstract_features for record in records],
            metadatas=[self._metadata(record) for record in records],
        )
        record_map = {record.id: record for record in self._load_records()}
        for record in records:
            record_map[record.id] = record
        self._write_records(list(record_map.values()))

    def _load_records(self) -> list[CaseRecord]:
        if not self.records_file.exists():
            return []
        payload = json.loads(self.records_file.read_text(encoding="utf-8"))
        return [CaseRecord(**item) for item in payload]

    def _write_records(self, records: list[CaseRecord]) -> None:
        records.sort(key=lambda item: (item.source, item.title, item.id))
        self.records_file.write_text(
            json.dumps([record.model_dump() for record in records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _metadata(record: CaseRecord) -> dict[str, str | float]:
        return {
            "title": record.title,
            "case_id": record.id,
            "requirement": record.requirement,
            "expected_styles": json.dumps(record.expected_styles, ensure_ascii=False),
            "recommended_styles": json.dumps(record.recommended_styles, ensure_ascii=False),
            "notes": record.notes,
            "status": record.status,
            "source": record.source,
            "confidence": record.confidence,
            "updated_at": record.updated_at,
        }

    @staticmethod
    def _abstract_from_features(requirement: str, features: ExtractedFeatures) -> str:
        return "\n".join(
            [
                f"需求：{requirement}",
                f"领域：{features.domain}",
                f"关键词：{'、'.join(features.keywords)}",
                f"业务能力：{'、'.join(features.business_capabilities)}",
                f"架构驱动：{'、'.join(features.architecture_drivers)}",
                f"数据流：{features.data_flow}",
                f"质量属性：{json.dumps(features.quality_attributes, ensure_ascii=False, sort_keys=True)}",
                f"约束：{json.dumps(features.constraints, ensure_ascii=False, sort_keys=True)}",
            ]
        )

    @staticmethod
    def _abstract_from_text(requirement: str, notes: str) -> str:
        return "\n".join([f"需求：{requirement}", f"说明：{notes}"])

    @staticmethod
    def _stable_id(source: str, requirement: str) -> str:
        digest = hashlib.sha256(f"{source}:{requirement}".encode("utf-8")).hexdigest()[:16]
        return f"{source}-{digest}"

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
