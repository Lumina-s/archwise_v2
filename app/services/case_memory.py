from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.schemas import CandidateEvaluation, CaseRecord, CaseRequest, ExtractedFeatures
from app.services.langchain_chroma_store import LangChainChromaStore
from app.services.langchain_embeddings import LLMClientEmbeddings
from app.services.llm_client import LLMClient


class CaseConsolidationDecision(BaseModel):
    """LLM 对'新案例 vs 现有案例库'的去重/一致性判定（Hermes 式 create-or-refine）。"""

    action: Literal["merge", "skip", "create"] = "create"
    target_case_id: str = ""
    merged_recommended_styles: list[str] = Field(default_factory=list)
    conflict_with_trusted: bool = False
    reason: str = ""


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
        return await self.add_manual_case(case, status="trusted")

    async def add_manual_case(self, case: CaseRequest, status: str = "candidate") -> CaseRecord:
        clean_status = "trusted" if status == "trusted" else "candidate"
        now = self._now()
        record = CaseRecord(
            id=self._stable_id("manual", case.requirement),
            title=case.title,
            requirement=case.requirement,
            abstract_features=self._abstract_from_text(case.requirement, case.notes),
            expected_styles=case.expected_styles,
            recommended_styles=case.expected_styles,
            notes=case.notes,
            status=clean_status,
            source="manual",
            confidence=1.0 if clean_status == "trusted" else 0.6,
            created_at=now,
            updated_at=now,
        )
        await self._upsert_records([record])
        return record

    async def check_manual_case(self, case: CaseRequest) -> dict[str, Any]:
        """提交前的去重/矛盾提示（人工新增用，复用同一套判定，但只提示不落库）。"""
        records = self._load_records()
        decision = await self._consolidate_decision(
            requirement=case.requirement,
            recommended_styles=case.expected_styles,
            domain="",
            keywords=[],
            records=records,
        )
        record_map = {record.id: record for record in records}
        target = record_map.get(decision.target_case_id)
        if decision.action == "merge" and target:
            verdict, message = "duplicate", f"与候选案例《{target.title}》高度相似，建议改为补充/合并。"
        elif decision.action == "skip" and target:
            verdict, message = "duplicate", f"与可信案例《{target.title}》重复且结论一致，可不必新增。"
        elif decision.conflict_with_trusted and target:
            verdict, message = "conflict", f"与可信案例《{target.title}》推荐分歧（{('、'.join(target.recommended_styles)) or '—'}），请确认。"
        else:
            verdict, message = "new", "未发现重复或矛盾，可以新增。"
        similar = (
            {
                "id": target.id,
                "title": target.title,
                "status": target.status,
                "recommended_styles": target.recommended_styles or target.expected_styles,
            }
            if target
            else None
        )
        return {"verdict": verdict, "message": message, "similar": similar, "reason": decision.reason}

    async def delete_case(self, case_id: str) -> dict[str, Any]:
        records = self._load_records()
        if not any(record.id == case_id for record in records):
            raise ValueError(f"案例不存在：{case_id}")
        remaining = [record for record in records if record.id != case_id]
        self._write_records(remaining)
        await self.chroma_store.adelete([case_id])
        return {"ok": True, "deleted": case_id}

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
        await self.bootstrap_seed_cases()
        recommended = [candidate.name for candidate in candidates[:3]]
        confidence = max(0.0, min(1.0, candidates[0].score / 100))
        records = self._load_records()
        record_map = {record.id: record for record in records}
        decision = await self._consolidate_decision(
            requirement=requirement,
            recommended_styles=recommended,
            domain=features.domain,
            keywords=features.keywords,
            records=records,
        )
        now = self._now()

        # skip: an existing trusted case already covers this scenario consistently
        if decision.action == "skip" and decision.target_case_id in record_map:
            return record_map[decision.target_case_id]

        # merge: consolidate into an existing candidate (kills near-duplicate cards)
        if decision.action == "merge" and decision.target_case_id in record_map:
            target = record_map[decision.target_case_id]
            merged_styles = decision.merged_recommended_styles or list(
                dict.fromkeys([*target.recommended_styles, *recommended])
            )
            merged = target.model_copy(
                update={
                    "recommended_styles": merged_styles[:5],
                    "confidence": max(target.confidence, confidence),
                    "updated_at": now,
                    "notes": "运行时自动捕获并合并相似案例，待人工确认。",
                }
            )
            await self._upsert_records([merged])
            return merged

        # create (optionally flagged as diverging from a trusted precedent)
        notes = "运行时自动捕获，待人工确认后进入可信检索集合。"
        if decision.conflict_with_trusted and decision.target_case_id in record_map:
            notes = f"运行时捕获：与可信案例《{record_map[decision.target_case_id].title}》推荐分歧，待人工核对。"
        record = CaseRecord(
            id=self._stable_id("runtime", requirement),
            title=f"{features.domain} 推荐案例",
            requirement=requirement,
            abstract_features=self._abstract_from_features(requirement, features),
            expected_styles=[],
            recommended_styles=recommended,
            notes=notes,
            status="candidate",
            source="runtime",
            confidence=confidence,
            created_at=now,
            updated_at=now,
        )
        await self._upsert_records([record])
        return record

    async def _consolidate_decision(
        self,
        *,
        requirement: str,
        recommended_styles: list[str],
        records: list[CaseRecord],
        domain: str = "",
        keywords: list[str] | None = None,
    ) -> CaseConsolidationDecision:
        """Let the LLM scan a compact index of the whole library and decide merge/skip/create."""
        index = self._library_index(records)
        if not index:
            return CaseConsolidationDecision(action="create")
        new_case = {
            "requirement": requirement[:300],
            "domain": domain,
            "keywords": (keywords or [])[:10],
            "recommended_styles": recommended_styles,
        }
        system_message = "你是软件架构案例库的去重与一致性 Agent，只输出 JSON。"
        user_prompt = (
            "判断【新案例】相对【现有案例库】应当如何处理，三选一：\n"
            "- merge：与某条 status=candidate 的候选案例是同一场景 → 合并进它（merged_recommended_styles 给出合并后的推荐，稳定风格保留、分歧的可并列）。\n"
            "- skip：与某条 status=trusted 的可信案例是同一场景且推荐一致 → 跳过不新增（target_case_id 填该可信案例）。\n"
            "- create：与已有案例都不像 → 新建。若与某条 trusted 同场景但推荐明显矛盾，仍用 create 且 conflict_with_trusted=true、target_case_id 填该可信案例。\n"
            "硬规则：绝不能 merge 进 trusted 案例（可信集合只由人工修改）。\n"
            "只返回字段：action, target_case_id, merged_recommended_styles, conflict_with_trusted, reason。\n\n"
            f"新案例：{json.dumps(new_case, ensure_ascii=False)}\n"
            f"现有案例库：{json.dumps(index, ensure_ascii=False)}\n"
        )
        try:
            decision = await self.llm_client._ainvoke_structured(
                system_message=system_message,
                user_prompt=user_prompt,
                schema=CaseConsolidationDecision,
                temperature=0.0,
                max_tokens=400,
            )
            return self._validate_decision(decision, records)
        except Exception:
            return CaseConsolidationDecision(action="create", reason="去重判定不可用，降级为新建。")

    @staticmethod
    def _library_index(records: list[CaseRecord]) -> list[dict[str, Any]]:
        return [
            {
                "id": record.id,
                "status": record.status,
                "title": record.title,
                "summary": record.requirement[:80],
                "recommended_styles": record.recommended_styles or record.expected_styles,
            }
            for record in records
        ]

    @staticmethod
    def _validate_decision(decision: CaseConsolidationDecision, records: list[CaseRecord]) -> CaseConsolidationDecision:
        record_map = {record.id: record for record in records}
        if decision.action == "merge":
            target = record_map.get(decision.target_case_id)
            if not target or target.status != "candidate":
                return CaseConsolidationDecision(action="create", reason="合并目标无效（非候选案例），降级为新建。")
        elif decision.action == "skip":
            target = record_map.get(decision.target_case_id)
            if not target or target.status != "trusted":
                return CaseConsolidationDecision(action="create", reason="跳过目标无效（非可信案例），降级为新建。")
        return decision

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
