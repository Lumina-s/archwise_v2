import asyncio

from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import CandidateEvaluation, ExtractedFeatures
from app.services.exceptions import RequirementParsingError
from app.services.recommendation_service import RecommendationService


client = TestClient(app)


def llm_features(
    domain: str = "即时通信",
    data_flow: str = "event_stream",
    keywords: list[str] | None = None,
    business_capabilities: list[str] | None = None,
    quality_attributes: dict[str, float] | None = None,
) -> ExtractedFeatures:
    quality_attributes = quality_attributes or {
        "concurrency": 0.95,
        "realtime": 0.9,
        "reliability": 0.82,
        "scalability": 0.85,
        "data_intensity": 0.45,
        "ai_reasoning": 0.0,
    }
    return ExtractedFeatures(
        domain=domain,
        keywords=keywords or ["即时通讯", "万人在线", "实时", "可靠", "扩展"],
        business_capabilities=business_capabilities or ["用户体系", "消息通信", "在线状态", "通知提醒", "视频通话"],
        architecture_drivers=["高并发", "实时性", "高可用", "弹性伸缩"],
        topology_expectations={
            "must_have_components": ["消息服务", "状态服务", "事件总线"],
            "must_have_relations": ["消息服务->事件总线"],
            "quality_infrastructure": ["负载均衡", "缓存集群", "监控服务"],
        },
        quality_attributes=quality_attributes,
        constraints={
            "scale_mentions": ["万人在线"],
            "deployment": ["跨平台"],
            "requires_high_availability": True,
            "requires_future_extension": True,
        },
        data_flow=data_flow,
        ambiguity_notes=[],
    )


def patch_llm(monkeypatch, features: ExtractedFeatures | None = None) -> None:
    async def fake_extract_features(self, requirement):
        return features or llm_features()

    async def fake_review_candidates(self, requirement, extracted_features, candidates):
        return ["DeepSeek 复核候选排序合理"]

    async def fake_generate_report(self, requirement, extracted_features, candidates):
        return "- **高并发** 场景需要异步削峰和服务拆分。"

    async def fake_stream_report(self, requirement, extracted_features, candidates):
        yield "- **高并发** 场景需要异步削峰和服务拆分。"

    async def fake_extract_capabilities(self, requirement, extracted_features):
        return []

    async def fake_review_topology_coverage_gap(self, requirement, extracted_features, graph_knowledge, request_timeout=None):
        return None

    async def fake_propose_patch(self, requirement, extracted_features, coverage, graph_knowledge, request_timeout=None):
        return None

    async def fake_recommend_architectures(self, requirement, extracted_features, styles, top_k, case_references=None):
        style_map = {item["id"]: item for item in styles}
        if extracted_features.data_flow == "pipeline":
            style_ids = ["pipe_filter", "microservices", "serverless"]
        elif extracted_features.data_flow == "request_response":
            style_ids = ["layered", "mvc", "monolithic"]
        else:
            style_ids = ["event_driven", "microservices", "cqrs"]
        candidates = []
        for index, style_id in enumerate(style_ids[:top_k]):
            style = style_map[style_id]
            candidates.append(
                CandidateEvaluation(
                    style_id=style_id,
                    name=style["name"],
                    score=94 - index * 4,
                    raw_score=94 - index * 4,
                    recommendation_role="核心推荐" if index == 0 else "备选方案",
                    confidence="高" if index == 0 else "中高",
                    matched_reasons=["测试替身生成候选"],
                    risks=[],
                    deductions=[],
                    quality_scores=style["quality_scores"],
                )
            )
        return candidates, {
            "composition_needed": False,
            "primary_style": candidates[0].name,
            "supporting_styles": [],
            "reason": "测试替身组合建议",
            "triggers": [],
            "overengineering_warnings": [],
            "review_notes": ["DeepSeek 复核候选排序合理"],
        }

    async def fake_embed_texts(self, texts):
        return [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr("app.services.llm_client.LLMClient.extract_features", fake_extract_features)
    monkeypatch.setattr("app.services.llm_client.LLMClient.recommend_architectures", fake_recommend_architectures)
    monkeypatch.setattr("app.services.llm_client.LLMClient.review_candidates", fake_review_candidates)
    monkeypatch.setattr("app.services.llm_client.LLMClient.generate_report", fake_generate_report)
    monkeypatch.setattr("app.services.llm_client.LLMClient.stream_report", fake_stream_report)
    monkeypatch.setattr("app.services.llm_client.LLMClient.extract_capabilities", fake_extract_capabilities)
    monkeypatch.setattr("app.services.llm_client.LLMClient.review_topology_coverage_gap", fake_review_topology_coverage_gap)
    monkeypatch.setattr("app.services.llm_client.LLMClient.propose_topology_knowledge_patch", fake_propose_patch)
    monkeypatch.setattr("app.services.llm_client.LLMClient.embed_texts", fake_embed_texts)


def test_recommend_im_returns_at_least_three_candidates(monkeypatch):
    patch_llm(monkeypatch)
    response = client.post(
        "/api/recommend",
        json={
            "requirement": "开发一个跨平台即时通讯系统，支持万人同时在线，消息实时可靠，后续扩展视频通话",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["candidates"]) == 3
    assert payload["final_recommendation"]["name"] in {"事件驱动架构", "微服务架构", "CQRS 架构"}
    assert payload["features"]["data_flow"] == "event_stream"
    assert payload["decision_trace"]["rule_evidence"]["enabled"] is True
    assert payload["decision_trace"]["case_memory_evidence"]["captured_candidate_case_id"].startswith("runtime-")
    assert payload["decision_trace"]["local_matcher_evidence"]
    assert "本地规则提取硬信号" not in " ".join(payload["trace"])


def test_runtime_candidate_case_can_be_trusted(monkeypatch):
    patch_llm(monkeypatch)
    response = client.post(
        "/api/recommend",
        json={
            "requirement": "开发跨平台即时通讯系统，支持万人在线，消息实时可靠，后续扩展视频通话",
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    case_id = response.json()["decision_trace"]["case_memory_evidence"]["captured_candidate_case_id"]

    trust_response = client.post("/api/knowledge/cases/trust", json={"case_id": case_id})
    assert trust_response.status_code == 200
    assert trust_response.json()["status"] == "trusted"

    cases_response = client.get("/api/cases")
    assert cases_response.status_code == 200
    records = cases_response.json()
    assert any(item["id"] == case_id and item["status"] == "trusted" for item in records)


def test_recommend_returns_prompt_when_deepseek_parse_unavailable(monkeypatch):
    async def fake_extract_features(self, requirement):
        self.last_error = "DeepSeek API Key 未配置，无法进行需求解析。"
        return None

    monkeypatch.setattr("app.services.llm_client.LLMClient.extract_features", fake_extract_features)

    response = client.post(
        "/api/recommend",
        json={"requirement": "开发一个在线教育平台，支持直播和录播", "top_k": 3},
    )

    assert response.status_code == 503
    assert "需求解析失败" in response.json()["detail"]
    assert "DeepSeek API Key 未配置" in response.json()["detail"]


def test_styles_include_required_knowledge_base_size():
    response = client.get("/api/styles")
    assert response.status_code == 200
    assert len(response.json()) >= 10


def test_knowledge_graph_has_nodes_and_edges():
    response = client.get("/api/knowledge/graph")
    assert response.status_code == 200
    payload = response.json()
    assert payload["nodes"]
    assert payload["edges"]


def test_neo4j_status_endpoint_is_available():
    response = client.get("/api/knowledge/neo4j/status")
    assert response.status_code == 200
    payload = response.json()
    assert "configured" in payload


def test_service_generates_matrix_and_trace(monkeypatch):
    patch_llm(
        monkeypatch,
        llm_features(
            "数据分析",
            "pipeline",
            keywords=["日志", "ETL", "清洗", "转换", "报表", "分析", "流水线"],
            business_capabilities=["数据采集", "数据管道", "离线分析", "数据可视化", "任务调度"],
            quality_attributes={
                "concurrency": 0.35,
                "realtime": 0.2,
                "reliability": 0.65,
                "scalability": 0.7,
                "data_intensity": 0.95,
                "ai_reasoning": 0.0,
            },
        ),
    )

    service = RecommendationService()
    result = asyncio.run(service.recommend("日志 ETL 清洗、转换和报表分析平台，需要流水线处理", top_k=3))

    assert result.comparison_matrix
    assert result.trace
    assert result.final_recommendation.name in {"管道-过滤器架构", "微服务架构", "Serverless 架构"}
    assert any("HybridReasoningOrchestrator" in item for item in result.trace)
    assert any("DeepSeek 主解析" in item for item in result.trace)
    assert any("知识图谱" in item for item in result.trace)
    assert result.decision_trace["rule_evidence"]["enabled"] is True
    assert result.decision_trace["case_memory_evidence"]["captured_candidate_case_id"].startswith("runtime-")
    assert result.decision_trace["composition_evidence"]["source"] == "composition_recommender"
    assert not any("本地规则提取硬信号" in item for item in result.trace)


def test_topology_repair_graph_repeats_until_coverage_passes(monkeypatch):
    service = RecommendationService()
    features = llm_features(
        "电商交易",
        "transactional",
        keywords=["订单", "支付"],
        business_capabilities=["订单管理", "支付结算"],
        quality_attributes={
            "concurrency": 0.4,
            "realtime": 0.2,
            "reliability": 0.7,
            "scalability": 0.5,
            "data_intensity": 0.4,
            "ai_reasoning": 0.0,
        },
    ).model_copy(
        update={
            "topology_expectations": {
                "must_have_components": ["订单服务", "支付服务"],
                "must_have_relations": ["订单服务->支付服务"],
                "quality_infrastructure": [],
            }
        }
    )
    ctx = service.orchestrator._context(
        "电商订单支付系统",
        service.knowledge_service.list_styles(),
        3,
        topology_options={"fast_mode": True, "repair_max_rounds": 2, "llm_timeout_seconds": 3},
    )
    patch_calls = {"count": 0}

    async def fake_review_gap(requirement, extracted_features, graph_knowledge, request_timeout=None):
        return None

    async def fake_propose_patch(requirement, extracted_features, coverage, graph_knowledge, request_timeout=None):
        patch_calls["count"] += 1
        if patch_calls["count"] == 1:
            missing_components = list(coverage.get("missing_components", []))
            return {
                "scenario_id": "ecommerce_payment",
                "scenario_name": "电商支付",
                "capabilities": [
                    {"name": name, "components": missing_components, "stores": [], "edges": []}
                    for name in coverage.get("missing_capabilities", [])
                ],
                "components": missing_components,
                "stores": [],
                "edges": [],
                "reason": "补齐业务能力和组件",
            }
        edges = []
        for relation in coverage.get("missing_relations", []):
            if "->" not in relation:
                continue
            source, target = relation.split("->", 1)
            edges.append({"source": source, "target": target, "label": "调用", "kind": "sync"})
        return {
            "scenario_id": "ecommerce_payment",
            "scenario_name": "电商支付",
            "capabilities": [],
            "components": [],
            "stores": [],
            "edges": edges,
            "reason": "补齐订单到支付关系",
        }

    monkeypatch.setattr(service.llm_client, "review_topology_coverage_gap", fake_review_gap)
    monkeypatch.setattr(service.llm_client, "propose_topology_knowledge_patch", fake_propose_patch)

    graph_knowledge, repair_trace = asyncio.run(
        service.orchestrator._repair_topology_knowledge(
            ctx,
            features,
            {"components": [], "stores": [], "edges": [], "scenarios": [], "capabilities": []},
            features.business_capabilities,
        )
    )

    actions = [item["action"] for item in repair_trace]
    assert actions.count("coverage_check") == 2
    assert actions.count("llm_patch_merged") == 2
    assert actions[-1] == "llm_patch_merged"
    assert repair_trace[-1]["coverage_after"]["score"] >= service.orchestrator.TOPOLOGY_COVERAGE_THRESHOLD
    assert patch_calls["count"] == 2
    assert any(edge["source"] == "订单服务" and edge["target"] == "支付服务" for edge in graph_knowledge["edges"])


def test_service_raises_when_deepseek_parse_unavailable(monkeypatch):
    async def fake_extract_features(self, requirement):
        self.last_error = "DeepSeek 返回 JSON 不符合 Schema。"
        return None

    monkeypatch.setattr("app.services.llm_client.LLMClient.extract_features", fake_extract_features)

    service = RecommendationService()
    try:
        asyncio.run(service.recommend("开发一个系统", top_k=3))
    except RequirementParsingError as exc:
        assert "需求解析失败" in str(exc)
    else:
        raise AssertionError("Expected RequirementParsingError")
