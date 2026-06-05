from fastapi.testclient import TestClient

from app.knowledge_main import app as knowledge_app
from app.llm_gateway_main import app as llm_gateway_app
from app.main import app as main_app


def test_main_app_exposes_llm_gateway_and_knowledge_boundaries():
    client = TestClient(main_app)
    openapi = client.get("/openapi.json").json()
    paths = openapi["paths"]

    assert "/api/llm/status" in paths
    assert "/api/llm/embeddings" in paths
    assert "/api/knowledge/cases/trust" in paths
    assert "/api/knowledge/neo4j/reindex-vectors" in paths
    assert paths["/api/llm/status"]["get"]["tags"] == ["llm-gateway"]
    assert paths["/api/styles"]["get"]["tags"] == ["knowledge"]


def test_llm_gateway_can_run_as_independent_asgi_app(monkeypatch):
    async def fake_ping(self):
        return {"configured": True, "ok": True, "model": "test-model", "base_url": "http://test"}

    async def fake_embed_texts(self, texts):
        return [[float(index), 1.0] for index, _ in enumerate(texts)]

    monkeypatch.setattr("app.services.llm_client.LLMClient.ping", fake_ping)
    monkeypatch.setattr("app.services.llm_client.LLMClient.embed_texts", fake_embed_texts)

    client = TestClient(llm_gateway_app)
    assert client.get("/api/llm/status").json()["ok"] is True
    assert client.post("/api/llm/embeddings", json={"texts": ["a", "b"]}).json() == {
        "vectors": [[0.0, 1.0], [1.0, 1.0]]
    }


def test_knowledge_service_can_run_as_independent_asgi_app():
    client = TestClient(knowledge_app)

    styles = client.get("/api/styles")
    assert styles.status_code == 200
    assert len(styles.json()) >= 10

    graph = client.get("/api/knowledge/graph")
    assert graph.status_code == 200
    assert graph.json()["nodes"]
