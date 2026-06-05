import asyncio

from app.models.schemas import ExtractedFeatures
from app.knowledge.styles import load_default_styles
from app.services.knowledge_graph import KnowledgeGraphService
from app.services.knowledge_service import KnowledgeService
from app.services.neo4j_aura import Neo4jAuraService


class RecordingSession:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))
        return []


class RecordingTx:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))


class FakeEmbeddings:
    async def aembed_documents(self, texts):
        return [[float(index), 1.0, 0.0] for index, _ in enumerate(texts)]

    async def aembed_query(self, text):
        return [0.2, 0.8, 0.0]


class FakeNeo4j:
    configured = True

    def __init__(self):
        self.embedding_payload = None
        self.query_embedding = None

    def fetch_topology_node_records(self):
        return {
            "BusinessCapability": [
                {"label": "BusinessCapability", "name": "支付结算", "context": {"components": ["支付服务"]}}
            ],
            "ArchitectureComponent": [
                {"label": "ArchitectureComponent", "name": "支付服务", "context": {"capabilities": ["支付结算"]}}
            ],
            "DataStore": [],
        }

    def upsert_topology_embeddings(self, payload, dimension):
        self.embedding_payload = {"payload": payload, "dimension": dimension}
        return {"ok": True, "vectors_indexed": len(payload), "dimension": dimension}

    def retrieve_topology_knowledge(self, requirement, keywords, qualities, business_capabilities, domain, query_embedding=None):
        self.query_embedding = query_embedding
        return {
            "components": ["支付服务"],
            "stores": [],
            "edges": [],
            "scenarios": [],
            "capabilities": ["支付结算"],
            "vector_matches": [{"label": "ArchitectureComponent", "name": "支付服务", "score": 0.92}],
        }


def test_neo4j_vector_indexes_use_native_vector_indexes():
    session = RecordingSession()

    Neo4jAuraService._create_vector_indexes(session, 3)

    statements = "\n".join(query for query, _ in session.calls)
    assert "CREATE VECTOR INDEX topology_business_capability_embedding" in statements
    assert "FOR (n:BusinessCapability) ON (n.embedding)" in statements
    assert "`vector.dimensions`: 3" in statements
    assert "`vector.similarity_function`: 'cosine'" in statements


def test_neo4j_topology_embedding_write_sets_vector_properties():
    tx = RecordingTx()

    Neo4jAuraService._set_topology_embeddings(
        tx,
        "ArchitectureComponent",
        [{"name": "支付服务", "text": "节点名称: 支付服务", "embedding": [1.0, 0.0, 0.0]}],
    )

    query, params = tx.calls[0]
    assert "MATCH (n:ArchitectureComponent {name: item.name})" in query
    assert "n.embedding = item.embedding" in query
    assert params["items"][0]["name"] == "支付服务"


def test_neo4j_style_write_persists_full_architecture_style_payload():
    tx = RecordingTx()
    style = load_default_styles()[0]

    Neo4jAuraService._merge_style(tx, style)

    query, params = tx.calls[0]
    assert "s.strengths = $strengths" in query
    assert "s.weaknesses = $weaknesses" in query
    assert "s.topology = $topology" in query
    assert "s.rules_json = $rules_json" in query
    assert params["strengths"] == style.strengths
    assert params["weaknesses"] == style.weaknesses
    assert params["topology"] == style.topology
    assert params["rules_json"]


def test_knowledge_graph_reindexes_topology_vectors_through_embeddings():
    service = KnowledgeGraphService()
    service.neo4j = FakeNeo4j()
    service.embeddings = FakeEmbeddings()

    result = asyncio.run(service.reindex_topology_vectors())

    assert result == {"ok": True, "vectors_indexed": 2, "dimension": 3}
    assert service.neo4j.embedding_payload["dimension"] == 3
    assert [item["name"] for item in service.neo4j.embedding_payload["payload"]] == ["支付结算", "支付服务"]


def test_knowledge_graph_retrieve_passes_query_embedding_to_neo4j():
    service = KnowledgeGraphService()
    service.neo4j = FakeNeo4j()
    service.embeddings = FakeEmbeddings()
    features = ExtractedFeatures(
        domain="电商交易",
        keywords=["支付"],
        business_capabilities=["支付结算"],
        architecture_drivers=["高可用"],
        topology_expectations={"must_have_components": [], "must_have_relations": [], "quality_infrastructure": []},
        quality_attributes={
            "concurrency": 0.4,
            "realtime": 0.2,
            "reliability": 0.8,
            "scalability": 0.6,
            "data_intensity": 0.3,
            "ai_reasoning": 0.0,
        },
        constraints={},
        data_flow="transactional",
        ambiguity_notes=[],
    )

    result = asyncio.run(service.retrieve_topology_knowledge("支付系统", features))

    assert service.neo4j.query_embedding == [0.2, 0.8, 0.0]
    assert result["vector_matches"][0]["name"] == "支付服务"


class FakeConfiguredNeo4j:
    configured = True

    def __init__(self):
        self.styles = [load_default_styles()[0]]
        self.synced_styles = []

    def fetch_styles(self):
        return self.styles

    def sync_styles(self, styles):
        self.synced_styles.extend(styles)
        return {"ok": True, "styles_synced": len(styles)}


class FakeGraphService:
    def __init__(self):
        self.neo4j = FakeConfiguredNeo4j()

    def list_styles(self):
        return self.neo4j.fetch_styles()

    def add_style(self, style):
        result = self.neo4j.sync_styles([style])
        if not result.get("ok"):
            raise RuntimeError(result)
        return style


def test_knowledge_service_uses_neo4j_style_store_when_configured():
    graph_service = FakeGraphService()
    service = KnowledgeService(FakeEmbeddings(), graph_service=graph_service)
    style = load_default_styles()[1]

    assert service.list_styles() == graph_service.neo4j.styles
    assert service.add_style(style) == style
    assert graph_service.neo4j.synced_styles == [style]
