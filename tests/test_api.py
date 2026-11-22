from fastapi.testclient import TestClient

from complibot.api import create_app, role_for_key


def client_for(bot):
    settings = bot.settings.with_(internal_api_keys=frozenset({"secret-internal"}))
    bot.settings = settings
    return TestClient(create_app(bot=bot, settings=settings))


def test_health(bot):
    with client_for(bot) as c:
        body = c.get("/health").json()
        assert body["status"] == "ok" and body["chunks"] > 0 and body["llm"] == "fake:fake"


def test_ask_returns_cited_answer(bot):
    with client_for(bot) as c:
        r = c.post("/ask", json={"question": "How many business days to stop using data after consent is withdrawn?"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "answered"
        assert body["citations"][0]["section_id"] == "2.3"
        assert body["role"] == "public"


def test_role_comes_from_api_key_not_body(bot):
    q = {"question": "What recovery time objective applies to each critical system?"}
    with client_for(bot) as c:
        public = c.post("/ask", json={**q, "role": "internal"}).json()  # body field is ignored
        assert public["role"] == "public"
        assert not any("tech-risk" in i for i in public["retrieved_ids"])
        internal = c.post("/ask", json=q, headers={"X-API-Key": "secret-internal"}).json()
        assert internal["role"] == "internal" and internal["citations"][0]["section_id"] == "5.1.1"
        wrong = c.post("/ask", json=q, headers={"X-API-Key": "guess"}).json()
        assert wrong["role"] == "public"


def test_search_endpoint(bot):
    with client_for(bot) as c:
        hits = c.post("/search", json={"question": "encryption laptops", "top_k": 2}).json()
        assert len(hits) <= 2 and hits[0]["section_id"] == "3.1.2"


def test_validation_errors(bot):
    with client_for(bot) as c:
        assert c.post("/ask", json={"question": ""}).status_code == 422
        assert c.post("/ask", json={"question": "valid question", "top_k": 99}).status_code == 422


def test_missing_index_returns_503(tmp_path):
    from complibot.config import Settings

    app = create_app(settings=Settings(index_dir=tmp_path / "nope"))
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/ask", json={"question": "anything at all"}).status_code == 503


def test_role_for_key():
    from complibot.config import Settings

    s = Settings(internal_api_keys=frozenset({"k1"}))
    assert role_for_key("k1", s) == "internal"
    assert role_for_key(None, s) == "public"
    assert role_for_key("", s) == "public"
