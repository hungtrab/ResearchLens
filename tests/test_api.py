import json

import pytest
from fastapi.testclient import TestClient

from researchlens.api import create_app


def test_lifespan_health_headers_and_frontend(client):
    response = client.get("/api/health")
    health = response.json()
    assert isinstance(health.pop("slm"), bool)  # true only where llama.cpp and the model file are installed
    assert health == {"status": "ok", "papers": 6, "review_model": True, "solr_enabled": False}
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=strict" in response.headers["set-cookie"]
    assert len(client.cookies["rl_profile"]) == 64
    page = client.get("/")
    assert page.status_code == 200 and "ResearchLens" in page.text


@pytest.mark.parametrize("params", [
    {"method": "unknown"}, {"limit": 0}, {"limit": 101}, {"offset": -1},
    {"year_min": 2021, "year_max": 2018}, {"year_min": 1799},
    {"q": "x" * 501}, {"q": '"unterminated'},
])
def test_search_validation(client, params):
    assert client.get("/api/search", params=params).status_code == 422


def test_search_detail_graph_and_facets_are_consistent(client):
    response = client.get("/api/search", params={"q": "graph AND retrieval", "method": "bm25"})
    assert response.status_code == 200
    result = response.json()
    assert result["total"] == 1 and result["items"][0]["id"] == "P4"
    assert result["elapsed_ms"] >= 0
    detail = client.get("/api/papers/P1").json()
    assert [p["id"] for p in detail["references"]] == ["P3"]
    assert [p["id"] for p in detail["cited_by"]] == ["P2"]
    assert detail["review_count"] == 1 and detail["keywords"]
    assert "GCN" in [e["text"] for e in detail["entities"]["methods"]]
    graph = client.get("/api/graph", params={"paper_id": "P1", "limit": 3}).json()
    assert {n["id"] for n in graph["nodes"]} == {"P1", "P2", "P3"}
    facets = client.get("/api/facets").json()
    assert facets["years"] == [2022, 2021, 2020, 2019, 2018, 2017]
    assert "SIGIR" in facets["venues"] and "graphs" in facets["topics"]
    stats = client.get("/api/stats").json()
    assert stats["papers"] == 6 and stats["edges"] == 5 and stats["reviews"] == 5


@pytest.mark.parametrize("endpoint", ["/api/papers/unknown", "/api/papers/unknown/reviews", "/api/graph?paper_id=unknown"])
def test_missing_paper_returns_404(client, endpoint):
    assert client.get(endpoint).status_code == 404


def test_reviews_and_text_analysis(client):
    result = client.get("/api/reviews", params={"limit": 2, "offset": 1}).json()
    assert result["total"] == 5 and len(result["items"]) == 2
    review = client.get("/api/papers/P1/reviews").json()["items"][0]
    assert review["observed_label"] == "positive"
    assert review["analysis"]["evidence"] and review["prediction"]["probabilities"]
    result = client.post("/api/opinion", json={"text": "The experiments are not convincing."}).json()
    assert result["label"] == "critical"
    result = client.post("/api/extract", json={"text": "BERT on SQuAD"}).json()
    assert result["datasets"][0]["text"] == "SQuAD"
    assert client.post("/api/opinion", json={"text": ""}).status_code == 422
    assert client.post("/api/extract", json={"text": "x" * 30001}).status_code == 422


@pytest.mark.parametrize("body", [{"seeds": ["missing"]}, {"method": "unknown"}, {"diversity": -0.1}, {"diversity": 1.1}, {"limit": 0}, {"limit": 51}, {"seeds": ["P1"] * 10001}])
def test_recommendation_validation(client, body):
    assert client.post("/api/recommendations", json=body).status_code == 422


def test_api_recommendation_and_coldstart(client):
    result = client.post("/api/recommendations", json={"seeds": ["P1"], "method": "content", "diversity": 0, "limit": 3}).json()
    assert not result["cold_start"] and result["items"][0]["id"] == "P2"
    assert len(result["items"]) == 3
    assert all(p["id"] != "P1" for p in result["items"])
    assert client.post("/api/recommendations", json={}).json()["cold_start"]


def test_library_export_history_and_session_isolation(client, app_config):
    assert client.get("/api/library").json()["ids"] == []
    client.get("/api/search", params={"q": "graph"})
    assert client.post("/api/library/P6").json() == {"saved": True}
    assert client.post("/api/library/P6").status_code == 200
    library = client.get("/api/library").json()
    assert library["ids"] == ["P6"] and library["history"] == ["graph"]
    export = client.get("/api/library/export", params={"format": "json"})
    assert json.loads(export.text)[0]["id"] == "P6"
    assert "attachment" in export.headers["content-disposition"]
    bibtex = client.get("/api/library/export").text
    assert "@article{P6," in bibtex and r"\&" in bibtex
    assert "An Nguyen and Binh Tran" in bibtex and "10.1234/P6" in bibtex
    with TestClient(create_app(**app_config)) as other:
        assert other.get("/api/library").json() == {"ids": [], "items": [], "history": []}
        other.cookies.set("rl_profile", client.cookies["rl_profile"])
        assert other.get("/api/library").json()["ids"] == ["P6"]
    assert client.delete("/api/library/P6").json() == {"saved": False}
    assert client.get("/api/library").json()["ids"] == []
    assert client.post("/api/library/unknown").status_code == 404


@pytest.mark.parametrize("method,path,kwargs", [
    ("post", "/api/library/P1", {}),
    ("delete", "/api/library/P1", {}),
    ("post", "/api/recommendations", {"json": {}}),
])
def test_cross_origin_writes_rejected(client, method, path, kwargs):
    response = getattr(client, method)(path, headers={"Origin": "https://attacker.example"}, **kwargs)
    assert response.status_code == 403
    assert client.get("/api/library").json()["ids"] == []


def test_same_origin_write_allowed_and_invalid_cookie_rotated(client):
    client.cookies.clear()
    client.cookies.set("rl_profile", "invalid")
    response = client.post("/api/library/P1", headers={"Origin": "http://testserver"})
    assert response.status_code == 200
    assert "rl_profile=" in response.headers["set-cookie"]
    profile = response.cookies["rl_profile"]
    assert len(profile) == 64 and profile != "invalid"


def test_missing_corpus_fails_with_actionable_message(tmp_path):
    with pytest.raises(RuntimeError, match="scripts/collect.py"):
        with TestClient(create_app(data_dir=tmp_path, db_path=tmp_path / "state.sqlite")):
            pass


def test_optional_reviews_file_is_not_required(app_config):
    (app_config["data_dir"] / "reviews.jsonl").unlink()
    with TestClient(create_app(**app_config)) as client:
        assert client.get("/api/health").json()["review_model"] is False
        assert client.get("/api/reviews").json() == {"total": 0, "items": []}
        assert client.post("/api/opinion", json={"text": "The writing is clear."}).json()["prediction"] is None


def test_jsonl_preserves_unicode_line_separator_inside_fields(app_config, papers, reviews):
    papers[0]["abstract"] += " Unicode paragraph separator \u2028 and line break marker \u2029 remain inside JSON."
    reviews[0]["text"] += " Additional evidence.\u2028The code is useful."
    for name, rows in [("papers", papers), ("reviews", reviews)]:
        (app_config["data_dir"] / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf8"
        )
    with TestClient(create_app(**app_config)) as client:
        assert "\u2028" in client.get("/api/papers/P1").json()["abstract"]
        assert "\u2028" in client.get("/api/papers/P1/reviews").json()["items"][0]["text"]


def test_large_library_seed_set_is_accepted_and_entirely_excluded(app_config, papers):
    corpus = [dict(papers[i % len(papers)], id=f"S{i}", references=[]) for i in range(60)]
    (app_config["data_dir"] / "papers.jsonl").write_text("".join(json.dumps(row) + "\n" for row in corpus))
    (app_config["data_dir"] / "reviews.jsonl").unlink()
    seeds = [row["id"] for row in corpus[:51]]
    with TestClient(create_app(**app_config)) as client:
        response = client.post("/api/recommendations", json={"seeds": seeds, "limit": 50})
        assert response.status_code == 200
        recommended = {row["id"] for row in response.json()["items"]}
        assert recommended.isdisjoint(seeds)
        assert recommended == {row["id"] for row in corpus[51:]}


def test_replacing_corpus_hides_stale_library_ids_and_exports(app_config, papers):
    with TestClient(create_app(**app_config)) as client:
        client.post("/api/library/P1")
        client.post("/api/library/P2")
        profile = client.cookies["rl_profile"]
    retained = [p for p in papers if p["id"] != "P1"]
    (app_config["data_dir"] / "papers.jsonl").write_text("".join(json.dumps(p) + "\n" for p in retained))
    (app_config["data_dir"] / "reviews.jsonl").unlink()
    with TestClient(create_app(**app_config)) as client:
        client.cookies.set("rl_profile", profile)
        library = client.get("/api/library").json()
        assert library["ids"] == ["P2"]
        assert [p["id"] for p in library["items"]] == ["P2"]
        assert [p["id"] for p in client.get("/api/library/export?format=json").json()] == ["P2"]
        assert "@article{P1," not in client.get("/api/library/export").text
        response = client.post("/api/recommendations", json={"seeds": library["ids"]})
        assert response.status_code == 200
