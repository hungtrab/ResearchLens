"""Project and reader lifecycle checks with independent anonymous profiles."""
import json
import math

import pytest
from fastapi.testclient import TestClient

from researchlens.api import create_app
from researchlens.store import Store


def create_project(client, title="Graph research", brief="Compare graph neural networks and retrieval."):
    response = client.post("/api/projects", json={"title": title, "brief": brief})
    assert response.status_code == 201
    return response.json()


def test_project_crud_membership_and_reading_hydration(client):
    assert client.get("/api/projects").json() == {"items": []}
    project = create_project(client, "  Graph research  ")
    assert project["title"] == "Graph research" and project["paper_ids"] == []
    base = "/api/projects/" + project["id"]
    for _ in range(2):
        assert client.post(base + "/papers/P1").json() == {"added": True}
    client.put("/api/papers/P1/reading", json={"status": "reading", "page": 7, "notes": "Check the proof."})
    detail = client.get(base).json()
    assert detail["paper_ids"] == ["P1"]
    assert detail["items"][0]["reading"]["page"] == 7
    assert client.get("/api/projects").json()["items"][0]["id"] == project["id"]
    update = client.put(base, json={"title": "Revised project", "brief": "New research question."})
    assert update.status_code == 200 and update.json()["title"] == "Revised project"
    assert update.json()["paper_ids"] == ["P1"]
    assert client.delete(base + "/papers/P1").json() == {"removed": True}
    assert client.get(base).json()["paper_ids"] == []
    assert client.delete(base).json() == {"deleted": True}
    assert client.get(base).status_code == 404
    assert client.get("/api/projects").json()["items"] == []
    assert client.get("/api/papers/P1/reading").json()["notes"] == "Check the proof."


@pytest.mark.parametrize("body", [{"title": ""}, {"title": " \n "}, {"title": "x" * 161}, {"title": "Research", "brief": "x" * 6001}])
def test_project_input_validation(client, body):
    assert client.post("/api/projects", json=body).status_code == 422


def test_project_and_notes_are_isolated_between_profiles(client, app_config):
    project = create_project(client)
    base = "/api/projects/" + project["id"]
    client.post(base + "/papers/P1")
    client.put("/api/papers/P1/reading", json={"status": "read", "page": 9, "notes": "Private note"})
    with TestClient(create_app(**app_config)) as stranger:
        assert stranger.get("/api/projects").json()["items"] == []
        for suffix in ["", "/suggestions", "/graph"]:
            assert stranger.get(base + suffix).status_code == 404
        assert stranger.put(base, json={"title": "Hijack"}).status_code == 404
        assert stranger.delete(base).status_code == 404
        assert stranger.post(base + "/papers/P2").status_code == 404
        assert stranger.delete(base + "/papers/P1").status_code == 404
        assert stranger.get("/api/papers/P1/reading").json()["notes"] == ""
        assert stranger.get("/api/reading").json()["items"] == []
        stranger.put("/api/papers/P1/reading", json={"notes": "Other profile", "page": 2})
    assert client.get(base).json()["paper_ids"] == ["P1"]
    assert client.get("/api/papers/P1/reading").json()["notes"] == "Private note"


def test_suggestions_use_brief_relevance_exclude_members_and_tolerate_prose(client):
    project = create_project(client, 'Unfinished "question', 'graph neural networks; compare "benchmark -retrieval AND methods')
    base = "/api/projects/" + project["id"]
    client.post(base + "/papers/P1")
    response = client.get(base + "/suggestions")
    assert response.status_code == 200
    result = response.json()
    assert "P1" not in [p["id"] for p in result["items"]]
    assert result["items"][0]["id"] == "P2"
    assert "graph" in result["query"]
    assert all(p["reason"].startswith("Brief terms:") for p in result["items"])
    no_terms = create_project(client, "qzxunknown", "zzyyunknown")
    assert client.get(f"/api/projects/{no_terms['id']}/suggestions").json()["items"] == []


def test_project_graph_membership_and_true_directed_citations(client, engine):
    project = create_project(client)
    base = "/api/projects/" + project["id"]
    empty = client.get(base + "/graph").json()
    assert empty["nodes"] == empty["edges"] == []
    client.post(base + "/papers/P1")
    client.post(base + "/papers/P5")
    result = client.get(base + "/graph").json()
    nodes = {p["id"]: p for p in result["nodes"]}
    assert set(nodes) == {"P1", "P2", "P3", "P5"}
    assert {pid for pid, p in nodes.items() if p["member"]} == {"P1", "P5"}
    assert result["project_papers"] == 2 and result["candidate_nodes"] == 4
    assert {(e["source"], e["target"]) for e in result["edges"]} == {("P1", "P3"), ("P2", "P1"), ("P2", "P3")}
    assert all(engine.graph.has_edge(e["source"], e["target"]) for e in result["edges"])
    assert all(math.isfinite(n[c]) for n in nodes.values() for c in ["x", "y"])
    members = client.get(base + "/graph", params={"neighbors": False}).json()
    assert {n["id"] for n in members["nodes"]} == {"P1", "P5"}
    assert members["edges"] == []
    limited = client.get(base + "/graph", params={"limit": 1}).json()
    assert len(limited["nodes"]) == 1 and limited["nodes"][0]["member"]
    assert client.get(base + "/graph?limit=201").status_code == 422


@pytest.mark.parametrize("body", [{"status": "done"}, {"page": 0}, {"page": 10001}, {"notes": "x" * 30001}])
def test_reading_validation(client, body):
    assert client.put("/api/papers/P1/reading", json=body).status_code == 422


def test_reading_notes_status_page_persist_across_app_restart(client, app_config):
    initial = client.get("/api/papers/P1/reading").json()
    assert {key: initial[key] for key in ["status", "page", "notes", "updated"]} == {"status": "queued", "page": 1, "notes": "", "updated": None}
    notes = "Ý tưởng: graph → retrieval\n<script>alert('literal text')</script>"
    response = client.put("/api/papers/P1/reading", json={"status": "reading", "page": 12, "notes": notes})
    assert response.status_code == 200 and response.json()["notes"] == notes
    profile = client.cookies["rl_profile"]
    with TestClient(create_app(**app_config)) as restored:
        restored.cookies.set("rl_profile", profile)
        state = restored.get("/api/papers/P1/reading").json()
        assert state["notes"] == notes and state["page"] == 12 and state["status"] == "reading"
        reading = restored.get("/api/reading").json()["items"]
        assert len(reading) == 1 and reading[0]["id"] == "P1"
        assert reading[0]["reading"]["notes"] == notes


def test_reader_draft_key_is_stable_and_profile_paper_scoped(client, app_config):
    first = client.get("/api/papers/P1/reading").json()["draft_key"]
    assert len(first) == 64 and all(c in "0123456789abcdef" for c in first)
    assert client.get("/api/papers/P1/reading").json()["draft_key"] == first
    assert client.get("/api/papers/P2/reading").json()["draft_key"] != first
    profile = client.cookies["rl_profile"]
    assert first != profile
    assert profile not in client.get("/api/papers/P1/reading").text
    client.put("/api/papers/P1/reading", json={"status": "read", "notes": "Edited note", "page": 4})
    assert client.get("/api/papers/P1/reading").json()["draft_key"] == first
    with TestClient(create_app(**app_config)) as other:
        assert other.get("/api/papers/P1/reading").json()["draft_key"] != first
        other.cookies.clear()
        other.cookies.set("rl_profile", profile)
        assert other.get("/api/papers/P1/reading").json()["draft_key"] == first


def test_suggestions_find_eligible_papers_beyond_first_hundred_members(app_config, papers):
    # Equal retrieval scores make the corpus IDs a deterministic ranking; all first
    # 100 results are project members, while eleven eligible matches still exist.
    corpus = [dict(papers[0], id=f"L{i:03}", title=f"Graph neural networks {i}", references=[]) for i in range(111)]
    (app_config["data_dir"] / "papers.jsonl").write_text("".join(json.dumps(p) + "\n" for p in corpus))
    (app_config["data_dir"] / "reviews.jsonl").unlink()
    with TestClient(create_app(**app_config)) as client:
        project = create_project(client, "Graph neural networks", "")
        profile = client.cookies["rl_profile"]
        for paper in corpus[:100]:
            client.app.state.store.project_paper(profile, project["id"], paper["id"], True)
        result = client.get(f"/api/projects/{project['id']}/suggestions")
        assert result.status_code == 200
        suggested = [p["id"] for p in result.json()["items"]]
        assert suggested == [p["id"] for p in corpus[100:110]]


def test_suggestions_deduplicate_title_variants_without_merging_graph_ids(app_config, papers):
    titles = ["Graph neural networks", "GRAPH: Neural Networks!", "Graph neural architectures", "GRAPH neural architectures.", "Graph neural networks survey"]
    corpus = [dict(papers[0], id=f"V{i}", title=title, references=[]) for i, title in enumerate(titles)]
    corpus[0]["references"] = [p["id"] for p in corpus[1:]]
    (app_config["data_dir"] / "papers.jsonl").write_text("".join(json.dumps(p) + "\n" for p in corpus))
    (app_config["data_dir"] / "reviews.jsonl").unlink()
    with TestClient(create_app(**app_config)) as client:
        project = create_project(client, "Graph neural networks", "")
        base = f"/api/projects/{project['id']}"
        client.post(base + "/papers/V0")
        response = client.get(base + "/suggestions")
        assert response.status_code == 200
        suggested = {p["id"] for p in response.json()["items"]}
        assert suggested.isdisjoint({"V0", "V1"})
        assert len(suggested & {"V2", "V3"}) == 1
        assert "V4" in suggested
        assert len(suggested) == 2
        graph = client.get(base + "/graph").json()
        assert {n["id"] for n in graph["nodes"]} == {p["id"] for p in corpus}
        assert {(e["source"], e["target"]) for e in graph["edges"]} == {("V0", f"V{i}") for i in range(1, 5)}


def test_workspace_missing_resources_and_cross_origin_writes(client):
    project = create_project(client)
    assert client.post(f"/api/projects/{project['id']}/papers/unknown").status_code == 404
    assert client.get("/api/papers/unknown/reading").status_code == 404
    assert client.put("/api/papers/unknown/reading", json={}).status_code == 404
    assert client.post("/api/projects", json={"title": "Injected"}, headers={"Origin": "https://attacker.example"}).status_code == 403
    assert client.put("/api/papers/P1/reading", json={}, headers={"Origin": "https://attacker.example"}).status_code == 403


def test_store_project_ownership_cascade_and_reader_persistence(tmp_path, papers):
    path = tmp_path / "workspace.sqlite"
    store = Store(path)
    store.ingest(papers)
    store.write_project("alice", "project", "Title", "Brief")
    store.project_paper("alice", "project", "P1", True)
    store.project_paper("alice", "project", "P1", True)
    assert store.project("alice", "project")["paper_ids"] == ["P1"]
    store.write_project("bob", "project", "Hijacked", "bad")
    assert store.project("alice", "project")["title"] == "Title"
    assert store.project("bob", "project") is None
    with pytest.raises(KeyError):
        store.project_paper("bob", "project", "P2", True)
    store.delete_project("bob", "project")
    assert store.project("alice", "project") is not None
    store.write_reading("alice", "P1", dict(status="read", page=8, notes="Retain independently"))
    store.delete_project("alice", "project")
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM project_papers").fetchone()[0] == 0
    restored = Store(path)
    assert restored.reading("alice", "P1")["notes"] == "Retain independently"
    assert restored.readings("bob") == []


def test_project_and_reader_filter_stale_ids_after_corpus_change(client, app_config, papers):
    project = create_project(client)
    client.post(f"/api/projects/{project['id']}/papers/P1")
    client.put("/api/papers/P1/reading", json={"notes": "Previously read"})
    profile = client.cookies["rl_profile"]
    (app_config["data_dir"] / "papers.jsonl").write_text("".join(json.dumps(p) + "\n" for p in papers if p["id"] != "P1"))
    (app_config["data_dir"] / "reviews.jsonl").unlink()
    with TestClient(create_app(**app_config)) as restored:
        restored.cookies.set("rl_profile", profile)
        assert restored.get(f"/api/projects/{project['id']}").json()["paper_ids"] == []
        assert restored.get(f"/api/projects/{project['id']}/graph").json()["nodes"] == []
        assert restored.get("/api/reading").json()["items"] == []
