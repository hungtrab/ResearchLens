"""Offline conference catalog, public-source adapter and explicit import contracts."""
import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import httpx
import pytest
from fastapi.testclient import TestClient

from researchlens.api import create_app
from researchlens.catalog import Catalog, CONFERENCES, title_key
from researchlens.discovery import same_paper
from researchlens.sources import OpenAlex, SourceUnavailable, abstract, normalize_work
from scripts import crawl_conferences as crawl


@pytest.fixture(autouse=True)
def no_live_discovery_network(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected live network access in discovery tests")
    monkeypatch.setattr("researchlens.sources.httpx.get", forbidden)


@pytest.fixture
def work():
    return {
        "id": "https://openalex.org/W12345", "title": "Discovering graph structure in research papers",
        "doi": "https://doi.org/10.1234/discovery", "publication_year": 2024,
        "authorships": [{"author": {"display_name": "An Nguyễn"}}, {"author": {"display_name": "Binh Tran"}}],
        "primary_location": {"source": {"id": "https://openalex.org/S4306419644", "display_name": "ICML"}, "pdf_url": "https://proceedings.mlr.press/old.pdf"},
        "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/2401.12345"},
        "abstract_inverted_index": {"Graph": [0], "discovery": [1], "methods": [2]},
        "referenced_works": ["https://openalex.org/W999", "https://attacker.example/W99", "bad-id"],
        "cited_by_count": 42, "topics": [{"display_name": "Graph mining"}], "open_access": {"is_oa": True},
    }


@pytest.fixture
def catalog_rows(papers):
    return [
        dict(papers[0], id="C1", conference="ICML", year=2024, source="Official proceedings", title="Graph discovery systems", doi=None),
        dict(papers[1], id="C2", conference="ACL", year=2023, source="Official proceedings", title="Language retrieval models", abstract="Graph mining for documents", doi=None),
        dict(papers[4], id="C3", conference="ACL", year=2025, source="Official proceedings", title="Review sentiment extraction", doi=None),
    ]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf8")


def populate_catalog(client, app_config, rows):
    path = app_config["data_dir"] / "conferences" / "papers.jsonl"
    write_jsonl(path, rows)
    client.app.state.catalog.sync(path)


def respond(monkeypatch, payload, status=200):
    calls = []
    def get(url, **kwargs):
        calls.append(dict(url=url, **kwargs))
        return httpx.Response(status, json=payload, request=httpx.Request("GET", url))
    monkeypatch.setattr("researchlens.sources.httpx.get", get)
    return calls


def test_catalog_sync_fts_ranking_filters_and_snapshot_replacement(tmp_path, catalog_rows):
    path = tmp_path / "papers.jsonl"
    write_jsonl(path, catalog_rows)
    catalog = Catalog(tmp_path / "catalog.sqlite")
    catalog.sync(path)
    assert catalog.get("C1")["title"] == "Graph discovery systems"
    assert catalog.get("absent") is None
    assert [p["id"] for p in catalog.search("graph")["items"]] == ["C1", "C2"]
    assert [p["id"] for p in catalog.search("", conference="ACL", year_min=2024, year_max=2026)["items"]] == ["C3"]
    result = catalog.search("", offset=1, limit=1)
    assert result["total"] == 3 and result["items"][0]["id"] == "C1"
    assert catalog.search("zzunknown")["items"] == []
    assert catalog.stats()["papers"] == 3
    catalog.sync(path)
    assert catalog.stats()["papers"] == 3
    write_jsonl(path, [catalog_rows[2]])
    catalog.sync(path)
    assert catalog.get("C1") is None and catalog.search("graph")["total"] == 0
    assert Catalog(tmp_path / "catalog.sqlite").stats()["papers"] == 1


@pytest.mark.parametrize("query", ['"unfinished', "graph NEAR(", "graph OR *", "'; DROP TABLE catalog; --", "title:graph", "!!!"])
def test_fts_treats_query_syntax_as_terms_without_sql_or_parser_injection(tmp_path, catalog_rows, query):
    path = tmp_path / "papers.jsonl"
    write_jsonl(path, catalog_rows)
    catalog = Catalog(tmp_path / "catalog.sqlite")
    catalog.sync(path)
    result = catalog.search(query)
    assert isinstance(result["items"], list)
    assert catalog.stats()["papers"] == 3 and catalog.get("C1")
    if query == "!!!":
        assert result["total"] == 0


def test_catalog_bad_snapshot_rolls_back_and_missing_source_preserves_last_success(tmp_path, catalog_rows):
    path = tmp_path / "papers.jsonl"
    catalog = Catalog(tmp_path / "catalog.sqlite")
    catalog.sync(path)
    assert catalog.stats()["papers"] == 0
    write_jsonl(path, catalog_rows)
    catalog.sync(path)
    path.write_text("not JSON\n")
    with pytest.raises(ValueError):
        catalog.sync(path)
    assert catalog.stats()["papers"] == 3 and catalog.search("graph")["total"] == 2
    path.unlink()
    catalog.sync(path)
    assert catalog.stats()["papers"] == 3


def test_catalog_batched_import_rolls_back_all_rows_on_duplicate_id(tmp_path, catalog_rows):
    path = tmp_path / "papers.jsonl"
    catalog = Catalog(tmp_path / "catalog.sqlite")
    write_jsonl(path, catalog_rows)
    catalog.sync(path)
    large = [dict(catalog_rows[0], id=f"B{i}") for i in range(1005)]
    large.append(dict(large[0]))
    write_jsonl(path, large)
    with pytest.raises(sqlite3.IntegrityError):
        catalog.sync(path)
    assert catalog.stats()["papers"] == 3
    assert catalog.get("B0") is None and catalog.search("graph")["total"] == 2


def test_openalex_normalization_preserves_metadata_and_uses_oa_pdf(work):
    paper = normalize_work(work)
    assert paper["id"] == "W12345" and paper["conference"] == "ICML"
    assert paper["abstract"] == "Graph discovery methods"
    assert paper["authors"] == ["An Nguyễn", "Binh Tran"]
    assert paper["references"] == ["W999"]
    assert paper["pdf_url"] == "https://arxiv.org/pdf/2401.12345"
    assert paper["citations"] == 42 and paper["open_access"] is True
    assert paper["metadata_url"] == "https://api.openalex.org/works/W12345"
    assert abstract(None) == ""
    assert abstract({"first": [0, -1, 100000, "1"], "last": [2]}) == "first last"
    assert title_key("Việt Nam: Graph!") == title_key("VIỆT NAM graph")


@pytest.mark.parametrize("change", [{"id": "https://attacker.example/not-work"}, {"id": ""}, {"title": ""}])
def test_invalid_openalex_metadata_is_rejected(work, change):
    work.update(change)
    with pytest.raises(ValueError):
        normalize_work(work)


def test_openalex_missing_optional_metadata_has_defaults(work):
    result = normalize_work({"id": work["id"], "title": work["title"]})
    assert result["abstract"] == "" and result["authors"] == result["references"] == []
    assert result["venue"] == "Unknown venue" and result["conference"] is None
    assert not result["open_access"] and result["citations"] == 0


def test_openalex_search_parameters_normalization_and_cache_copy(monkeypatch, work):
    calls = respond(monkeypatch, {"results": [work], "meta": {"count": 25}})
    source = OpenAlex()
    first = source.search("graph", year_min=2020, year_max=2026, offset=20)
    first["items"][0]["title"] = "Mutated by caller"
    second = source.search("graph", year_min=2020, year_max=2026, offset=20)
    assert len(calls) == 1 and second["items"][0]["title"] == work["title"]
    params = calls[0]["params"]
    assert params["search"] == "graph" and params["page"] == 2 and params["per-page"] == 20
    assert params["filter"] == "from_publication_date:2020-01-01,to_publication_date:2026-12-31"
    assert calls[0]["follow_redirects"] is False and calls[0]["timeout"] == 25
    assert second["total"] == 25 and second["cached_seconds"] == 300


def test_cache_expiry_capacity_and_api_key_only_in_headers(monkeypatch):
    calls = respond(monkeypatch, {"value": {"unchanged": True}})
    monkeypatch.setenv("OPENALEX_API_KEY", "test-secret-key")
    clock = [0]
    monkeypatch.setattr("researchlens.sources.time.monotonic", lambda: clock[0])
    source = OpenAlex()
    source.get("/works/W1")
    clock[0] = 299
    source.get("/works/W1")
    assert len(calls) == 1
    clock[0] = 301
    source.get("/works/W1")
    assert len(calls) == 2
    assert calls[0]["headers"]["Authorization"] == "Bearer test-secret-key"
    assert "test-secret-key" not in calls[0]["url"] and "api_key" not in calls[0]["params"]
    for i in range(130):
        source.get(f"/works/W{i}")
    assert len(source.cache) <= 128


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_openalex_http_errors_are_actionable_source_failures(monkeypatch, status):
    respond(monkeypatch, {"error": "upstream"}, status)
    source = OpenAlex()
    with pytest.raises(SourceUnavailable):
        source.search("graph")
    assert source.cache == {}


def test_openalex_invalid_json_network_and_not_found(monkeypatch):
    def invalid(url, **kwargs):
        return httpx.Response(200, text="<html>unavailable</html>", request=httpx.Request("GET", url))
    monkeypatch.setattr("researchlens.sources.httpx.get", invalid)
    with pytest.raises(SourceUnavailable):
        OpenAlex().search("graph")
    def unavailable(*args, **kwargs):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr("researchlens.sources.httpx.get", unavailable)
    with pytest.raises(SourceUnavailable):
        OpenAlex().search("graph")
    respond(monkeypatch, {}, 404)
    with pytest.raises(KeyError):
        OpenAlex().paper("W123")
    with pytest.raises(ValueError):
        OpenAlex().paper("../../etc/passwd")


@pytest.mark.parametrize("payload", [[], {}, {"results": None, "meta": {"count": 1}}, {"results": [None], "meta": {"count": 1}}])
def test_malformed_online_search_payload_returns_service_error(client, monkeypatch, payload):
    respond(monkeypatch, payload)
    response = client.get("/api/online/search", params={"q": "graph"})
    assert response.status_code == 503
    assert client.get("/api/search?q=graph").status_code == 200


@pytest.mark.parametrize("change", [{"authorships": [None]}, {"referenced_works": None}, {"abstract_inverted_index": {"graph": None}}])
def test_malformed_online_paper_import_does_not_mutate_workspace(client, monkeypatch, work, change):
    work.update(change)
    respond(monkeypatch, work)
    response = client.post("/api/discovery/import", json={"id": "W12345", "source": "openalex"})
    assert response.status_code == 503
    assert len(client.app.state.engine.ids) == 6 and client.app.state.store.imported() == []


def test_remote_year_type_cannot_poison_workspace_filters_or_statistics(client, monkeypatch, work):
    work["publication_year"] = "2025"
    respond(monkeypatch, work)
    response = client.post("/api/discovery/import", json={"id": "W12345", "source": "openalex"})
    # Either reject malformed metadata at the source boundary or normalize it;
    # never persist a type that makes otherwise-valid local endpoints fail.
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        assert client.app.state.engine.by_id["W12345"]["year"] == 2025
    else:
        assert "W12345" not in client.app.state.engine.pos
    assert client.get("/api/stats").status_code == 200
    assert client.get("/api/facets").status_code == 200
    assert client.get("/api/search?year_min=2024").status_code == 200


def test_same_paper_doi_and_conservative_title_identity():
    assert same_paper(dict(title="A", doi="https://doi.org/10.1234/ABC"), dict(title="Changed", doi="10.1234/abc"))
    first = dict(title="Graph structure discovery in research papers", year=2024)
    assert same_paper(first, dict(title="GRAPH Structure Discovery in Research Papers!", year=2024))
    assert not same_paper(first, dict(first, year=2025))
    assert not same_paper(dict(title="Graph", year=2024), dict(title="Graph", year=2024))


def test_catalog_api_status_filters_and_benchmark_separation(client, app_config, catalog_rows):
    populate_catalog(client, app_config, catalog_rows)
    manifest = {"years": [2020, 2026], "collected_at": "2026-10-14", "sources": [{"conference": "ICML", "status": "fetched", "urls": ["https://source.example"]}]}
    (app_config["data_dir"] / "conferences/manifest.json").write_text(json.dumps(manifest))
    response = client.get("/api/catalog/search", params={"q": "graph", "conference": "ICML", "year_min": 2024})
    assert response.status_code == 200
    result = response.json()
    assert result["total"] == 1 and result["items"][0]["workspace_id"] is None
    assert result["elapsed_ms"] >= 0
    assert client.get("/api/catalog/papers/C1").json()["id"] == "C1"
    assert client.get("/api/catalog/papers/absent").status_code == 404
    status = client.get("/api/discovery/status").json()
    assert status["papers"] == 3 and status["workspace_papers"] == status["benchmark_papers"] == 6
    assert status["conferences"] == list(CONFERENCES)
    assert "urls" not in status["sources"][0]
    assert "C1" not in client.app.state.engine.pos


@pytest.mark.parametrize("path,params", [
    ("/api/catalog/search", {"conference": "Unknown"}),
    ("/api/catalog/search", {"year_min": 2025, "year_max": 2020}),
    ("/api/catalog/search", {"offset": -1}),
    ("/api/catalog/search", {"limit": 101}),
    ("/api/online/search", {"q": " "}),
    ("/api/online/search", {"q": "x"}),
    ("/api/online/search", {"q": "graph", "offset": 1}),
    ("/api/online/search", {"q": "graph", "offset": 10000}),
    ("/api/online/search", {"q": "graph", "year_min": 2026, "year_max": 2020}),
])
def test_discovery_api_validation_without_network(client, path, params):
    assert client.get(path, params=params).status_code == 422


def test_online_api_live_result_is_not_imported_and_errors_do_not_break_local_search(client, monkeypatch, work):
    calls = respond(monkeypatch, {"results": [work], "meta": {"count": 1}})
    result = client.get("/api/online/search", params={"q": " graph "})
    assert result.status_code == 200 and result.json()["items"][0]["workspace_id"] is None
    assert calls[0]["params"]["search"] == "graph"
    assert "W12345" not in client.app.state.engine.pos
    respond(monkeypatch, {}, 429)
    assert client.get("/api/online/search", params={"q": "different"}).status_code == 503
    assert client.get("/api/search?q=graph").status_code == 200


def test_catalog_import_persists_and_can_be_used_in_projects_without_exposing_private_state(client, app_config, catalog_rows):
    populate_catalog(client, app_config, catalog_rows)
    response = client.post("/api/discovery/import", json={"id": "C1", "source": "catalog", "enrich": False})
    assert response.status_code == 200 and response.json()["added"] is True
    assert response.json()["workspace_papers"] == 7
    assert client.get("/api/papers/C1").status_code == 200
    assert client.post("/api/library/C1").status_code == 200
    client.put("/api/papers/C1/reading", json={"notes": "Private imported-paper note"})
    project = client.post("/api/projects", json={"title": "Discovery project"}).json()
    assert client.post(f"/api/projects/{project['id']}/papers/C1").status_code == 200
    assert client.get("/api/catalog/papers/C1").json()["workspace_id"] == "C1"
    assert not client.post("/api/discovery/import", json={"id": "C1", "source": "catalog", "enrich": False}).json()["added"]
    profile = client.cookies["rl_profile"]
    with TestClient(create_app(**app_config)) as other:
        assert other.get("/api/papers/C1").status_code == 200
        assert other.get("/api/library").json()["ids"] == []
        assert other.get("/api/projects").json()["items"] == []
        assert other.get("/api/papers/C1/reading").json()["notes"] == ""
        status = other.get("/api/discovery/status").json()
        assert status["benchmark_papers"] == 6 and status["workspace_papers"] == 7
        other.cookies.clear()
        other.cookies.set("rl_profile", profile)
        assert other.get("/api/library").json()["ids"] == ["C1"]
        assert other.get("/api/papers/C1/reading").json()["notes"] == "Private imported-paper note"


def test_import_deduplicates_existing_doi_and_remembers_external_alias(client, app_config, catalog_rows):
    row = dict(catalog_rows[0], doi="https://doi.org/10.1234/P1")
    populate_catalog(client, app_config, [row])
    response = client.post("/api/discovery/import", json={"id": "C1", "source": "catalog", "enrich": False})
    assert response.json()["id"] == "P1" and response.json()["added"] is False
    assert len(client.app.state.engine.ids) == 6
    assert client.app.state.store.imported_alias("C1") == "P1"
    assert client.get("/api/catalog/papers/C1").json()["workspace_id"] == "P1"


def test_online_import_normalized_metadata_and_reference_edges(client, monkeypatch, work):
    # The imported reference resolves once the target exists in the workspace.
    target = dict(work, id="https://openalex.org/W999", title="Earlier graph mining foundations", doi="https://doi.org/10.1234/earlier", referenced_works=[])
    monkeypatch.setattr(client.app.state.online, "paper", lambda pid: normalize_work(target if pid == "W999" else work))
    assert client.post("/api/discovery/import", json={"id": "W999", "source": "openalex"}).json()["added"]
    assert client.post("/api/discovery/import", json={"id": "W12345", "source": "openalex"}).json()["added"]
    assert client.app.state.engine.graph.has_edge("W12345", "W999")
    assert client.get("/api/papers/W12345").json()["abstract"] == "Graph discovery methods"


def test_catalog_enrichment_preserves_official_fields_and_tracks_both_aliases(client, app_config, catalog_rows, work, monkeypatch):
    official = dict(catalog_rows[0], title=work["title"], year=2024, url="https://proceedings.mlr.press/official.html", pdf_url="https://proceedings.mlr.press/official.pdf")
    populate_catalog(client, app_config, [official])
    monkeypatch.setattr(client.app.state.online, "search", lambda *args: {"items": [normalize_work(work)]})
    response = client.post("/api/discovery/import", json={"id": "C1", "source": "catalog"})
    assert response.json()["id"] == "W12345" and response.json()["warning"] is None
    imported = client.app.state.engine.by_id["W12345"]
    assert imported["official_url"] == official["url"] and imported["pdf_url"] == official["pdf_url"]
    assert imported["abstract"] == "Graph discovery methods"
    assert client.app.state.store.imported_alias("C1") == "W12345"
    assert client.app.state.store.imported_alias("W12345") == "W12345"


@pytest.mark.parametrize("failure", ["no_match", "unavailable"])
def test_enrichment_failure_keeps_official_metadata_with_warning(client, app_config, catalog_rows, monkeypatch, failure):
    populate_catalog(client, app_config, catalog_rows)
    def search(*args, **kwargs):
        if failure == "unavailable":
            raise SourceUnavailable("offline")
        return {"items": []}
    monkeypatch.setattr(client.app.state.online, "search", search)
    response = client.post("/api/discovery/import", json={"id": "C1", "source": "catalog"})
    assert response.status_code == 200 and response.json()["id"] == "C1"
    assert response.json()["added"] and response.json()["warning"]


@pytest.mark.parametrize("body,status", [
    ({"id": "absent", "source": "catalog", "enrich": False}, 404),
    ({"id": "../escape", "source": "catalog"}, 422),
    ({"id": "P1", "source": "invalid"}, 422),
    ({"id": "NOT_OPENALEX", "source": "openalex"}, 422),
])
def test_import_validation(client, body, status):
    assert client.post("/api/discovery/import", json=body).status_code == status


def test_online_import_source_failures_and_cross_origin_writes(client, monkeypatch):
    respond(monkeypatch, {}, 404)
    assert client.post("/api/discovery/import", json={"id": "W404", "source": "openalex"}).status_code == 404
    respond(monkeypatch, {}, 503)
    assert client.post("/api/discovery/import", json={"id": "W503", "source": "openalex"}).status_code == 503
    response = client.post("/api/discovery/import", json={"id": "W1", "source": "openalex"}, headers={"Origin": "https://attacker.example"})
    assert response.status_code == 403


def test_failed_import_persistence_keeps_existing_engine(client, app_config, catalog_rows, monkeypatch):
    populate_catalog(client, app_config, catalog_rows)
    before = client.app.state.engine
    def failure(*args, **kwargs):
        raise sqlite3.OperationalError("simulated disk failure")
    monkeypatch.setattr(client.app.state.store, "persist_import", failure)
    with pytest.raises(sqlite3.OperationalError):
        client.post("/api/discovery/import", json={"id": "C1", "source": "catalog", "enrich": False})
    assert client.app.state.engine is before
    assert "C1" not in before.pos and client.app.state.store.imported() == []


def test_concurrent_import_of_same_paper_commits_once(client, monkeypatch, work):
    barrier = Barrier(2)
    def fetch(pid):
        barrier.wait(timeout=5)
        return normalize_work(copy.deepcopy(work))
    monkeypatch.setattr(client.app.state.online, "paper", fetch)
    def submit():
        return client.post("/api/discovery/import", json={"id": "W12345", "source": "openalex"})
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: submit(), range(2)))
    assert all(r.status_code == 200 for r in responses)
    assert sorted(r.json()["added"] for r in responses) == [False, True]
    assert len(client.app.state.engine.ids) == 7
    assert len(client.app.state.store.imported()) == 1


def test_concurrent_distinct_imports_do_not_lose_updates(client, monkeypatch, work):
    barrier = Barrier(2)
    def fetch(pid):
        barrier.wait(timeout=5)
        metadata = dict(work, id="https://openalex.org/" + pid, title="Research discovery " + pid,
                        doi="https://doi.org/10.1234/" + pid)
        return normalize_work(metadata)
    monkeypatch.setattr(client.app.state.online, "paper", fetch)
    def submit(pid):
        return client.post("/api/discovery/import", json={"id": pid, "source": "openalex"})
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(submit, ["W101", "W102"]))
    assert all(r.status_code == 200 and r.json()["added"] for r in responses)
    assert {"W101", "W102"}.issubset(client.app.state.engine.pos)
    assert len(client.app.state.engine.ids) == 8
    assert {p["id"] for p in client.app.state.store.imported()} == {"W101", "W102"}


@pytest.mark.parametrize("order", [("W101", "W102"), ("W102", "W101")])
def test_import_aliases_resolve_citations_before_and_after_restart(client, app_config, catalog_rows, work, monkeypatch, order):
    official = dict(catalog_rows[0], title="Graph structure discovery in research papers", doi=None)
    populate_catalog(client, app_config, [official])
    assert client.post("/api/discovery/import", json={"id": "C1", "source": "catalog", "enrich": False}).json()["added"]
    alias = dict(work, id="https://openalex.org/W101", title=official["title"], publication_year=official["year"])
    citing = dict(work, id="https://openalex.org/W102", title="New citation discovery algorithms", doi="https://doi.org/10.1234/new", referenced_works=["https://openalex.org/W101"])
    monkeypatch.setattr(client.app.state.online, "paper", lambda pid: normalize_work(alias if pid == "W101" else citing))
    for pid in order:
        assert client.post("/api/discovery/import", json={"id": pid, "source": "openalex"}).status_code == 200
    assert client.app.state.store.imported_alias("W101") == "C1"
    assert "W101" not in client.app.state.engine.pos
    assert client.app.state.engine.graph.has_edge("W102", "C1")
    assert [p["id"] for p in client.get("/api/papers/W102").json()["references"]] == ["C1"]
    with TestClient(create_app(**app_config)) as restored:
        assert restored.app.state.engine.graph.has_edge("W102", "C1")


def test_crawler_virtual_program_filters_workshops_and_derives_pdf(monkeypatch):
    rows = [
        {"name": "Graph discovery", "eventtype": "Poster", "paper_url": "https://openreview.net/forum?id=abc", "authors": [{"fullname": "An Nguyen"}], "abstract": "<p>Graph methods</p>", "sourceurl": "main"},
        {"name": "Workshop paper", "eventtype": "Poster", "sourceurl": "workshop/program"},
        {"name": "Keynote", "eventtype": "Keynote"},
    ]
    def fetch(url):
        return json.dumps({"results": rows}) if url.endswith(".json") else '<script src="/data/orals-posters.json"></script>'
    monkeypatch.setattr(crawl, "fetch", fetch)
    result, urls = crawl.virtual("ICLR", 2024)
    assert len(result) == 1 and result[0]["pdf_url"] == "https://openreview.net/pdf?id=abc"
    assert result[0]["authors"] == ["An Nguyen"] and "<p>" not in result[0]["abstract"]
    assert urls == ["https://iclr.cc/data/orals-posters.json"]


def test_old_virtual_schema_keeps_distinct_event_ids_when_paper_urls_are_missing(monkeypatch):
    first = {"id": 1506, "uid": "8fecb20817b3847419bb3de39a609afe", "name": "Generalization bounds for deep convolutional neural networks", "eventtype": "Poster", "paper_pdf_url": None, "eventmedia": []}
    second = dict(first, id=1507, uid="other-stable-event-id", name="Graph discovery in neural networks")
    rows = [first, second, copy.deepcopy(first)]
    monkeypatch.setattr(crawl, "fetch", lambda url: json.dumps(rows) if url.endswith(".json") else "<html>Program</html>")
    parsed, _ = crawl.virtual("ICLR", 2020)
    result = crawl.merge_records([parsed])
    assert len(result) == 2
    assert len({p["id"] for p in result}) == 2
    assert {p["title"] for p in result} == {first["name"], second["name"]}
    # A program-index URL is a valid provenance fallback when no paper link was
    # published; it must not become the identity that collapses distinct events.
    assert all(p["pdf_url"] is None for p in result)


def test_old_virtual_schema_extracts_openreview_from_eventmedia(monkeypatch):
    row = {"id": 7199, "uid": "stable-old-event", "name": "MoReL: Multi-omics Relational Learning", "eventtype": "Poster", "eventmedia": [{"type": "URL", "name": "OpenReview", "uri": "https://openreview.net/forum?id=DnG75_KyHjX"}]}
    monkeypatch.setattr(crawl, "fetch", lambda url: json.dumps([row]) if url.endswith(".json") else "<html>Program</html>")
    result, _ = crawl.virtual("ICLR", 2022)
    assert result[0]["url"] == "https://openreview.net/forum?id=DnG75_KyHjX"
    assert result[0]["pdf_url"] == "https://openreview.net/pdf?id=DnG75_KyHjX"


def test_virtual_program_excludes_parent_spotlight_sessions(monkeypatch):
    rows = [
        {"id": 65847, "name": "Lightning Talks4B-1", "eventtype": "Spotlight", "virtualsite_url": "/virtual/2022/session/64760", "paper_url": "", "children_ids": [11, 12]},
        {"id": 11, "name": "Graph discovery methods", "eventtype": "Poster", "paper_url": "https://openreview.net/forum?id=paper"},
    ]
    monkeypatch.setattr(crawl, "fetch", lambda url: json.dumps(rows) if url.endswith(".json") else "<html>Program</html>")
    result, _ = crawl.virtual("NeurIPS", 2022)
    assert [p["title"] for p in result] == ["Graph discovery methods"]


def test_crawler_anthology_parses_markup_authors_and_skips_frontmatter(monkeypatch):
    tree = {"sha": "pinnedcommit", "tree": [{"path": "data/xml/2024.acl.xml"}, {"path": "data/xml/2024.emnlp.xml"}]}
    xml = '<collection id="2024.acl"><volume id="long"><paper id="0"><title>Preface</title></paper><paper id="1"><title>Graph <i>Discovery</i></title><author><first>An</first><last>Nguyen</last></author><abstract>Research <b>evidence</b>.</abstract><doi>10.1234/acl</doi></paper></volume></collection>'
    seen = []
    def fetch(url):
        seen.append(url)
        return xml
    monkeypatch.setattr(crawl, "fetch", fetch)
    result, urls = crawl.anthology("ACL", 2024, tree)
    assert len(result) == 1 and result[0]["title"] == "Graph Discovery"
    assert result[0]["abstract"] == "Research evidence." and result[0]["authors"] == ["An Nguyen"]
    assert result[0]["doi"] == "https://doi.org/10.1234/acl"
    assert result[0]["pdf_url"] == "https://aclanthology.org/2024.acl-long.1.pdf"
    assert urls == seen and "/pinnedcommit/" in urls[0]


def test_findings_collection_assigns_each_volume_only_to_its_conference(monkeypatch):
    tree = {"sha": "pinnedcommit", "tree": [{"path": "data/xml/2025.findings.xml"}]}
    xml = '<collection id="2025.findings">' + ''.join(
        f'<volume id="{conference}"><paper id="1"><title>{conference.upper()} findings paper</title></paper></volume>'
        for conference in ["acl", "emnlp", "naacl"]
    ) + '</collection>'
    monkeypatch.setattr(crawl, "fetch", lambda url: xml)
    for conference in ["ACL", "EMNLP", "NAACL"]:
        result, urls = crawl.anthology(conference, 2025, tree)
        assert len(result) == 1
        assert result[0]["conference"] == conference
        assert result[0]["title"] == conference + " findings paper"
        assert result[0]["id"] == f"ACL-2025.findings-{conference.lower()}.1"
        assert len(urls) == 1 and urls[0].endswith("/2025.findings.xml")


def test_crawler_pmlr_parses_official_volume(monkeypatch):
    home = '<a href="/v235/">ICML 2024</a>'
    volume = '<title>International Conference on Machine Learning</title><div class="paper"><p class="title">Graph &amp; Learning</p><span class="authors">An Nguyen, Binh Tran</span><a href="https://proceedings.mlr.press/v235/graph24.html">abs</a><a href="https://proceedings.mlr.press/v235/graph24.pdf">Download PDF</a></div>'
    monkeypatch.setattr(crawl, "fetch", lambda url: home if url == "https://proceedings.mlr.press/" else volume)
    result, urls = crawl.pmlr(2024)
    assert result[0]["title"] == "Graph & Learning" and result[0]["id"] == "PMLR-v235-graph24"
    assert result[0]["authors"] == ["An Nguyen", "Binh Tran"]
    assert urls == ["https://proceedings.mlr.press/v235/"]


def test_crawler_cvf_resolves_relative_links(monkeypatch):
    html = '<dt class="ptitle"><a href="/content/CVPR2024/html/graph.html">Graph Vision</a></dt><dd><input name="query_author" value="An Nguyen"><a href="/content/CVPR2024/papers/graph_paper.pdf">pdf</a></dd>'
    monkeypatch.setattr(crawl, "fetch", lambda url: html)
    result, _ = crawl.cvf(2024)
    assert result[0]["authors"] == ["An Nguyen"]
    assert result[0]["pdf_url"] == "https://openaccess.thecvf.com/content/CVPR2024/papers/graph_paper.pdf"


def test_crawler_old_cvf_index_falls_back_from_invalid_day_parameter(monkeypatch):
    seen = []
    def fetch(url):
        seen.append(url)
        if "?day=all" in url:
            return "<html>Incorrect DATE value: all</html>"
        return '<dt class="ptitle"><a href="/content_CVPR_2020/html/graph.html">Graph Vision</a></dt><dd><a href="/content_CVPR_2020/papers/graph_paper.pdf">pdf</a></dd>'
    monkeypatch.setattr(crawl, "fetch", fetch)
    result, _ = crawl.cvf(2020)
    assert len(result) == 1 and result[0]["title"] == "Graph Vision"
    assert "https://openaccess.thecvf.com/CVPR2020" in seen


def test_crawler_cvf_2020_day_navigation_fetches_unique_days_and_deduplicates_papers(monkeypatch):
    def entry(name):
        return f'<dt class="ptitle"><a href="/content_CVPR_2020/html/{name}.html">Vision {name}</a></dt><dd><a href="/content_CVPR_2020/papers/{name}_paper.pdf">pdf</a></dd>'
    nav = ''.join(f'<a href="CVPR2020.py?day=2020-06-{day}">Day {day}</a>' for day in ["16", "17", "16", "18"])
    nav += '<a href="CVPR2020.py?day=2019-06-16">Other year</a><a href="https://attacker.example/day">Unrelated</a>'
    base = "https://openaccess.thecvf.com/"
    days = {base + "CVPR2020.py?day=2020-06-16": entry("A"),
            base + "CVPR2020.py?day=2020-06-17": entry("B"),
            base + "CVPR2020.py?day=2020-06-18": entry("A") + entry("C")}
    calls = []
    def fetch(url):
        calls.append(url)
        if url == base + "CVPR2020?day=all":
            return "<html>Incorrect DATE value: all</html>"
        if url == base + "CVPR2020":
            return nav
        return days[url]
    monkeypatch.setattr(crawl, "fetch", fetch)
    result, urls = crawl.cvf(2020)
    assert {p["title"] for p in result} == {"Vision A", "Vision B", "Vision C"}
    assert len(result) == len({p["id"] for p in result}) == 3
    assert set(urls) == {base + "CVPR2020", *days}
    assert all(calls.count(day) == 1 for day in days)
    assert all(p["pdf_url"].startswith(base + "content_CVPR_2020/papers/") for p in result)


def test_crawler_merge_keeps_openalex_identity_and_enriches_official_metadata(work):
    online = normalize_work(work)
    online["abstract"] = ""
    official = dict(online, id="OFFICIAL1", title=online["title"].upper() + "!", source="Official proceedings", abstract="Official abstract", url="https://official.example/paper", metadata_url="https://official.example/index")
    next_year = dict(official, id="OFFICIAL2", year=2025)
    result = crawl.merge_records([[official, copy.deepcopy(official), next_year], [online]])
    assert len(result) == 2
    merged = next(p for p in result if p["id"] == "W12345")
    assert merged["alternate_ids"] == ["OFFICIAL1"]
    assert merged["abstract"] == "Official abstract" and merged["official_url"] == official["url"]
    assert merged["references"] == ["W999"]


def test_crawler_openalex_exhausts_cursor_and_rejects_repeated_cursor(monkeypatch, work):
    pages = [
        {"results": [work], "meta": {"count": 1, "next_cursor": "next"}},
        {"results": [], "meta": {"count": 1, "next_cursor": None}},
    ]
    calls = []
    def fetch(url):
        calls.append(url)
        return json.dumps(pages[len(calls) - 1])
    monkeypatch.setattr(crawl, "fetch", fetch)
    monkeypatch.setattr(crawl.time, "sleep", lambda _: None)
    rows, urls = crawl.openalex("ICML", 2020, 2026)
    assert len(rows) == 1 and len(urls) == 2 and "cursor=next" in urls[1]
    assert "publication_year%3A2020-2026" in urls[0]
    calls.clear()
    pages[1] = dict(pages[0])
    with pytest.raises(ValueError, match="repeated its cursor"):
        crawl.openalex("ICML", 2020, 2026)
