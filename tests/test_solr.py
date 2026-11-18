"""Optional Solr contract tests; HTTP is mocked, actual Solr is tested separately."""
import httpx
import pytest

from researchlens.solr import SolrSearch, escape, expression


def mock_solr(monkeypatch, docs=None, total=None, status=200):
    captured = []
    docs = [{"id": "P1", "score": 3.25}] if docs is None else docs
    def get(url, **kwargs):
        captured.append(dict(url=url, **kwargs))
        return httpx.Response(status, json={"response": {"numFound": len(docs) if total is None else total, "docs": docs}}, request=httpx.Request("GET", url))
    monkeypatch.setattr("researchlens.solr.httpx.get", get)
    return captured


@pytest.mark.parametrize("token", ["+", "-", "!", "(", ")", "{", "}", "[", "]", "^", '"', "~", "*", "?", ":", "\\", "/", "&&", "||"])
def test_lucene_special_characters_are_escaped(token):
    assert escape(token) == "\\" + token


def test_query_translation_preserves_boolean_phrase_exclusions():
    assert expression("") == "*:*"
    assert expression("graph neural") == '("graph" OR "neural")'
    assert expression("graph AND neural") == '("graph" AND "neural")'
    phrase = expression('"graph neural" -benchmark')
    assert '"graph neural"' in phrase
    assert ' AND "graph neural"' in phrase
    assert phrase.endswith(' AND NOT "benchmark"')
    assert expression("-benchmark") == '*:* AND NOT "benchmark"'
    with pytest.raises(ValueError, match="quotation"):
        expression('"unterminated')


@pytest.mark.parametrize("query", ["{!lucene}*:*", "{!join from=id to=id}foo", "title:secret", "graph || *:*"])
def test_queries_cannot_inject_local_parameters_or_unrestricted_field_search(query):
    translated = expression(query)
    assert "{!" not in translated
    assert "*:*" not in translated
    assert "title:" not in translated
    assert translated.startswith('("')


def test_adapter_maps_results_scores_and_all_filters(engine, monkeypatch):
    captured = mock_solr(monkeypatch, total=12)
    result = SolrSearch("http://solr.example/solr/papers/").search(
        engine, "graph", year_min=2018, year_max=2022, topic="graphs", venue='SIGIR:2022', open_access=True, offset=4, limit=7
    )
    assert result["total"] == 12
    assert result["items"][0]["id"] == "P1" and result["items"][0]["score"] == 3.25
    request = captured[0]
    assert request["url"] == "http://solr.example/solr/papers/select"
    assert request["timeout"] == 15
    params = request["params"]
    assert params["q"] == '("graph")'
    assert params["qf"] == "p_title^2 p_abstract"
    assert params["start"] == 4 and params["rows"] == 7
    assert params["fq"] == ["p_year:[2018 TO 2022]", 'p_topic:"graphs"', 'p_venue:"SIGIR\\:2022"', "p_oa:true"]
    assert params["sort"] == "score desc,id asc"


@pytest.mark.parametrize("kwargs,expected", [({"year_min": 2020}, "p_year:[2020 TO *]"), ({"year_max": 2020}, "p_year:[* TO 2020]")])
def test_open_ended_year_filters(engine, monkeypatch, kwargs, expected):
    captured = mock_solr(monkeypatch, docs=[])
    result = SolrSearch("http://solr.example").search(engine, **kwargs)
    assert captured[0]["params"]["fq"] == [expected]
    assert result == {"total": 0, "items": []}


def test_empty_query_sorts_by_corpus_pagerank(engine, monkeypatch):
    captured = mock_solr(monkeypatch, docs=[{"id": "P3"}])
    result = SolrSearch("http://solr.example").search(engine)
    assert captured[0]["params"]["sort"] == "p_pagerank desc,id asc"
    assert "fq" not in captured[0]["params"]
    assert result["items"][0]["id"] == "P3"


def test_filter_values_escape_local_parameters(engine, monkeypatch):
    captured = mock_solr(monkeypatch)
    SolrSearch("http://solr.example").search(engine, topic='{!lucene}*:*', venue='ACL" OR *:*')
    for value in captured[0]["params"]["fq"]:
        assert "{!" not in value and "*:*" not in value
    assert '\\"' in captured[0]["params"]["fq"][1]


def test_stale_solr_identifiers_fail_explicitly(engine, monkeypatch):
    mock_solr(monkeypatch, docs=[{"id": "deleted-paper", "score": 1}])
    with pytest.raises(RuntimeError, match="stale"):
        SolrSearch("http://solr.example").search(engine, "graph")


def test_http_error_propagates_from_adapter(engine, monkeypatch):
    mock_solr(monkeypatch, status=500)
    with pytest.raises(httpx.HTTPStatusError):
        SolrSearch("http://solr.example").search(engine, "graph")


def test_api_reports_disabled_solr(client):
    assert not client.get("/api/health").json()["solr_enabled"]
    response = client.get("/api/search", params={"q": "graph", "method": "solr"})
    assert response.status_code == 503
    assert "not enabled" in response.json()["detail"]


def test_api_enabled_solr_returns_results_and_records_history(client, monkeypatch):
    monkeypatch.setenv("SOLR_URL", "http://solr.example")
    captured = mock_solr(monkeypatch)
    assert client.get("/api/health").json()["solr_enabled"]
    response = client.get("/api/search", params={"q": "graph", "method": "solr", "offset": 2})
    assert response.status_code == 200
    result = response.json()
    assert result["method"] == "solr" and result["offset"] == 2 and result["elapsed_ms"] >= 0
    assert result["items"][0]["id"] == "P1"
    assert captured[0]["params"]["start"] == 2
    assert client.get("/api/library").json()["history"] == ["graph"]


@pytest.mark.parametrize("failure", ["http", "connection", "stale"])
def test_api_unavailable_solr_returns_503(client, monkeypatch, failure):
    monkeypatch.setenv("SOLR_URL", "http://solr.example")
    if failure == "connection":
        def unavailable(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")
        monkeypatch.setattr("researchlens.solr.httpx.get", unavailable)
    elif failure == "http":
        mock_solr(monkeypatch, status=500)
    else:
        mock_solr(monkeypatch, docs=[{"id": "stale"}])
    response = client.get("/api/search", params={"q": "graph", "method": "solr"})
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]
    assert client.get("/api/library").json()["history"] == []


def test_api_unclosed_query_is_422_before_solr_request(client, monkeypatch):
    monkeypatch.setenv("SOLR_URL", "http://solr.example")
    calls = mock_solr(monkeypatch)
    response = client.get("/api/search", params={"q": '"unfinished', "method": "solr"})
    assert response.status_code == 422 and calls == []


def test_api_malformed_solr_response_is_service_failure(client, monkeypatch):
    monkeypatch.setenv("SOLR_URL", "http://solr.example")
    def invalid_json(url, **kwargs):
        return httpx.Response(200, text="Temporary upstream failure", request=httpx.Request("GET", url))
    monkeypatch.setattr("researchlens.solr.httpx.get", invalid_json)
    response = client.get("/api/search", params={"q": "graph", "method": "solr"})
    assert response.status_code == 503
