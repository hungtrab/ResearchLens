import math

import networkx as nx
import numpy as np
import pytest

from researchlens.mining import MiningEngine, extract, graph_metrics, opinion, query_parts, tokens


def ids(result):
    return [p["id"] for p in result["items"]]


@pytest.mark.parametrize("method", ["bm25", "tfidf", "hybrid", "pagerank"])
def test_search_enforces_relevance_for_every_ranking(engine, method):
    result = engine.search("sentiment", method=method)
    assert ids(result) == ["P5"]
    assert all(math.isfinite(p["score"]) for p in result["items"])


@pytest.mark.parametrize("method", ["bm25", "tfidf"])
def test_exact_topic_outranks_partial_matches(engine, method):
    result = engine.search("graph neural networks", method=method)
    assert result["items"][0]["id"] == "P1"
    assert result["items"][0]["score"] > next(p["score"] for p in result["items"] if p["id"] == "P3")


def test_phrase_and_boolean_and_exclusion(engine):
    assert set(ids(engine.search('"graph neural networks"'))) == {"P1", "P2"}
    assert ids(engine.search("graph AND retrieval")) == ["P4"]
    assert set(ids(engine.search("graph neural"))) == {"P1", "P2", "P3", "P4"}
    assert ids(engine.search('"graph neural networks" -benchmark')) == ["P1"]
    assert engine.search("qzxunknown") == {"total": 0, "items": []}


def test_empty_query_popularity_and_pagination(engine):
    result = engine.search("")
    assert result["total"] == 6
    assert result["items"][0]["id"] == "P3"
    assert ids(engine.search("", offset=2, limit=2)) == ids(result)[2:4]
    assert engine.search("", offset=99)["items"] == []


def test_filters_compose(engine):
    result = engine.search("", year_min=2019, year_max=2020, venue="SIGIR", topic="retrieval", open_access=True)
    assert ids(result) == ["P4"]
    assert engine.search("", topic="unavailable")["total"] == 0


def test_tokenization_and_parser_errors():
    assert tokens("The GRAPH and graph GCN") == ["graph", "graph", "gcn"]
    with pytest.raises(ValueError, match="quotation"):
        query_parts('"unfinished phrase')


def test_bm25_single_term_matches_formula(engine):
    column = engine.vectorizer.vocabulary_["graph"]
    tf = engine.counts[:, column].toarray().ravel()
    n = len(engine.ids)
    df = np.count_nonzero(tf)
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
    expected = idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * engine.length / engine.avg_length))
    np.testing.assert_allclose(engine.lexical_scores("graph"), expected)


def test_graph_respects_citing_to_cited_direction(engine):
    assert set(engine.graph.edges) == {("P1", "P3"), ("P2", "P3"), ("P2", "P1"), ("P4", "P3"), ("P6", "P3")}
    assert engine.pagerank["P3"] > engine.pagerank["P2"]
    assert engine.authority["P3"] > engine.authority["P2"]
    assert engine.hubs["P2"] > engine.hubs["P3"]
    assert sum(engine.pagerank.values()) == pytest.approx(1)
    assert sum(engine.authority.values()) == pytest.approx(1)
    assert all(v >= 0 for v in engine.authority.values())


def test_empty_and_edgeless_graph_metrics():
    assert graph_metrics(nx.DiGraph()) == ({}, {}, {}, {})
    g = nx.DiGraph()
    g.add_nodes_from(["a", "b"])
    pr, authority, hub, communities = graph_metrics(g)
    assert pr == pytest.approx({"a": 0.5, "b": 0.5})
    assert all(v == 0 for v in authority.values())
    assert all(v == 0 for v in hub.values())
    assert len(set(communities.values())) == 2


def test_subgraph_contains_focus_and_correct_edges(engine):
    result = engine.subgraph("P3", limit=3)
    node_ids = {n["id"] for n in result["nodes"]}
    assert "P3" in node_ids and len(node_ids) == 3
    assert all(e["source"] in node_ids and e["target"] in node_ids for e in result["edges"])
    assert all(engine.graph.has_edge(e["source"], e["target"]) for e in result["edges"])
    assert all(math.isfinite(n[coord]) for n in result["nodes"] for coord in ["x", "y"])
    assert engine.subgraph("P3", 1)["edges"] == []
    with pytest.raises(KeyError):
        engine.subgraph("missing")


@pytest.mark.parametrize("method", ["content", "graph", "hybrid", "popular"])
def test_recommendation_excludes_seeds_and_returns_finite_explanations(engine, method):
    result = engine.recommend(["P1", "P1"], limit=99, method=method)
    assert not result["cold_start"]
    assert len(ids(result)) == len(set(ids(result))) == 5
    assert "P1" not in ids(result)
    assert all(p["reason"] and math.isfinite(p["score"]) for p in result["items"])
    assert ids(result) == ids(engine.recommend(["P1"], limit=99, method=method))


def test_recommendation_content_graph_coldstart_and_all_seeds(engine):
    assert ids(engine.recommend(["P1"], method="content", diversity=0))[0] == "P2"
    assert ids(engine.recommend(["P1"], method="graph", diversity=0))[0] == "P3"
    cold = engine.recommend([], limit=2)
    assert cold["cold_start"] and ids(cold)[0] == "P3" and len(ids(cold)) == 2
    assert engine.recommend(engine.ids)["items"] == []


def test_maximal_diversity_reduces_selected_redundancy(engine):
    relevant = engine.recommend(["P1"], method="content", limit=2, diversity=0)
    diverse = engine.recommend(["P1"], method="content", limit=2, diversity=1)
    def similarity(result):
        a, b = [engine.pos[pid] for pid in ids(result)]
        return (engine.tfidf[a] @ engine.tfidf[b].T).toarray().item()
    assert similarity(diverse) < similarity(relevant)


@pytest.mark.parametrize("kwargs", [{"seeds": ["missing"]}, {"seeds": [], "diversity": -0.1}, {"seeds": [], "method": "unknown"}])
def test_recommendation_rejects_invalid_options(engine, kwargs):
    with pytest.raises(ValueError):
        engine.recommend(**kwargs)


@pytest.mark.parametrize("text,label", [
    ("The evaluation is strong.", "positive"),
    ("The evaluation is not strong.", "critical"),
    ("The evaluation is not weak.", "positive"),
    ("The writing is unclear.", "critical"),
    ("The evaluation is described.", "mixed"),
])
def test_opinion_polarity_and_negation(text, label):
    result = opinion(text)
    assert result["label"] == label
    assert result["evidence"][0]["text"] == text


def test_opinion_evidence_preserves_sentences_and_aspects():
    text = "The experiments are thorough. The code is missing. A general observation."
    result = opinion(text)
    assert len(result["evidence"]) == 2
    assert result["evidence"][0]["aspects"] == ["evaluation"]
    assert result["evidence"][1]["aspects"] == ["reproducibility"]
    assert all(e["text"] in text for e in result["evidence"])
    assert opinion("")["evidence"] == []


def test_information_extraction_exact_spans():
    text = "Việt Nam: BERT uses SQuAD and MS MARCO; compare BM25, GCN and CIFAR-100. See 10.1234/abc and 2301.12345v2"
    entities = extract(text)
    assert {e["text"] for e in entities["datasets"]} == {"SQuAD", "MS MARCO", "CIFAR-100"}
    assert {e["text"] for e in entities["methods"]} == {"BERT", "BM25", "GCN"}
    assert entities["doi"][0]["text"] == "10.1234/abc"
    assert entities["arxiv"][0]["text"] == "2301.12345v2"
    for group in entities.values():
        for entity in group:
            assert text[entity["start"]:entity["end"]] == entity["text"]


def test_empty_corpus_rejected():
    with pytest.raises(ValueError, match="paper"):
        MiningEngine([])


def test_duplicate_paper_identifiers_rejected(papers):
    with pytest.raises(ValueError, match="[Dd]uplicate|unique"):
        MiningEngine(papers + [dict(papers[0])])


def test_exclusions_respect_word_boundaries(engine):
    assert ids(engine.search("paragraph -graph")) == ["P6"]
    assert set(ids(engine.search("-graph"))) == {"P3", "P5", "P6"}


def test_phrase_boundaries_and_whitespace(papers):
    papers[0]["title"] = "Neural networks"
    papers[0]["abstract"] = "A neural\nnetwork architecture"
    papers[1]["title"] = "Neural networks"
    papers[1]["abstract"] = "A neural network architecture"
    papers[2]["title"] = "Neural networks"
    papers[2]["abstract"] = "A neural networked architecture"
    engine = MiningEngine(papers[:3])
    assert set(ids(engine.search('"neural network"'))) == {"P1", "P2"}


@pytest.mark.parametrize("text", ["", "the and or", "12345 !?!"])
def test_corpus_without_indexable_vocabulary_rejected(papers, text):
    paper = dict(papers[0], title=text, abstract=text)
    with pytest.raises(ValueError, match="indexable"):
        MiningEngine([paper])


def test_and_uses_token_boundaries_in_multiline_query(engine):
    expected = ids(engine.search("graph AND retrieval"))
    assert expected and ids(engine.search("graph OR retrieval")) != expected
    assert ids(engine.search("graph\tAND\nretrieval")) == expected
