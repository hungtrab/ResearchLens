"""Fusion ranks a candidate by its content/graph rank and its co-citation rank together."""
import json

import pytest
from fastapi.testclient import TestClient

from researchlens.api import create_app
from researchlens.mining import MiningEngine

# S has no references, so personalized PageRank keeps all mass on S and the
# content/graph rank is the content rank: C1 (same title as S), B (3 shared
# terms), N (1 shared term), then K, W1, W2, W3 tied at zero in corpus order.
# W1 and W2 cite S with K, W3 cites S with B: co-citation ranks K first, B second.
TITLES = {
    "S": ("alpha beta gamma delta", []),
    "C1": ("alpha beta gamma delta", []),
    "B": ("alpha beta gamma omega", []),
    "N": ("alpha sigma tau", []),
    "K": ("zeta eta theta", []),
    "W1": ("kappa lambda", ["S", "K"]),
    "W2": ("rho phi", ["S", "K"]),
    "W3": ("chi psi mu", ["S", "B"]),
}
BEST = 1 / 62 + 1 / 62


@pytest.fixture
def fusion_papers(papers):
    return [dict(papers[0], id=pid, title=title, abstract="", references=refs)
            for pid, (title, refs) in TITLES.items()]


@pytest.fixture
def fusion_engine(fusion_papers):
    return MiningEngine(fusion_papers)


def ids(result):
    return [p["id"] for p in result["items"]]


def by_id(result):
    return {p["id"]: p for p in result["items"]}


def test_fusion_is_the_default_recommender(fusion_engine):
    default = fusion_engine.recommend(["S"], diversity=0)
    assert default == fusion_engine.recommend(["S"], method="fusion", diversity=0)
    assert ids(default) == ["B", "K", "C1", "N", "W1", "W2", "W3"]
    assert ids(fusion_engine.recommend(["S"], method="hybrid", diversity=0))[:3] == ["C1", "B", "N"]


def test_candidate_ranked_well_by_both_signals_beats_single_signal_leaders(fusion_engine):
    result = fusion_engine.recommend(["S"], limit=3, diversity=0)
    assert ids(result) == ["B", "K", "C1"]
    scores = [p["score"] for p in result["items"]]
    assert scores == pytest.approx([1.0, (1 / 61 + 1 / 64) / BEST, (1 / 61) / BEST])


def test_fusion_never_returns_seeds(fusion_engine):
    result = fusion_engine.recommend(["S", "K", "S"], limit=50, diversity=0)
    assert set(ids(result)) == {"C1", "B", "N", "W1", "W2", "W3"}
    assert fusion_engine.recommend(fusion_engine.ids, method="fusion")["items"] == []


def test_co_cited_fusion_items_expose_witnesses(fusion_engine):
    items = by_id(fusion_engine.recommend(["S"], diversity=0))
    seed = {"seed_id": "S", "seed_title": "alpha beta gamma delta"}
    assert items["K"]["co_citation_count"] == 2
    assert items["K"]["witnesses"] == [dict(seed, paper_id="W1", paper_title="kappa lambda"),
                                       dict(seed, paper_id="W2", paper_title="rho phi")]
    assert items["K"]["reason"] == ("Co-citation rank 1 (2 papers cite it with a saved paper) + content/graph rank 4, "
                                    "reciprocal rank fusion k=60.")
    assert items["B"]["co_citation_count"] == 1
    assert items["B"]["witnesses"] == [dict(seed, paper_id="W3", paper_title="chi psi mu")]
    assert items["B"]["reason"].startswith("Co-citation rank 2 (1 paper cites it with a saved paper) + "
                                           "content/graph rank 2, reciprocal rank fusion k=60. Shared terms: ")
    assert set(items["B"]["shared_terms"]) == {"alpha", "beta", "gamma"}


def test_fusion_fills_the_limit_with_candidates_nobody_co_cites(fusion_engine):
    assert ids(fusion_engine.recommend(["S"], method="cocitation", limit=5)) == ["K", "B"]
    result = fusion_engine.recommend(["S"], limit=5, diversity=0)
    assert ids(result) == ["B", "K", "C1", "N", "W1"]
    uncited = by_id(result)
    for pid in ("C1", "N", "W1"):
        assert uncited[pid]["co_citation_count"] == 0 and uncited[pid]["witnesses"] == []
    assert uncited["C1"]["reason"].startswith(
        "Content/graph rank 1, not co-cited with a saved paper, reciprocal rank fusion k=60.")
    assert uncited["W1"]["reason"] == "Content/graph rank 5, not co-cited with a saved paper, reciprocal rank fusion k=60."


def test_fusion_respects_diversity(fusion_engine):
    diverse = fusion_engine.recommend(["S"], diversity=1)
    assert ids(diverse) == ["B", "K", "W1", "W2", "W3", "N", "C1"]
    assert all(p["reason"].endswith(" Diversity reranking enabled (weight 1).") for p in diverse["items"])


@pytest.mark.parametrize("method", ["rrf", "Fusion", ""])
def test_unknown_method_is_rejected_with_seeds(fusion_engine, method):
    with pytest.raises(ValueError):
        fusion_engine.recommend(["S"], method=method)


def test_api_defaults_to_fusion_without_decision_model(app_config, fusion_papers):
    (app_config["data_dir"] / "papers.jsonl").write_text("".join(json.dumps(p) + "\n" for p in fusion_papers))
    (app_config["data_dir"] / "reviews.jsonl").unlink()
    with TestClient(create_app(**app_config)) as client:
        client.app.state.decision = client.app.state.decision_topic = None
        response = client.post("/api/recommendations", json={"seeds": ["S"], "diversity": 0})
        assert response.status_code == 200
        result = response.json()
        assert ids(result) == ["B", "K", "C1", "N", "W1", "W2", "W3"]
        assert [w["paper_id"] for w in by_id(result)["B"]["witnesses"]] == ["W3"]
        default = client.post("/api/recommendations", json={"seeds": ["S"]}).json()
        assert default == client.post("/api/recommendations", json={"seeds": ["S"], "method": "fusion"}).json()
        assert client.post("/api/recommendations", json={"seeds": ["S"], "method": "rrf"}).status_code == 422
