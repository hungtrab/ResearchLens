"""Co-citation means a shared citing paper, with independently checkable evidence."""
import json

import pytest
from fastapi.testclient import TestClient

from researchlens.api import create_app
from researchlens.mining import MiningEngine


@pytest.fixture
def cocitation_papers(papers):
    # W1 cites two seeds and repeats A: it still casts exactly one vote for A.
    # W4 shares references with W1, but is never cited together with a seed.
    # U is directly connected to S1 without a shared citer.
    citations = {
        "C": [], "B": [], "A": ["S1"], "S2": [], "S1": ["U"],
        "U": ["S1"], "W1": ["S1", "S2", "A", "A", "B"],
        "W2": ["S1", "A", "C"], "W3": ["S2", "A"],
        "W4": ["B", "C"], "W5": ["S1", "A"],
    }
    return [dict(papers[0], id=pid, title=f"Research paper {pid}", references=references)
            for pid, references in citations.items()]


@pytest.fixture
def cocitation_engine(cocitation_papers):
    return MiningEngine(cocitation_papers)


def assert_evidence(result, papers, seeds):
    references = {paper["id"]: set(paper["references"]) for paper in papers}
    seeds = set(seeds)
    for candidate in result["items"]:
        pid = candidate["id"]
        supported_by = {source for source, refs in references.items()
                        if pid in refs and refs & seeds}
        assert pid not in seeds
        assert supported_by
        assert candidate["co_citation_count"] == len(supported_by)
        assert candidate["witnesses"]
        pairs = [(w["paper_id"], w["seed_id"]) for w in candidate["witnesses"]]
        assert len(pairs) == len(set(pairs))
        for source, seed in pairs:
            assert source in supported_by and seed in seeds
            assert {pid, seed}.issubset(references[source])


def test_co_citation_counts_distinct_citers_across_multiple_seeds(cocitation_engine, cocitation_papers):
    result = cocitation_engine.recommend(["S1", "S2"], method="cocitation", limit=20)
    assert result["cold_start"] is False
    assert [(p["id"], p["co_citation_count"]) for p in result["items"]] == [("A", 4), ("B", 1), ("C", 1)]
    assert_evidence(result, cocitation_papers, ["S1", "S2"])


def test_seed_duplicates_and_order_do_not_inflate_support(cocitation_engine):
    first = cocitation_engine.recommend(["S1", "S2"], method="cocitation")
    repeated = cocitation_engine.recommend(["S2", "S1", "S1", "S2"], method="cocitation")
    assert repeated == first


def test_single_seed_does_not_count_citers_that_only_reference_other_seeds(cocitation_engine, cocitation_papers):
    result = cocitation_engine.recommend(["S1"], method="cocitation")
    assert [(p["id"], p["co_citation_count"]) for p in result["items"]] == [("A", 3), ("B", 1), ("C", 1), ("S2", 1)]
    assert_evidence(result, cocitation_papers, ["S1"])


def test_unsupported_candidates_are_never_used_to_fill_limit(cocitation_engine):
    result = cocitation_engine.recommend(["S1", "S2"], method="cocitation", limit=50)
    assert {p["id"] for p in result["items"]} == {"A", "B", "C"}
    # Shared bibliography (W4 -> B,C) alone is not co-citation with W4.
    assert cocitation_engine.recommend(["W4"], method="cocitation") == {"cold_start": False, "items": []}
    assert cocitation_engine.recommend(cocitation_engine.ids, method="cocitation")["items"] == []


def test_co_citation_order_is_deterministic_and_independent_of_mmr(cocitation_engine, cocitation_papers):
    plain = cocitation_engine.recommend(["S1", "S2"], method="cocitation", diversity=0)
    diverse = cocitation_engine.recommend(["S1", "S2"], method="cocitation", diversity=1)
    reverse_engine = MiningEngine(list(reversed(cocitation_papers)))
    reverse = reverse_engine.recommend(["S1", "S2"], method="cocitation", diversity=1)
    selected = lambda result: [(p["id"], p["co_citation_count"], p["witnesses"]) for p in result["items"]]
    assert selected(plain) == selected(diverse) == selected(reverse)
    assert [p["id"] for p in plain["items"]] == ["A", "B", "C"]
    limited = cocitation_engine.recommend(["S1", "S2"], method="cocitation", limit=2)
    assert [p["id"] for p in limited["items"]] == ["A", "B"]


def test_co_citation_cold_start_preserves_existing_influence_fallback(cocitation_engine):
    result = cocitation_engine.recommend([], method="cocitation", limit=3)
    expected = cocitation_engine.recommend([], method="popular", limit=3)
    assert result == expected and result["cold_start"] is True


def test_api_co_citation_counts_witnesses_empty_support_and_validation(app_config, cocitation_papers):
    path = app_config["data_dir"] / "papers.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in cocitation_papers))
    (app_config["data_dir"] / "reviews.jsonl").unlink()
    with TestClient(create_app(**app_config)) as client:
        response = client.post("/api/recommendations", json={"method": "cocitation", "seeds": ["S1", "S2"], "limit": 10, "diversity": 1})
        assert response.status_code == 200
        result = response.json()
        assert [(p["id"], p["co_citation_count"]) for p in result["items"]] == [("A", 4), ("B", 1), ("C", 1)]
        assert_evidence(result, cocitation_papers, ["S1", "S2"])
        empty = client.post("/api/recommendations", json={"method": "cocitation", "seeds": ["W4"]})
        assert empty.status_code == 200 and empty.json() == {"cold_start": False, "items": []}
        assert client.post("/api/recommendations", json={"method": "cocitation", "seeds": []}).json()["cold_start"]
        assert client.post("/api/recommendations", json={"method": "cocitation", "seeds": ["unknown"]}).status_code == 422
