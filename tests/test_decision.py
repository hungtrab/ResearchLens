"""Behaviour of the decision-model recommender, as the app and the API expose it."""
import pickle
from pathlib import Path

import numpy as np
import pytest

from researchlens.decision import FEATURES, DecisionModel, with_ranks

ROOT = Path(__file__).resolve().parents[1]
MODEL, VECTORS = ROOT / "artifacts/decision_model.pkl", ROOT / "artifacts/specter.npz"
needs_model = pytest.mark.skipif(not (MODEL.exists() and VECTORS.exists()), reason="decision model artifacts missing")


@pytest.fixture
def decision():
    return DecisionModel.load(MODEL, VECTORS)


def test_decision_method_needs_a_model(engine):
    with pytest.raises(ValueError, match="Decision model"):
        engine.recommend(["P1"], method="decision")


@needs_model
def test_decision_ranks_by_probability_and_explains_each_item(engine, decision):
    items = engine.recommend(["P1"], limit=5, method="decision", diversity=0, decision=decision)["items"]
    assert items and "P1" not in [p["id"] for p in items]
    scores = [p["score"] for p in items]
    assert scores == sorted(scores, reverse=True) and all(0 <= s <= 1 for s in scores)
    assert all(p["reason"].startswith(f"Decision model score {p['score']:.2f}") for p in items)


@needs_model
def test_decision_reason_counts_the_witnesses_it_shows(engine, decision):
    for item in engine.recommend(["P1"], limit=5, method="decision", diversity=0, decision=decision)["items"]:
        n = len(item["witnesses"])
        assert (f"{n} co-citing paper" in item["reason"]) == (n > 0)


@needs_model
def test_papers_without_an_embedding_get_zero_vectors(engine, decision):
    assert not np.any(decision.vectors(engine))  # fixture ids are not in the SPECTER table


def test_load_rejects_a_model_trained_on_other_features(tmp_path):
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(dict(variant="gbdt_rank", model=None, features=FEATURES[:-1])))
    with pytest.raises(ValueError, match="different features"):
        DecisionModel.load(path, VECTORS)


def test_rank_features_are_scale_free():
    x = np.array([[1.0, 10.0], [2.0, 30.0], [3.0, 20.0]])
    ranked = with_ranks(x)
    assert ranked.shape == (3, 4)
    assert np.array_equal(ranked[:, 2:], with_ranks(x * 1000)[:, 2:])
    assert ranked[:, 2:].max() == 1


@needs_model
def test_api_serves_decision_recommendations(client):
    response = client.post("/api/recommendations", json=dict(seeds=["P1"], method="decision", limit=3, diversity=0))
    assert response.status_code == 200
    assert response.json()["items"][0]["reason"].startswith("Decision model score ")


def test_api_reports_a_missing_model(client):
    client.app.state.decision = None
    response = client.post("/api/recommendations", json=dict(seeds=["P1"], method="decision"))
    assert response.status_code == 422 and "Decision model" in response.json()["detail"]


@needs_model
def test_top_recommendation_is_the_models_most_likely_paper(engine, decision):
    ids, probability, _ = decision.rank(engine, ["P1"])
    top = engine.recommend(["P1"], limit=1, method="decision", diversity=0, decision=decision)["items"][0]
    assert top["id"] == ids[int(np.argmax(probability))]
    assert top["score"] == pytest.approx(float(probability.max()))
