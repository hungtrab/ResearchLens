import sqlite3

import pytest

from researchlens.reviews import deduplicate, predict, rating_label, train_model
from researchlens.store import Store
from scripts.collect import abstract, title_key


def test_store_profiles_persist_and_save_idempotently(tmp_path, papers):
    path = tmp_path / "nested" / "state.sqlite"
    store = Store(path)
    store.ingest(papers)
    store.save("alice", "P1")
    store.save("alice", "P1")
    store.save("bob", "P2")
    restored = Store(path)
    assert restored.saved("alice") == ["P1"]
    assert restored.saved("bob") == ["P2"]
    restored.remove("alice", "P2")
    assert restored.saved("bob") == ["P2"]
    restored.remove("alice", "P1")
    assert restored.saved("alice") == []


def test_store_foreign_keys_rollback_and_parameterization(tmp_path, papers):
    store = Store(tmp_path / "state.sqlite")
    store.ingest(papers)
    with pytest.raises(sqlite3.IntegrityError):
        store.save("alice", "unknown")
    query = "graph'); DROP TABLE papers; --"
    store.remember("alice", query)
    assert store.history("alice") == [query]
    store.save("alice", "P1")
    assert store.saved("alice") == ["P1"]


def test_history_is_recent_bounded_and_profile_scoped(tmp_path):
    store = Store(tmp_path / "state.sqlite")
    for i in range(30):
        store.remember("alice", f"query {i}")
    store.remember("bob", "private query")
    assert store.history("alice") == [f"query {i}" for i in range(29, 21, -1)]
    assert store.history("bob") == ["private query"]
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM history WHERE profile=?", ("alice",)).fetchone()[0] == 20


def test_ingestion_updates_metadata_without_losing_library(tmp_path, papers):
    store = Store(tmp_path / "state.sqlite")
    store.ingest(papers)
    store.save("alice", "P1")
    papers[0]["title"] = "Corrected title"
    store.ingest(papers)
    assert store.saved("alice") == ["P1"]
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == len(papers)
        assert "Corrected title" in db.execute("SELECT payload FROM papers WHERE id='P1'").fetchone()[0]


def test_review_deduplication_normalizes_whitespace_case_but_keeps_other_papers():
    rows = [
        dict(paper_id="P1", text="The evaluation is strong."),
        dict(paper_id="P1", text=" THE   Evaluation\nis strong. "),
        dict(paper_id="P2", text="The evaluation is strong."),
    ]
    assert deduplicate(rows) == [rows[0], rows[2]]
    assert deduplicate(deduplicate(rows)) == deduplicate(rows)


@pytest.mark.parametrize("rating,label", [(1, "critical"), (4, "critical"), (5, "mixed"), (6, "mixed"), (7, "positive"), (10, "positive")])
def test_rating_labels(rating, label):
    assert rating_label(rating) == label


def test_review_model_excludes_test_split_and_predicts_probabilities(reviews):
    reviews += [dict(id="heldout", paper_id="P6", text="heldoutword heldoutword", rating=10, split="test")]
    model = train_model(reviews)
    assert "heldoutword" not in model.steps[0][1].vocabulary_
    result = predict(model, "The evaluation is strong and experiments are convincing")
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert result["label"] in result["probabilities"]
    assert all(0 <= v <= 1 for v in result["probabilities"].values())


def test_review_model_handles_no_or_single_class_training(reviews):
    assert train_model([]) is None
    assert train_model([reviews[0]]) is None
    assert predict(None, "An opinion") is None


def test_review_model_handles_training_with_no_shared_vocabulary():
    rows = [
        dict(paper_id="P1", text="excellent", rating=9, split="train"),
        dict(paper_id="P2", text="weak", rating=2, split="train"),
    ]
    # A tiny imported corpus should still support retrieval and heuristic opinions.
    model = train_model(rows)
    assert model is None or predict(model, "excellent") is not None


def test_collector_metadata_helpers():
    assert abstract({"graph": [0, 2], "neural": [1]}) == "graph neural graph"
    assert abstract(None) == ""
    assert title_key("Graph Neural Networks: A Review!") == title_key("GRAPH neural networks a review")
