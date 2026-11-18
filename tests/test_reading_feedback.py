"""Page-two evaluation may use the visible click, never judgments of unseen papers."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix


@pytest.fixture
def feedback(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("evaluate_reading_feedback")


def test_next_pages_excludes_all_first_page_candidates_under_every_rule(feedback):
    original = np.array([9., 8., 7., 6., 5., 4., 3.])
    clicked = np.array([100., 90., 80., 0., 10., 6., 3.])
    first = np.array([0, 1, 2])
    saved = (original.copy(), clicked.copy(), first.copy())
    result = feedback.next_pages(original, clicked, first)
    assert {name: values.tolist() for name, values in result.items()} == {
        "continue": [3, 4, 5], "clicked_only": [4, 5, 6], "centroid": [4, 5, 3]}
    assert all(set(values).isdisjoint(first) and len(set(values)) == len(values) for values in result.values())
    for before, after in zip(saved, [original, clicked, first]):
        np.testing.assert_array_equal(before, after)


def test_next_pages_stable_ties_and_no_positive_signal_fallback(feedback):
    scores = np.ones(7)
    result = feedback.next_pages(scores, scores, np.array([1, 3, 5]))
    assert all(values.tolist() == [0, 2, 4] for values in result.values())
    reordered = feedback.next_pages(scores, scores, np.array([5, 1, 3]))
    assert all(np.array_equal(result[name], reordered[name]) for name in result)


@pytest.mark.parametrize("size,expected", [(3, []), (4, [3]), (5, [3, 4])])
def test_next_pages_never_backfills_with_seen_candidates(feedback, size, expected):
    scores = np.arange(size, 0, -1, dtype=float)
    result = feedback.next_pages(scores, scores, np.array([0, 1, 2]))
    assert all(values.tolist() == expected for values in result.values())


def test_feedback_run_cannot_use_unseen_labels_to_select_second_page(feedback, tmp_path, monkeypatch):
    candidates = list("ABCDEFG")
    papers = [dict(paper_id=pid, title=("Query " if pid in {"Q", "H"} else "Candidate ")+pid,
                   abstract=[f"Method evidence for {pid}."], pred_labels=["method_label"])
              for pid in ["Q", "H", *candidates]]
    splits = {"method": {"fold1_dev": ["H_method"], "fold1_test": ["Q_method"]},
              "result": {"fold1_dev": ["H_result"], "fold1_test": []},
              "background": {"fold1_dev": [], "fold1_test": []}}
    judgments = {"method": {"Q": dict(cands=candidates, relevance_adju=[3, 0, 0, 0, 0, 3, 0])},
                 "result": {}, "background": {}}
    def source(cache, name):
        if name == "abstracts-csfcube-preds.jsonl":
            return "\n".join(json.dumps(p) for p in papers).encode()
        if name == "evaluation_splits.json":
            return json.dumps(splits).encode()
        facet = name.removeprefix("test-pid2anns-csfcube-").removesuffix(".json")
        return json.dumps(judgments[facet]).encode()
    monkeypatch.setattr(feedback, "source", source)
    monkeypatch.setattr(feedback, "DEST", tmp_path)
    (tmp_path / "feedback_protocol.md").write_text("Frozen feedback fixture")
    class Lexical:
        def transform(self, texts):
            return csr_matrix(np.zeros((len(texts), 1)))
    def extractor(include_laya):
        assert include_laya is False
        return SimpleNamespace(lex=Lexical())
    monkeypatch.setattr(feedback, "DecisionFeatures", extractor)
    monkeypatch.setattr(feedback, "load_policy", lambda: ({"weights": [0, 1], "scale": [1, 1]}, "fixture"))
    vectors = {"A": [9, 10], "B": [8, 0], "C": [7, 0], "D": [6, 0],
               "E": [5, 9], "F": [4, 8], "G": [3, 7]}
    def encode(encoder, texts):
        return np.array([vectors[text[len("Candidate ")]] if text.startswith("Candidate ") else [1, 0]
                         for text in texts], dtype=np.float32)
    monkeypatch.setattr(feedback, "encode", encode)
    before = feedback.run(tmp_path / "cache")
    first = before["rows"][0]
    assert len(before["rows"]) == 1 and first["query"] == "Q"
    assert first["first"] == ["A", "B", "C"] and first["click"] == "A"
    assert first["next"] == {"continue": ["D", "E", "F"], "clicked_only": ["E", "F", "G"], "centroid": ["E", "F", "G"]}
    # Keep every visible judgment and therefore the simulated click unchanged;
    # invert only labels that the reader has not yet seen.
    judgments["method"]["Q"]["relevance_adju"][3:] = [3, 3, 0, 3]
    after = feedback.run(tmp_path / "cache")
    second = after["rows"][0]
    assert second["first"] == first["first"] and second["click"] == first["click"]
    assert second["next"] == first["next"]
    assert second["next_hits"] != first["next_hits"]
    assert all(set(values).isdisjoint(first["first"]) for values in second["next"].values())
