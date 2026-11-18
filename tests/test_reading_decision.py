"""Independent protocol regressions for the reading-decision study, offline only."""
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts import fit_reading_decision as decision


def row(group, pid, grade, value=0.5, split="dev"):
    return dict(group=group, pid=pid, grade=grade, split=split,
                lexical=value, semantic=value * 0.8, original=value * 0.6,
                specific=value * 0.4)


def write_features(path, rows):
    """Record a fresh extraction digest for intentional fixture changes."""
    features = path / "features.jsonl"
    features.write_text("".join(json.dumps(r)+"\n" for r in rows))
    source = dict(laya=["fixture-laya", "pinned-revision"],
                  encoder=["fixture-encoder", "pinned-revision"],
                  data_revision="fixture-data", feature_script_sha256="fixture-code",
                  corpus_sha256="fixture-corpus",
                  features_sha256=hashlib.sha256(features.read_bytes()).hexdigest())
    (path / "feature_provenance.json").write_text(json.dumps(source))


def brute_threshold(labels, scores):
    """Enumerate finite observed cutoffs and calculate sensitivity/specificity."""
    positives = sum(labels)
    negatives = len(labels) - positives
    evaluated = []
    for cutoff in set(scores):
        predictions = [score >= cutoff for score in scores]
        true_positive = sum(actual and predicted for actual, predicted in zip(labels, predictions))
        true_negative = sum(not actual and not predicted for actual, predicted in zip(labels, predictions))
        accuracy = 0.5 * (true_positive / positives + true_negative / negatives)
        evaluated.append((round(accuracy, 12), cutoff))
    return max(evaluated)[1]


def test_disjoint_training_excludes_paper_and_query_role_overlap_without_mutation():
    held = [row("query-held", "paper-held", 3)]
    candidates = [
        row("other", "paper-held", 0),       # same candidate in another query
        row("query-held", "other-paper", 0), # same held query
        row("other", "query-held", 0),       # held query used as a candidate
        row("paper-held", "other-paper", 0), # held candidate used as a query
        row("safe-query", "safe-paper", 2),
        row("safe-query-2", "safe-paper-2", 0),
    ]
    original = copy.deepcopy(candidates)
    assert decision.disjoint_train(candidates, held) == candidates[-2:]
    assert candidates == original


def test_threshold_cross_validation_scores_each_query_from_disjoint_training(monkeypatch):
    dev = [row("Q1", "shared", 3, .9), row("Q1", "N1", 0, .1),
           row("Q2", "shared", 2, .8), row("Q2", "N2", 1, .2),
           row("Q2", "Q1", 1, .3), row("Q3", "P3", 3, .7), row("Q3", "N3", 0, .4)]
    fits, scored = [], []
    def fit(training, columns):
        head = {"training": copy.deepcopy(training), "columns": columns}
        fits.append(head)
        return head
    def score(held, head):
        held_group = held[0]["group"]
        assert all(r["group"] == held_group for r in held)
        forbidden = {r[key] for r in held for key in ["pid", "group"]}
        expected = [r for r in dev if r["pid"] not in forbidden and r["group"] not in forbidden]
        assert head["training"] == expected
        assert head["columns"] == ["lexical"]
        scored.extend((r["group"], r["pid"]) for r in held)
        return np.array([r["lexical"] for r in held])
    monkeypatch.setattr(decision, "fit", fit)
    monkeypatch.setattr(decision, "score", score)
    cutoff = decision.choose_threshold(dev, ["lexical"])
    assert len(fits) == 3
    assert sorted(scored) == sorted((r["group"], r["pid"]) for r in dev)
    assert cutoff == brute_threshold([r["grade"] >= 2 for r in dev], [r["lexical"] for r in dev])


@pytest.mark.parametrize("labels,scores", [
    ([False, True, False, True], [.1, .4, .6, .9]),
    ([False, True], [.5, .5]),
    ([False, False, True, True], [.2, .3, .7, .9]),
    ([True, False, True, False, False], [.9, .8, .6, .2, .1]),
])
def test_threshold_maximizes_balanced_accuracy_with_higher_finite_tie_break(labels, scores):
    cutoff = decision.balanced_threshold(labels, scores)
    assert math.isfinite(cutoff)
    assert cutoff == brute_threshold(labels, scores)
    if scores == [.1, .4, .6, .9]:
        # Both .4 and .9 achieve .75 balanced accuracy; choose the higher cutoff.
        assert cutoff == .9


def test_fit_scaler_and_classifier_use_only_requested_features_and_relevance_labels():
    rows = [row("Q", f"P{i}", grade, value) for i, (grade, value) in enumerate([(0, -2), (1, -1), (2, 1), (3, 2)])]
    head = decision.fit(rows, ["lexical"])
    assert head["columns"] == ["lexical"]
    assert head["center"] == pytest.approx([0])
    assert head["scale"] == pytest.approx([math.sqrt(2.5)])
    scores = decision.score(rows, head)
    assert np.all(np.diff(scores) > 0)
    assert list(scores >= .5) == [False, False, True, True]
    changed = [dict(r, grade=3-r["grade"], semantic=1e9, original=-1e9, specific=0) for r in rows]
    np.testing.assert_allclose(decision.score(changed, head), scores)


def test_score_respects_stored_feature_order_and_exported_standardization():
    head = dict(columns=["semantic", "lexical"], center=[2, 4], scale=[2, 2], weights=[1, -.5], bias=.25)
    scores = decision.score([dict(semantic=6, lexical=2), dict(semantic=2, lexical=4)], head)
    assert scores.tolist() == pytest.approx([1/(1+math.exp(-2.75)), 1/(1+math.exp(-.25))])


@pytest.fixture
def pilot():
    # The same paper can receive a different judgment for a different brief.
    rows = [row("A", "shared", 3), row("A", "N1", 0),
            row("B", "shared", 0), row("B", "P2", 2), row("B", "N2", 1),
            row("C", "P3", 3), row("C", "N3", 0)]
    return rows, [.9, .1, .95, .7, .2, .6, .3]


def test_pilot_calibration_excludes_entire_brief_and_its_papers_from_other_briefs(pilot, monkeypatch):
    rows, scores = pilot
    training_calls = []
    actual_threshold = decision.balanced_threshold
    def threshold(labels, values):
        training_calls.append((list(labels), list(values)))
        return actual_threshold(labels, values)
    monkeypatch.setattr(decision, "balanced_threshold", threshold)
    result = decision.calibrate_pilot(rows, scores)
    for group, training in zip(["A", "B", "C"], training_calls):
        held_papers = {r["pid"] for r in rows if r["group"] == group}
        expected = [i for i, r in enumerate(rows) if r["group"] != group and r["pid"] not in held_papers]
        assert training == ([rows[i]["grade"] >= 2 for i in expected], [scores[i] for i in expected])
    calls = {(d["group"], d["pid"]): d for d in result["decisions"]}
    assert len(calls) == len(rows) == 7
    assert calls[("A", "shared")]["label"] is True
    assert calls[("B", "shared")]["label"] is False
    assert {d["cutoff"] for d in result["decisions"] if d["group"] == "A"} == {.6}
    assert {d["cutoff"] for d in result["decisions"] if d["group"] == "B"} == {.6}
    assert {d["cutoff"] for d in result["decisions"] if d["group"] == "C"} == {.7}
    assert result["correct"] == 5 and result["n"] == 7
    assert result["balanced_accuracy"] == pytest.approx(17/24)


def test_held_brief_labels_and_repeated_paper_labels_cannot_change_its_cutoff(pilot):
    rows, scores = pilot
    before = decision.calibrate_pilot(rows, scores)
    changed = [dict(r, grade=3-r["grade"]) if r["group"] == "A" or r["pid"] == "shared" else dict(r) for r in rows]
    after = decision.calibrate_pilot(changed, scores)
    first_cutoffs = [d["cutoff"] for d in before["decisions"] if d["group"] == "A"]
    second_cutoffs = [d["cutoff"] for d in after["decisions"] if d["group"] == "A"]
    assert first_cutoffs == second_cutoffs == [.6, .6]


def test_measure_includes_cutoff_boundary_and_uses_graded_relevance():
    rows = [row("A", "P1", 0), row("A", "P2", 3), row("B", "P3", 1), row("B", "P4", 2)]
    result = decision.measure(rows, [.1, .9, .6, .6], .6)
    assert result["positive"] == 2 and result["correct"] == 3
    assert result["confusion_tn_fp_fn_tp"] == [1, 1, 0, 2]
    assert result["balanced_accuracy"] == .75
    assert result["f1"] == .8
    assert result["brier"] == pytest.approx(.135)
    tied_ndcg = (2*(1+1/math.log2(3))) / (3+1/math.log2(3))
    assert result["ndcg_by_query"] == pytest.approx({"A": 1, "B": tied_ndcg})
    assert result["macro_ndcg_10"] == pytest.approx((1+tied_ndcg)/2)


@pytest.fixture
def study(tmp_path, monkeypatch):
    path = tmp_path / "evaluation" / "decision-study"
    path.mkdir(parents=True)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "papers.jsonl").write_text('{"id":"fixture"}\n')
    rows = [row(f"D{i}", f"DP{i}{j}", 3 if j else 0, .8 if j else .2)
            for i in range(3) for j in range(2)]
    rows += [row("D0", "TP00", 3, 9999), row("D1", "T0", 0, -9999),
             row("TP01", "other-candidate", 3, 9999)]
    rows += [row(f"T{i}", f"TP{i}{j}", 3 if j else 0, .9 if j else .1, "test")
             for i in range(2) for j in range(2)]
    rows += [row(f"B{i}", f"BP{i}{j}", 3 if j else 0, .7 if j else .3, "pilot")
             for i in range(3) for j in range(2)]
    rows.append(row("B0", "unlabeled", None, .5, "pilot"))
    write_features(path, rows)
    ranked = {f"T{i}": [[f"TP{i}0", .8], [f"TP{i}1", .2]] for i in range(2)}
    (path / "specter_method_ranked.json").write_text(json.dumps(ranked))
    monkeypatch.setattr(decision, "STUDY", path)
    return path, rows


def test_main_uses_only_disjoint_development_rows_and_keeps_pilot_pair_order(study, monkeypatch):
    path, rows = study
    actual_fit = decision.fit
    fits = []
    def fit(training, columns):
        fits.append(copy.deepcopy(training))
        assert training and all(r["split"] == "dev" for r in training)
        assert all(r["pid"].startswith("DP") for r in training)
        return actual_fit(training, columns)
    monkeypatch.setattr(decision, "fit", fit)
    decision.main()
    report = json.loads((path / "decision_comparison.json").read_text())
    assert fits
    assert report["split"] == dict(dev_queries=3, dev_before_filter=9, dev_pairs=6, test_queries=2, test_pairs=4)
    expected = [[r["group"], r["pid"]] for r in rows if r["split"] == "pilot" and r["grade"] is not None]
    assert report["prediction_order"]["local_pilot"] == expected
    assert all(len(predictions) == 6 for predictions in report["predictions"]["local_pilot"].values())
    assert all(model["local_pilot"]["n"] == 6 for model in report["models"].values())
    provenance = json.loads((path / "feature_provenance.json").read_text())
    for method, filename in [("combined", "reading_policy.json"), ("text_only", "text_policy.json")]:
        policy = json.loads((path / filename).read_text())
        assert policy["method"] == method
        assert policy["columns"] == decision.ARMS[method]
        assert policy["threshold"] == report["models"][method]["threshold"]
        assert policy["weights"] == report["models"][method]["head"]["weights"]
        assert policy["source"] == provenance


def test_exported_policy_is_invariant_to_external_test_and_pilot_labels_features(study):
    path, rows = study
    decision.main()
    filenames = ["reading_policy.json", "text_policy.json"]
    before = {filename: json.loads((path / filename).read_text()) for filename in filenames}
    for r in rows:
        if r["split"] in {"test", "pilot"} and r["grade"] is not None:
            r["grade"] = 3-r["grade"]
            for feature in ["lexical", "semantic", "original", "specific"]:
                r[feature] = 1-r[feature]
    write_features(path, rows)
    decision.main()
    after = {filename: json.loads((path / filename).read_text()) for filename in filenames}
    for filename in filenames:
        # Provenance must reflect the changed extraction; learned parameters,
        # scaling and cutoff must remain independent of test/pilot observations.
        assert after[filename]["source"]["features_sha256"] != before[filename]["source"]["features_sha256"]
        assert {k: v for k, v in after[filename].items() if k != "source"} == {
            k: v for k, v in before[filename].items() if k != "source"}


def test_main_rejects_overlap_of_external_query_splits_before_fitting(study, monkeypatch):
    path, rows = study
    rows[0]["group"] = "T0"
    write_features(path, rows)
    def forbidden(*args, **kwargs):
        pytest.fail("Fitting must not begin with overlapping dev/test query splits")
    monkeypatch.setattr(decision, "fit", forbidden)
    with pytest.raises(AssertionError):
        decision.main()
    assert not (path / "reading_policy.json").exists()


def test_main_rejects_changed_features_before_fitting_or_exporting_policy(study, monkeypatch):
    path, rows = study
    rows[0]["lexical"] += .01
    (path / "features.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    def forbidden(*args, **kwargs):
        pytest.fail("Fitting must not begin when feature provenance does not match")
    monkeypatch.setattr(decision, "fit", forbidden)
    with pytest.raises(ValueError, match="Features differ"):
        decision.main()
    assert not (path / "reading_policy.json").exists()
    assert not (path / "text_policy.json").exists()


@pytest.fixture
def reading_cli(monkeypatch):
    # Match the import path of `python scripts/decide_reading.py`; constructors
    # remain injected, so no model dependency or network download is needed.
    monkeypatch.syspath_prepend(str(Path(decision.__file__).parent))
    return importlib.import_module("decide_reading")


def test_decide_both_policy_modes_use_injected_features_and_frozen_threshold(reading_cli):
    initializations, computations = [], []
    class FakeExtractor:
        def __init__(self, include_laya):
            self.include_laya = include_laya
            initializations.append(include_laya)
        def compute(self, goal, seed, title, abstract):
            computations.append((goal, seed, title, abstract))
            features = dict(lexical=.2, semantic=.4)
            if self.include_laya:
                features.update(original=.9, specific=.8)
            return features
    paper = dict(title="Candidate method", abstract="Evidence from an experiment.")
    for method, columns, threshold, expected_logit, action in [
        ("text_only", ["lexical", "semantic"], .7, .6, "defer"),
        ("combined", ["lexical", "semantic", "original", "specific"], .8, 2.3, "prioritize"),
    ]:
        policy = dict(method=method, columns=columns, center=[0]*len(columns), scale=[1]*len(columns),
                      weights=[1]*len(columns), bias=0, threshold=threshold, target="Method relevance")
        result = reading_cli.decide("Find graph methods", "Reference title", paper, policy, extractor_factory=FakeExtractor)
        assert result["action"] == action
        assert result["score"] == pytest.approx(1/(1+math.exp(-expected_logit)))
        assert result["threshold"] == threshold and result["target"] == policy["target"]
        assert set(result["signals"]) == set(columns)
    assert initializations == [False, True]
    assert computations == [("Find graph methods", "Reference title", paper["title"], paper["abstract"])] * 2


def test_missing_abstract_asks_reader_before_loading_any_extractor(reading_cli):
    def forbidden(*args, **kwargs):
        pytest.fail("Missing abstracts must not initialize an expensive feature extractor")
    for paper in [dict(title="Title only"), dict(title="Title only", abstract=None),
                  dict(title="Title only", abstract=""), dict(title="Title only", abstract=" \n\t ")]:
        result = reading_cli.decide("Goal", "Seed", paper, {}, extractor_factory=forbidden)
        assert result["action"] == "ask_reader"
        assert result["message_vi"]
        assert "score" not in result and "signals" not in result
