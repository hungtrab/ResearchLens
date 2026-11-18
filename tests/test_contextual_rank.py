"""Contextual sentence ranking and selection protocol, using offline fake states."""
import copy
import importlib
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scripts import contextual_rank as rank


def test_sentence_spans_exclude_title_and_special_tokens():
    result = list(rank.windows([10, 20], [[11, 12], [21, 22, 23]], sep=999, cls=998))
    assert result == [([998, 10, 20, 999, 11, 12, 21, 22, 23, 999], [(0, 4, 6), (1, 6, 9)])]
    ids, spans = result[0]
    assert [ids[a:b] for _, a, b in spans] == [[11, 12], [21, 22, 23]]


def test_window_boundary_fits_512_tokens_then_repeats_title_for_overflow():
    title = [70, 80]
    passages = [list(range(500)), list(range(900, 907)), [1234]]
    before = copy.deepcopy(passages)
    result = list(rank.windows(title, passages, sep=9999, cls=9998))
    assert len(result) == 2
    assert len(result[0][0]) == 512
    assert result[0][1] == [(0, 4, 504), (1, 504, 511)]
    assert result[1] == ([9998, 70, 80, 9999, 1234, 9999], [(2, 4, 5)])
    assert passages == before


def test_long_title_and_long_sentence_are_bounded_without_losing_later_sentences():
    title = list(range(200))
    passages = [list(range(1000, 1800)), [11, 12], [21]]
    result = list(rank.windows(title, passages, sep=9000, cls=9001))
    prefix = [9001, *title[:96], 9000]
    assert len(prefix) == 98
    assert len(result) == 2
    assert all(ids[:98] == prefix and len(ids) <= 512 and ids[-1] == 9000 for ids, _ in result)
    assert [j for _, spans in result for j, _, _ in spans] == [0, 1, 2]
    assert result[0][0][98:-1] == passages[0][:413]
    assert result[1][0][98:-1] == passages[1] + passages[2]


def test_empty_title_is_supported_but_empty_passage_is_rejected():
    assert list(rank.windows([], [], sep=2, cls=1)) == []
    assert list(rank.windows([], [[7]], sep=2, cls=1)) == [([1, 2, 7, 2], [(0, 2, 3)])]
    with pytest.raises(ValueError, match="empty passage"):
        list(rank.windows([3], [[4], []], sep=2, cls=1))


def test_contextual_vectors_pool_only_sentence_states_restore_order_and_keep_raw_scale(monkeypatch):
    token_rows = {
        "long-title": [101, 102], "long-first": list(range(1, 351)),
        "long-second": list(range(1001, 1201)), "long-third": list(range(2001, 2011)),
        "short-title": [201], "short-first": [11, 12], "short-second": [21],
    }
    records = {"LONG": ("long-title", ["long-first", "long-second", "long-third"]),
               "SHORT": ("short-title", ["short-first", "short-second"])}
    for i in range(8):
        token_rows[f"title-{i}"] = [300+i]
        token_rows[f"sentence-{i}"] = [30+i]
        records[f"P{i}"] = (f"title-{i}", [f"sentence-{i}"])
    loads, batch_sizes = [], []
    class Batch(dict):
        def to(self, device):
            assert device == "cpu"
            return self
    class Tokenizer:
        cls_token_id = 900001
        sep_token_id = 900002
        def __call__(self, texts, add_special_tokens, verbose):
            assert add_special_tokens is False and verbose is False
            return {"input_ids": [token_rows[text] for text in texts]}
        def pad(self, rows, padding, return_tensors):
            assert padding is True and return_tensors == "pt"
            data = np.full((len(rows), max(len(r["input_ids"]) for r in rows)), 800000)
            for i, row in enumerate(rows):
                data[i, :len(row["input_ids"])] = row["input_ids"]
            return Batch(input_ids=data)
    class Tensor:
        def __init__(self, value):
            self.value = value
        def cpu(self):
            return self
        def numpy(self):
            return self.value
    class Model:
        config = SimpleNamespace(hidden_size=2)
        def eval(self):
            return self
        def to(self, device):
            assert device == "cpu"
            return self
        def __call__(self, input_ids):
            batch_sizes.append(len(input_ids))
            positions = np.arange(input_ids.shape[1])[None, :]
            hidden = np.stack([input_ids, 2*input_ids+positions], axis=-1).astype(np.float32)
            return SimpleNamespace(last_hidden_state=Tensor(hidden))
    def tokenizer_loader(location, revision):
        loads.append(("tokenizer", location, revision))
        return Tokenizer()
    def model_loader(location, revision):
        loads.append(("model", location, revision))
        return Model()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
        AutoModel=SimpleNamespace(from_pretrained=model_loader)))
    result = rank.contextual_vectors(records, model_path=Path("/offline/model"))
    def expected(tokens, start):
        mean_token = sum(tokens)/len(tokens)
        mean_position = start+(len(tokens)-1)/2
        return [mean_token, 2*mean_token+mean_position]
    np.testing.assert_allclose(result["LONG"], [expected(token_rows["long-first"], 4),
                                              expected(token_rows["long-second"], 4),
                                              expected(token_rows["long-third"], 204)])
    np.testing.assert_allclose(result["SHORT"], [expected([11, 12], 3), expected([21], 5)])
    for i in range(8):
        np.testing.assert_allclose(result[f"P{i}"], [expected([30+i], 3)])
    assert batch_sizes == [8, 3]
    assert all(values.dtype == np.float32 for values in result.values())
    assert np.linalg.norm(result["LONG"][0]) > 100
    assert loads == [("tokenizer", "/offline/model", rank.REVISION), ("model", "/offline/model", rank.REVISION)]


def test_nearest_uses_raw_squared_l2_and_best_sentence_not_cosine_or_document_mean():
    query = np.array([10., 0.])
    candidates = [np.array([[1., 0.]]), np.array([[9., 1.]]),
                  np.array([[20., 0.], [8., 0.]]), np.array([[10., 0.], [999., 999.]])]
    scores = rank.nearest(query, candidates)
    np.testing.assert_allclose(scores, [-81, -2, -4, 0])
    assert np.argsort(-scores).tolist() == [3, 1, 2, 0]
    # Normalizing the vectors would wrongly favor candidate 0 over candidate 1.
    np.testing.assert_allclose(rank.nearest(3*query, [3*m for m in candidates]), 9*scores)
    assert rank.nearest(query, []).size == 0


@pytest.mark.parametrize("candidate", [np.empty((0, 2)), np.zeros((1, 3))])
def test_nearest_rejects_missing_or_incompatible_sentence_vectors(candidate):
    with pytest.raises(ValueError):
        rank.nearest(np.array([1., 0.]), [candidate])


def test_reciprocal_rank_uses_one_based_ranks_and_stable_ties():
    scores = np.array([.2, .9, .9, -3.])
    np.testing.assert_allclose(rank.reciprocal(scores), [1/63, 1/61, 1/62, 1/64])
    np.testing.assert_allclose(rank.reciprocal(np.ones(3)), [1/61, 1/62, 1/63])
    np.testing.assert_allclose(rank.reciprocal(100*scores+7), rank.reciprocal(scores))
    assert rank.reciprocal(np.array([])).size == 0


@pytest.mark.parametrize("weight", [0, .25, .5, .75, 1])
def test_fusion_combines_rank_reciprocals_not_incompatible_raw_score_scales(weight):
    baseline = np.array([1000., 30., -9000.])
    contextual = np.array([-100., -20., -1.])
    expected = (1-weight)*np.array([1/61, 1/62, 1/63]) + weight*np.array([1/63, 1/62, 1/61])
    actual = rank.fused_scores(baseline, contextual, weight)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(rank.fused_scores(baseline*5+12, contextual*7-200, weight), actual)
    if weight == .5:
        assert actual[0] == actual[2]
        assert np.argsort(-actual, kind="stable")[0] == 0


@pytest.fixture
def evaluation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(rank.__file__).parent))
    return importlib.import_module("evaluate_contextual")


def test_transfer_topics_exclude_previously_seen_query_papers_across_all_facets(evaluation):
    splits = {
        "method": {"fold1_dev": ["A_method"], "fold1_test": ["B_method"]},
        "background": {"fold1_dev": ["A_background", "B_background", "C_background", "D_background"], "fold1_test": ["E_background"]},
        "result": {"fold1_dev": ["A_result", "C_result", "F_result"], "fold1_test": ["B_result", "G_result", "H_result"]},
    }
    previous = {"queries": {"method": [{"query": "B"}], "result": [{"query": "G"}]}}
    groups = evaluation.topics(splits, previous)
    assert groups["dev"] == ["A_method"]
    assert groups["method"] == ["B_method"] and groups["result"] == ["G_result"]
    assert set(groups["transfer"]) == {"C_background", "D_background", "E_background", "C_result", "F_result", "H_result"}
    seen = {key.split("_")[0] for name in ["dev", "method", "result"] for key in groups[name]}
    assert seen.isdisjoint(key.split("_")[0] for key in groups["transfer"])
    # Both new tasks from paper C remain eligible and must be one CI cluster.
    assert sum(key.startswith("C_") for key in groups["transfer"]) == 2


def test_metrics_use_grade_two_cutoff_graded_gain_and_stable_top_three(evaluation):
    result = evaluation.metrics(np.array([3, 0, 2, 1]), np.array([.9, .8, .7, .6]))
    expected_ndcg = (7+3/2+1/math.log2(5))/(7+3/math.log2(3)+1/2)
    assert result["hits"] == 2
    assert result["ndcg"] == pytest.approx(expected_ndcg)
    assert evaluation.metrics(np.array([0, 0, 0, 3]), np.ones(4))["hits"] == 0


def paired_rows():
    return [
        dict(query="A", facet="background", pool=[1, 2, 3], arms={"0": dict(hits=0, ndcg=.1), "1": dict(hits=3, ndcg=.7)}),
        dict(query="A", facet="result", pool=[1, 2, 3], arms={"0": dict(hits=3, ndcg=.8), "1": dict(hits=0, ndcg=.4)}),
        dict(query="B", facet="background", pool=[1, 2, 3, 4], arms={"0": dict(hits=0, ndcg=.2), "1": dict(hits=3, ndcg=.6)}),
    ]


def test_summary_counts_topics_pairs_and_slots_without_collapsing_facets(evaluation):
    result = evaluation.summary(paired_rows(), "1")
    assert result == pytest.approx(dict(queries=3, pairs=10, slots=9, hits=6, ndcg=1.7/3))


def test_paired_bootstrap_resamples_whole_query_paper_clusters(evaluation):
    rows = paired_rows()
    result = evaluation.paired(rows, "1")
    # Two draws of whole papers yield only AA, AB or BB. The contradictory
    # facets of A must travel together, so P@3 CI cannot extend below zero.
    assert result["clusters"] == 2
    assert result["p3_delta_ci"] == pytest.approx([0, 1])
    assert result["ndcg_delta_ci"] == pytest.approx([.1, .4])
    assert evaluation.paired(rows, "1") == result
    assert evaluation.paired(rows, "0")["p3_delta_ci"] == [0, 0]
    single = evaluation.paired(rows[:2], "1")
    assert single["clusters"] == 1
    assert single["p3_delta_ci"] == [0, 0]
    assert single["ndcg_delta_ci"] == pytest.approx([.1, .1])


@pytest.mark.parametrize("case,expected_weight", [("hits_before_ndcg", .25), ("ndcg_then_lower_weight", .5)])
def test_model_selection_uses_only_development_metrics_and_prespecified_tie_breaks(evaluation, tmp_path, monkeypatch, case, expected_weight):
    pids = ["QD", "QM", "QR", "TB", "TR", "A", "B", "C", "D", "E"]
    papers = [dict(paper_id=pid, title="Research paper " + pid,
                   abstract=["Method evidence is described.", "Result evidence is measured.", "Background evidence is reviewed."],
                   pred_labels=["method_label", "result_label", "background_label"]) for pid in pids]
    splits = {
        "method": {"fold1_dev": ["QD_method"], "fold1_test": ["QM_method"]},
        "background": {"fold1_dev": ["TB_background"], "fold1_test": []},
        "result": {"fold1_dev": ["TR_result"], "fold1_test": ["QR_result"]},
    }
    qrels = {facet: {q: dict(cands=["A", "B", "C", "D", "E"], relevance_adju=[3, 2, 2, 0, 0]) for q in queries}
             for facet, queries in [("method", ["QD", "QM"]), ("result", ["QR", "TR"]), ("background", ["TB"])]}
    (tmp_path / "passage_comparison.json").write_text(json.dumps({"queries": {"method": [{"query": "QM"}], "result": [{"query": "QR"}]}}))
    (tmp_path / "context_protocol.md").write_text("Frozen test protocol")
    monkeypatch.setattr(evaluation, "DEST", tmp_path)
    def source(cache, name):
        if name == "abstracts-csfcube-preds.jsonl":
            return "\n".join(json.dumps(p) for p in papers).encode()
        if name == "evaluation_splits.json":
            return json.dumps(splits).encode()
        facet = name.removeprefix("test-pid2anns-csfcube-").removesuffix(".json")
        return json.dumps(qrels[facet]).encode()
    monkeypatch.setattr(evaluation, "source", source)
    monkeypatch.setattr(evaluation, "contextual_vectors", lambda records, model_path: {pid: np.zeros((len(parts), 2)) for pid, (_, parts) in records.items()})
    class Lexical:
        def transform(self, texts):
            return csr_matrix(np.ones((len(texts), 1)))
    def extractor(include_laya):
        assert include_laya is False
        return SimpleNamespace(lex=Lexical())
    monkeypatch.setattr(evaluation, "DecisionFeatures", extractor)
    monkeypatch.setattr(evaluation, "encode", lambda encoder, texts: np.ones((len(texts), 2)))
    monkeypatch.setattr(evaluation, "load_policy", lambda: ({}, "fixture-policy-sha"))
    monkeypatch.setattr(evaluation, "score", lambda rows, policy: np.zeros(len(rows)))
    orders = {0: [0, 3, 4, 1, 2], .25: [1, 2, 0, 3, 4], .5: [1, 2, 0, 3, 4], .75: [1, 2, 0, 3, 4], 1: [0, 1, 3, 4, 2]}
    if case == "ndcg_then_lower_weight":
        orders[.5] = orders[.75] = [0, 1, 2, 3, 4]
    def fused(baseline, contextual, weight):
        result = np.empty(5)
        result[orders[weight]] = np.arange(5, 0, -1)
        return result
    monkeypatch.setattr(evaluation, "fused_scores", fused)
    before = evaluation.run(tmp_path / "cache", model_path=None)
    assert before["selected_weight"] == expected_weight
    assert before["dev_grid"][1]["ndcg"] > before["dev_grid"][.25]["ndcg"]
    for facet in qrels:
        for q, judgments in qrels[facet].items():
            if q != "QD":
                judgments["relevance_adju"] = [0, 0, 0, 3, 3]
    after = evaluation.run(tmp_path / "cache", model_path=None)
    assert after["selected_weight"] == before["selected_weight"]
    assert after["dev_grid"] == before["dev_grid"]
    assert after["summary"]["method"]["selected"]["hits"] != before["summary"]["method"]["selected"]["hits"]
