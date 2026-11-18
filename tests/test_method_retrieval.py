"""Offline, hand-checkable regressions for the method retrieval evaluator."""
import importlib
import io
import json
import math
import tarfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def retrieval(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("evaluate_method_retrieval")


def test_bm25_phrase_arm_distinguishes_word_order_while_unigrams_tie(retrieval):
    documents = ["graph neural networks", "neural graph networks", "quantum field theory"]
    queries = ["graph neural", "neural graph", "qzxunknown"]
    unigram, phrase = retrieval.bm25_scores(documents, queries)
    assert unigram.shape == phrase.shape == (3, 3)
    assert np.issubdtype(unigram.dtype, np.floating) and np.issubdtype(phrase.dtype, np.floating)
    assert np.isfinite(unigram).all() and np.isfinite(phrase).all()
    assert unigram[0, 0] == pytest.approx(unigram[0, 1])
    np.testing.assert_allclose(unigram[0], unigram[1])
    assert unigram[0, 0] > unigram[0, 2]
    assert phrase[0, 0] > phrase[0, 1] > phrase[0, 2]
    assert phrase[1, 1] > phrase[1, 0] > phrase[1, 2]
    np.testing.assert_array_equal(unigram[2], np.zeros(3))
    np.testing.assert_array_equal(phrase[2], np.zeros(3))


def test_bm25_term_frequency_and_document_length_affect_ranking(retrieval):
    unigram, phrase = retrieval.bm25_scores(
        ["alpha alpha beta", "alpha beta beta", "gamma delta epsilon"], ["alpha", "beta"])
    assert unigram[0, 0] > unigram[0, 1] > unigram[0, 2]
    assert unigram[1, 1] > unigram[1, 0] > unigram[1, 2]
    assert unigram[0, 0] == pytest.approx(unigram[1, 1])
    assert unigram[0, 1] == pytest.approx(unigram[1, 0])
    # Three equally long documents, df(alpha)=2, k1=1.2: tf=2 then tf=1.
    idf = math.log(1 + 1.5/2.5)
    np.testing.assert_allclose(unigram[0], [idf*2*2.2/3.2, idf, 0])
    for scores in [unigram, phrase]:
        assert scores[0, 0] > scores[0, 1]
    short, long = "graph methods", "graph " + "filler " * 100
    for scores in retrieval.bm25_scores([short, long, "methods other"], ["graph"]):
        assert scores[0, 0] > scores[0, 1] > scores[0, 2]


def test_bm25_queries_are_independent_of_batch_order_and_preserve_document_columns(retrieval):
    docs = ["graph retrieval", "neural methods", "graph neural methods"]
    queries = ["graph", "neural methods", ""]
    original = retrieval.bm25_scores(docs, queries)
    reversed_docs = retrieval.bm25_scores(list(reversed(docs)), queries)
    reversed_queries = retrieval.bm25_scores(docs, list(reversed(queries)))
    for arm in range(2):
        np.testing.assert_allclose(original[arm], reversed_docs[arm][:, ::-1])
        np.testing.assert_allclose(original[arm], reversed_queries[arm][::-1])
        for i, query in enumerate(queries):
            np.testing.assert_allclose(original[arm][i], retrieval.bm25_scores(docs, [query])[arm][0])
        np.testing.assert_array_equal(original[arm][2], np.zeros(3))


@pytest.mark.parametrize("documents,queries,shape", [
    ([], [], (0, 0)), ([], ["graph", "methods"], (2, 0)),
    (["graph", "methods"], [], (0, 2)),
    (["", "   ", "\n"], ["graph", ""], (2, 3)),
])
def test_bm25_empty_inputs_keep_query_by_document_shape(retrieval, documents, queries, shape):
    arrays = retrieval.bm25_scores(documents, queries)
    assert len(arrays) == 2
    for values in arrays:
        assert isinstance(values, np.ndarray) and values.shape == shape
        np.testing.assert_array_equal(values, np.zeros(shape))


def test_fusion_sums_reciprocals_of_average_tie_ranks(retrieval):
    reader = np.array([5., 5., 1., 0.])
    phrase = np.array([0., 2., 2., 4.])
    before_reader, before_phrase = reader.copy(), phrase.copy()
    expected = np.array([1/61.5+1/64, 1/61.5+1/62.5, 1/63+1/62.5, 1/64+1/61])
    np.testing.assert_allclose(retrieval.fusion_scores(reader, phrase), expected)
    np.testing.assert_array_equal(reader, before_reader)
    np.testing.assert_array_equal(phrase, before_phrase)
    # Unlike stable ordinal ties, true average ranks cannot privilege input order.
    np.testing.assert_allclose(retrieval.fusion_scores(np.ones(4), np.ones(4)), np.full(4, 2/62.5))


def test_fusion_is_symmetric_monotone_scale_invariant_and_permutation_equivariant(retrieval):
    reader = np.array([9., 1., 9., -2.])
    phrase = np.array([2., 2., 1., 0.])
    result = retrieval.fusion_scores(reader, phrase)
    np.testing.assert_allclose(result, retrieval.fusion_scores(phrase, reader))
    np.testing.assert_allclose(result, retrieval.fusion_scores(3*reader+17, 20*phrase-9))
    permutation = np.array([3, 2, 0, 1])
    np.testing.assert_allclose(retrieval.fusion_scores(reader[permutation], phrase[permutation]), result[permutation])
    assert retrieval.fusion_scores(np.array([]), np.array([])).shape == (0,)
    np.testing.assert_allclose(retrieval.fusion_scores(np.array([4.]), np.array([-2.])), [2/61])


def test_fusion_rejects_mismatched_candidate_counts_instead_of_broadcasting(retrieval):
    with pytest.raises(ValueError):
        retrieval.fusion_scores(np.ones(1), np.ones(3))


def test_rerank_reorders_only_scored_prefix_and_preserves_stable_ties(retrieval):
    order = np.array([7, 3, 5, 1, 9])
    logits = np.array([.1, .9, .9])
    before_order, before_logits = order.copy(), logits.copy()
    result = retrieval.rerank_order(order, logits)
    np.testing.assert_array_equal(result, [3, 5, 7, 1, 9])
    np.testing.assert_array_equal(result[len(logits):], order[len(logits):])
    assert sorted(result.tolist()) == sorted(order.tolist())
    np.testing.assert_array_equal(order, before_order)
    np.testing.assert_array_equal(logits, before_logits)


@pytest.mark.parametrize("order,logits,expected", [
    ([], [], []), ([2, 0, 1], [], [2, 0, 1]),
    ([2, 0, 1], [100], [2, 0, 1]),
    ([2, 0, 1], [0, 0, 0], [2, 0, 1]),
    ([2, 0, 1], [-4, -1, -2], [0, 1, 2]),
])
def test_rerank_empty_partial_and_full_prefix_edges(retrieval, order, logits, expected):
    result = retrieval.rerank_order(np.array(order, dtype=int), np.array(logits, dtype=float))
    np.testing.assert_array_equal(result, expected)


def test_rerank_rejects_more_logits_than_candidate_indices(retrieval):
    with pytest.raises((ValueError, IndexError)):
        retrieval.rerank_order(np.array([2, 0]), np.array([.1, .2, .3]))


def test_metrics_distinguishes_any_hit_from_relevant_slot_count(retrieval):
    result = retrieval.metrics(np.array([3, 2, 2, 0]), positive=2)
    assert result["hit3"] == result["hit20"] == 1
    assert result["slots3"] == 3 and result["first"] == 1
    assert result["ndcg10"] == pytest.approx(1)
    stricter = retrieval.metrics(np.array([3, 2, 2, 0]), positive=3)
    assert stricter["slots3"] == 1 and stricter["first"] == 1
    assert stricter["ndcg10"] == result["ndcg10"]


def test_metrics_uses_exponential_graded_gains_and_one_based_first_hit(retrieval):
    result = retrieval.metrics(np.array([0, 1, 2, 3]), positive=2)
    expected_dcg = 1/math.log2(3) + 3/math.log2(4) + 7/math.log2(5)
    expected_ideal = 7 + 3/math.log2(3) + 1/math.log2(4)
    assert result["hit3"] == 1 and result["slots3"] == 1
    assert result["hit20"] == 1 and result["first"] == 3
    assert result["ndcg10"] == pytest.approx(expected_dcg/expected_ideal)


@pytest.mark.parametrize("position,hit3,hit20", [(3, 1, 1), (4, 0, 1), (20, 0, 1), (21, 0, 0)])
def test_metrics_cutoffs_are_inclusive_and_first_hit_is_full_rank(retrieval, position, hit3, hit20):
    grades = np.zeros(25, dtype=int)
    grades[position-1] = 2
    result = retrieval.metrics(grades, positive=2)
    assert result["hit3"] == hit3 and result["slots3"] == hit3
    assert result["hit20"] == hit20 and result["first"] == position
    assert result["ndcg10"] == pytest.approx(1/math.log2(position+1) if position <= 10 else 0)


@pytest.mark.parametrize("grades,ndcg", [([], 0), ([0, 0, 0], 0), ([1, 1, 1], 1)])
def test_metrics_empty_or_no_positive_hit_has_none_first(retrieval, grades, ndcg):
    result = retrieval.metrics(np.array(grades, dtype=int), positive=2)
    assert result["hit3"] == result["slots3"] == result["hit20"] == 0
    assert result["first"] is None
    assert result["ndcg10"] == pytest.approx(ndcg)


@pytest.fixture
def dataset_archive(retrieval, monkeypatch):
    """Supply the real adapters with tiny tar members, never a download."""
    datasets = importlib.import_module("method_datasets")

    def install(files):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for name, contents in files.items():
                data = contents.encode("utf-8")
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        payload = buffer.getvalue()
        monkeypatch.setattr(datasets, "archive", lambda *_: tarfile.open(fileobj=io.BytesIO(payload)))
        return datasets

    return install


def arc_row(citation_id, paper_id, title):
    return dict(citation_id=citation_id, cited_paper_id=paper_id,
                cited_paper_title=title, cited_paper_year=2018,
                citing_paper_id="source", citing_paper_title="Source paper",
                citing_paper_year=2020, intent="Uses",
                cleaned_cite_text="We apply the algorithm @@CITATION to tagging.")


def arc_members(train, test):
    return {f"acl-arc/{split}.jsonl": "".join(json.dumps(row) + "\n" for row in rows)
            for split, rows in [("train", train), ("dev", []), ("test", test)]}


def test_acl_arc_identical_citing_and_target_title_never_enters_query(dataset_archive, tmp_path):
    row = arc_row("query", "target", "Secret Target Method")
    # Real source defect: the citing title can be copied from the cited title.
    row["citing_paper_title"] = row["cited_paper_title"]
    datasets = dataset_archive(arc_members([], [row]))
    docs, queries, audit = datasets.acl_arc(tmp_path)
    assert docs == {"target": "Secret Target Method"}
    assert len(queries) == 1
    assert queries[0].text == "We apply the algorithm   to tagging."
    assert "Secret Target Method" not in queries[0].text
    assert queries[0].grades == {"target": 1}
    assert audit["target_title_inside_query"] == 0


def test_acl_arc_connected_id_title_aliases_collapse_and_exclude_self_and_future(dataset_archive, tmp_path):
    # A -- title one -- B -- title two -- C is one paper, not two title groups.
    train = [arc_row("a", "A", "Method One"), arc_row("b", "B", "Method One"),
             arc_row("c", "B", "Method Two"), arc_row("d", "C", "Method Two"),
             arc_row("s1", "S1", "Source One"), arc_row("s2", "S2", "Source One"),
             arc_row("s3", "S2", "Source Two"), arc_row("s4", "S3", "Source Two"),
             arc_row("f1", "F1", "Future One"), arc_row("f2", "F2", "Future One"),
             arc_row("f3", "F2", "Future Two"), arc_row("f4", "F3", "Future Two")]
    for row in train:
        if row["cited_paper_id"].startswith("F"):
            row["cited_paper_year"] = 2024
    valid = arc_row("valid", "C", "Method Two")
    self_query = arc_row("self", "S2", "Source Two")
    future_query = arc_row("future", "F3", "Future Two")
    future_query["cited_paper_year"] = 2024
    for row in [valid, self_query, future_query]:
        row["citing_paper_id"] = "S3"
    datasets = dataset_archive(arc_members(train, [valid, self_query, future_query]))
    docs, queries, audit = datasets.acl_arc(tmp_path)
    assert set(docs) == {"A", "S1", "F1"}
    assert [q.key for q in queries] == ["valid"]
    assert queries[0].grades == {"A": 1}
    assert queries[0].excluded == {"S1", "F1"}
    assert audit["ineligible"] == 2
    assert audit["catalog_papers"] == 3


def test_scierc_masking_removes_whole_method_only_and_preserves_unrelated_words(dataset_archive, tmp_path):
    texts = {"a": "CRF improves CRFs, microCRF and crf baselines.",
             "b": "CRF supports tagging."}
    files = {}
    for key, text in texts.items():
        files[f"raw/{key}.txt"] = text
        files[f"raw/{key}.ann"] = "T1\tMethod 0 3\tCRF\n"
        files[f"raw/{key}.txt.xml"] = (
            "<root><document><sentences><sentence><tokens><token>"
            "<CharacterOffsetBegin>0</CharacterOffsetBegin>"
            f"<CharacterOffsetEnd>{len(text)}</CharacterOffsetEnd>"
            "</token></tokens></sentence></sentences></document></root>")
    datasets = dataset_archive(files)
    docs, visible, masked, audit = datasets.scierc(tmp_path)
    assert docs == texts
    assert {q.key: q.text for q in visible} == texts
    hidden = {q.key: q for q in masked}
    assert hidden["a"].text == "the technique improves CRFs, microCRF and the technique baselines."
    assert hidden["b"].text == "the technique supports tagging."
    assert hidden["a"].grades == {"b": 1} and hidden["a"].excluded == {"a"}
    assert hidden["b"].grades == {"a": 1} and hidden["b"].excluded == {"b"}
    assert all(q.cluster == "crf" for q in masked)
    assert audit["eligible"] == 2


@pytest.mark.parametrize("threshold,expected_slots", [(1, 3), (2, 2), (3, 1), (4, 0)])
def test_laya_secondary_scores_only_shared_shortlist_with_stable_ties_and_grade_threshold(
        retrieval, threshold, expected_slots):
    decision = importlib.import_module("evaluate_method_decision")
    shortlist = ["weak", "strong", "none", "good"]
    docs = {key: f"Evidence for {key}" for key in shortlist + ["outside"]}
    grades = {"weak": 1, "strong": 2, "none": 0, "good": 3, "outside": 3}
    goal = "Find a concrete tagging method."
    rows = [{"query": "q", "cluster": "paper", "fusion_top20": shortlist.copy(),
             "arms": {"fusion": {"hit3": 0}}}]
    scores = {docs[k]: v for k, v in zip(shortlist, [.8, .8, .1, .9])}

    class FakeModel:
        def __init__(self):
            self.calls = []

        def predict(self, state, questions):
            state = json.loads(state)
            assert set(state) == {"goal", "candidate"}  # no judgment leakage
            assert state["goal"] == goal
            assert questions == {"match": decision.QUESTION}
            self.calls.append(state["candidate"])
            return {"answers": {"match": {"noul": scores[state["candidate"]]}}}

    model = FakeModel()
    result = decision.score_queries(model, rows, {"q": (goal, docs, grades)}, threshold)
    assert model.calls == [docs[k] for k in shortlist]
    assert result == [{"query": "q", "cluster": "paper", "scores": [.8, .8, .1, .9],
                       "top3": ["good", "weak", "strong"], "slots3": expected_slots,
                       "hit3": int(expected_slots > 0), "fusion_hit3": 0}]
    assert rows[0]["fusion_top20"] == shortlist
