import math

import pytest

from researchlens.evaluation import bootstrap_mean, ranking_metrics, summarize


def test_known_ranking_metrics():
    # Relevant hits at ranks 2 and 4; the third relevant item was not retrieved.
    result = ranking_metrics(["x", "a", "y", "b"], {"a", "b", "c"})
    assert result["precision_at_10"] == pytest.approx(2 / 10)
    assert result["recall_at_10"] == pytest.approx(2 / 3)
    assert result["map"] == pytest.approx((1 / 2 + 2 / 4) / 3)
    assert result["mrr"] == pytest.approx(1 / 2)
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3) + 1 / math.log2(4))
    assert result["ndcg_at_10"] == pytest.approx(expected)


def test_full_ranking_ap_is_not_truncated_at_ten():
    ranked = [f"n{i}" for i in range(10)] + ["relevant"]
    result = ranking_metrics(ranked, {"relevant"})
    assert result["precision_at_10"] == result["recall_at_10"] == result["ndcg_at_10"] == 0
    assert result["map"] == result["mrr"] == pytest.approx(1 / 11)


def test_perfect_ranking_empty_ranking_and_input_validation():
    result = ranking_metrics(["a", "b"], {"a", "b"})
    assert result["map"] == result["mrr"] == result["ndcg_at_10"] == result["recall_at_10"] == 1
    assert result["precision_at_10"] == 0.2
    assert all(value == 0 for value in ranking_metrics([], {"a"}).values())
    with pytest.raises(ValueError, match="relevant"):
        ranking_metrics(["a"], set())
    with pytest.raises(ValueError, match="duplicate"):
        ranking_metrics(["a", "a"], {"a"})


def test_bootstrap_is_reproducible_and_bounds_observed_values():
    values = [0.1, 0.4, 0.5, 0.8]
    first = bootstrap_mean(values, seed=42, samples=500)
    assert first == bootstrap_mean(values, seed=42, samples=500)
    assert min(values) <= first[0] <= first[1] <= max(values)
    assert bootstrap_mean([0.4] * 4) == pytest.approx([0.4, 0.4])
    with pytest.raises(ValueError, match="observations"):
        bootstrap_mean([])


def test_summary_keeps_methods_separate():
    rows = [{"method": "bm25", "map": 0.2}, {"method": "tfidf", "map": 0.9}, {"method": "bm25", "map": 0.6}]
    result = {row["method"]: row for row in summarize(rows, ["map"])}
    assert result["bm25"]["n"] == 2 and result["bm25"]["map"] == pytest.approx(0.4)
    assert result["tfidf"]["n"] == 1 and result["tfidf"]["map_ci95"] == pytest.approx([0.9, 0.9])
