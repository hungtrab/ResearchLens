"""Fit a small typed decision head on CSFCube development queries, then evaluate once."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_curve,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ndcg_score,
)
from sklearn.preprocessing import StandardScaler

STUDY = Path(__file__).resolve().parents[1] / "evaluation/decision-study"
ARMS = {
    "lexical": ["lexical"],
    "semantic": ["semantic"],
    "text_only": ["lexical", "semantic"],
    "laya_calibrated": ["original"],
    "laya_evidence": ["original", "specific"],
    "combined": ["lexical", "semantic", "original", "specific"],
}


def matrix(rows, columns):
    return np.array([[r[c] for c in columns] for r in rows])


def fit(rows, columns):
    x = matrix(rows, columns)
    scale = StandardScaler().fit(x)
    y = [r["grade"] >= 2 for r in rows]
    clf = LogisticRegression(C=1, max_iter=500).fit(scale.transform(x), y)
    return dict(
        columns=columns,
        center=scale.mean_.tolist(),
        scale=scale.scale_.tolist(),
        weights=clf.coef_[0].tolist(),
        bias=clf.intercept_[0],
    )


def score(rows, head):
    standardized = (matrix(rows, head["columns"]) - head["center"]) / head["scale"]
    return expit(standardized @ head["weights"] + head["bias"])


def disjoint_train(rows, held):
    forbidden = {r[k] for r in held for k in ("pid", "group")}
    return [
        r for r in rows if r["pid"] not in forbidden and r["group"] not in forbidden
    ]


def choose_threshold(dev, columns):
    # Each development query gets predictions from a different fitted model.
    y, scores = [], []
    for query in sorted({r["group"] for r in dev}):
        held = [r for r in dev if r["group"] == query]
        inner = disjoint_train(dev, held)
        scores.extend(score(held, fit(inner, columns)))
        y.extend(r["grade"] >= 2 for r in held)
    return balanced_threshold(y, scores)


def balanced_threshold(labels, scores):
    fp, tp, cutoffs = roc_curve(labels, scores, drop_intermediate=False)
    # Youden's J maximizes balanced accuracy. Prefer the higher finite cutoff on ties.
    finite = np.isfinite(cutoffs)
    return float(cutoffs[finite][np.argmax(np.round((tp - fp)[finite], 12))])


def calibrate_pilot(rows, scores):
    # Transfer the score with fixed weights. Only the action cutoff is local.
    decisions = []
    for group in sorted({r["group"] for r in rows}):
        target = [r for r in rows if r["group"] == group]
        excluded = {r["pid"] for r in target}
        train = [
            i
            for i, r in enumerate(rows)
            if r["group"] != group and r["pid"] not in excluded
        ]
        cutoff = balanced_threshold(
            [rows[i]["grade"] >= 2 for i in train], [scores[i] for i in train]
        )
        for i, r in enumerate(rows):
            if r["group"] == group:
                decisions.append(
                    dict(
                        group=group,
                        pid=r["pid"],
                        cutoff=cutoff,
                        prediction=bool(scores[i] >= cutoff),
                        label=r["grade"] >= 2,
                    )
                )
    actual = [d["label"] for d in decisions]
    calls = [d["prediction"] for d in decisions]
    return dict(
        correct=sum(a == b for a, b in zip(actual, calls)),
        n=len(calls),
        balanced_accuracy=balanced_accuracy_score(actual, calls),
        decisions=decisions,
    )


def compare_queries(a, b):
    deltas = np.array([a[q] - b[q] for q in sorted(a)])
    boot = np.random.default_rng(2026).choice(deltas, (10000, len(deltas))).mean(axis=1)
    return dict(
        delta=float(deltas.mean()),
        above=int((deltas > 0).sum()),
        queries=len(deltas),
        bootstrap_95=np.quantile(boot, [0.025, 0.975]).tolist(),
    )


def measure(rows, scores, threshold):
    scores = np.asarray(scores)
    labels = np.array([r["grade"] >= 2 for r in rows])
    calls = scores >= threshold
    by_query = {}
    for query in sorted({r["group"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["group"] == query]
        gains = [[2 ** rows[i]["grade"] - 1 for i in indices]]
        by_query[query] = float(ndcg_score(gains, [scores[indices]], k=10))
    return dict(
        n=len(rows),
        positive=int(labels.sum()),
        correct=int((calls == labels).sum()),
        balanced_accuracy=balanced_accuracy_score(labels, calls),
        f1=f1_score(labels, calls, zero_division=0),
        precision=precision_score(labels, calls, zero_division=0),
        recall=recall_score(labels, calls, zero_division=0),
        average_precision=average_precision_score(labels, scores),
        brier=brier_score_loss(labels, scores),
        confusion_tn_fp_fn_tp=confusion_matrix(labels, calls, labels=[False, True])
        .ravel()
        .tolist(),
        macro_ndcg_10=float(np.mean(list(by_query.values()))),
        ndcg_by_query=by_query,
    )


def main():
    source = json.loads((STUDY / "feature_provenance.json").read_text())
    feature_file = STUDY / "features.jsonl"
    if (
        hashlib.sha256(feature_file.read_bytes()).hexdigest()
        != source["features_sha256"]
    ):
        raise ValueError("Features differ from the recorded extraction.")
    rows = [json.loads(s) for s in feature_file.read_text().splitlines()]
    test = [r for r in rows if r["split"] == "test"]
    dev_raw = [r for r in rows if r["split"] == "dev"]
    assert not {r["group"] for r in test} & {r["group"] for r in dev_raw}
    dev = disjoint_train(dev_raw, test)
    pilot = [r for r in rows if r["split"] == "pilot" and r["grade"] is not None]
    split_info = dict(
        dev_queries=len({r["group"] for r in dev}),
        dev_before_filter=len(dev_raw),
        dev_pairs=len(dev),
        test_queries=len({r["group"] for r in test}),
        test_pairs=len(test),
    )
    report = dict(split=split_info, models={}, predictions={})
    report["always_defer"] = {
        name: measure(subset, np.zeros(len(subset)), 0.5)
        for name, subset in [("external_test", test), ("local_pilot", pilot)]
    }
    for method, columns in {"original_raw": None, **ARMS}.items():
        head = None if columns is None else fit(dev, columns)
        threshold = 0.5 if head is None else choose_threshold(dev, columns)
        stats = {}
        for name, subset in [("external_test", test), ("local_pilot", pilot)]:
            predictions = (
                [r["original"] for r in subset]
                if head is None
                else score(subset, head).tolist()
            )
            stats[name] = measure(subset, predictions, threshold)
            report["predictions"].setdefault(name, {})[method] = predictions
        report["models"][method] = dict(head=head, threshold=threshold, **stats)
        report.setdefault("exploratory_pilot_local_cutoffs", {})[method] = (
            calibrate_pilot(pilot, report["predictions"]["local_pilot"][method])
        )
    # This recipe is fixed before inspecting test results; keep ablations visible.
    for method, filename in [
        ("combined", "reading_policy.json"),
        ("text_only", "text_policy.json"),
    ]:
        model = report["models"][method]
        policy = dict(
            version=1,
            method=method,
            **model["head"],
            threshold=model["threshold"],
            target="CSFCube method relevance grade >= 2",
            source=source,
        )
        (STUDY / filename).write_text(json.dumps(policy, indent=2))
    specter = json.loads((STUDY / "specter_method_ranked.json").read_text())
    ranked = {}
    for q in sorted({r["group"] for r in test}):
        candidates = {pid: -dist for pid, dist in specter[q]}
        group = [r for r in test if r["group"] == q]
        assert set(candidates) == {r["pid"] for r in group}
        ranked[q] = float(
            ndcg_score(
                [[2 ** r["grade"] - 1 for r in group]],
                [[candidates[r["pid"]] for r in group]],
                k=10,
            )
        )
    report["specter_reference"] = dict(
        macro_ndcg_10=float(np.mean(list(ranked.values()))), ndcg_by_query=ranked
    )
    combined_queries = report["models"]["combined"]["external_test"]["ndcg_by_query"]
    references = {
        m: report["models"][m]["external_test"]["ndcg_by_query"]
        for m in ["original_raw", "text_only"]
    }
    report["paired_ndcg"] = {
        m: compare_queries(combined_queries, v)
        for m, v in {**references, "specter": ranked}.items()
    }
    report["prediction_order"] = {
        name: [(r["group"], r["pid"]) for r in subset]
        for name, subset in [("external_test", test), ("local_pilot", pilot)]
    }
    (STUDY / "decision_comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    print(split_info)
    for m, c in report["models"].items():
        a, b = c["external_test"], c["local_pilot"]
        print(
            m,
            "expert nDCG/AP/bal.acc",
            *(
                round(a[k], 3)
                for k in ("macro_ndcg_10", "average_precision", "balanced_accuracy")
            ),
            "local correct",
            b["correct"],
            "/",
            b["n"],
        )


if __name__ == "__main__":
    main()
