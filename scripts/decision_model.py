"""Decision model: should the reader open this paper next? A gradient-boosted classifier scores each candidate
from graph and SPECTER signals. It learns from self-made tasks on graphs cut at 2010-2018 (both protocols, no
human labels) and is tested once on the frozen E2, E2b and E3 tasks against fusion. Four variants are compared on the 2018 graph
(validation); only the winner is refit on 2010-2018 and tested. Edges hidden by E3 are
removed from every training graph and label. Run embed_specter.py first."""
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from researchlens.evaluation import bootstrap_mean, ranking_metrics
from scripts.evaluate_author_next import year
from researchlens.decision import FEATURES, semantic_features, with_ranks
from scripts.evaluate_ltr import Graph, author_tasks

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CUTOFFS = range(2010, 2019)
VALIDATION_CUTOFF = 2018
WINDOW = 2
MASKS = ("a", "b", "c")


class SemanticGraph(Graph):
    """Graph signals plus the two SPECTER signals."""

    def __init__(self, papers, cutoff, table):
        super().__init__(papers, cutoff)
        self.vectors = np.stack([table[pid] for pid in self.engine.ids]).astype(np.float32)

    def candidates(self, seeds):
        ids, rows = super().candidates(seeds)
        return ids, np.column_stack([rows, semantic_features(self.engine, self.vectors, ids, seeds)])


def masked_tasks(papers, g, last, masks=None):
    """Several masks of the reference-recovery task, so one graph yields more training tasks."""
    for p in papers:
        if not g.cutoff < year(p) <= last:
            continue
        refs = sorted(set(g.engine.ids) & set(p.get("references", [])))
        if len(refs) >= 6:
            for mask in masks or MASKS:
                hidden = set(random.Random(f"{mask}:{p['id']}").sample(refs, len(refs) // 3))
                yield sorted(set(refs) - hidden), hidden


def table_for(g, tasks):
    xs, ys = [], []
    for seeds, targets in tasks:
        ids, x = g.candidates(seeds)
        xs.append(x)
        ys.append(np.array([pid in targets for pid in ids]))
    return xs, ys


VARIANTS = {
    "gbdt_raw": (lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                        min_samples_leaf=100, l2_regularization=1.0,
                                                        random_state=42), False),
    "gbdt_rank": (lambda: VARIANTS["gbdt_raw"][0](), True),
    "logistic_raw": (lambda: make_pipeline(StandardScaler(),
                                           LogisticRegression(max_iter=3000, class_weight="balanced")), False),
    "logistic_rank": (lambda: VARIANTS["logistic_raw"][0](), True),
}


def fit(variant, xs, ys):
    make, ranks = VARIANTS[variant]
    X = np.vstack([with_ranks(x) if ranks else x for x in xs])
    return make().fit(X, np.concatenate(ys))


def ndcg_on(variant, model, xs, ys):
    ranks = VARIANTS[variant][1]
    out = []
    for x, y in zip(xs, ys):
        p = model.predict_proba(with_ranks(x) if ranks else x)[:, 1]
        order = np.argsort(-p, kind="stable")[:10]
        hits = set(np.flatnonzero(y))  # a task whose answers all miss the candidate pool scores 0
        out.append(ranking_metrics(list(order), hits)["ndcg_at_10"] if hits else 0.0)
    return float(np.mean(out))


def score(g, variant, model, tasks):
    """tasks: (seeds, targets, fusion nDCG@10, paper to leave out). Returns means and the paired delta vs fusion."""
    ranks, rows = VARIANTS[variant][1], []
    for seeds, targets, fusion_ndcg, source in tasks:
        ids, x = g.candidates(seeds)
        p = model.predict_proba(with_ranks(x) if ranks else x)[:, 1]
        order = [ids[i] for i in np.argsort(-p, kind="stable") if ids[i] != source]
        top, targets = order[:10], set(targets)
        m = ranking_metrics(top, targets)
        rows.append((m["ndcg_at_10"], m["recall_at_10"], float(bool(set(top) & targets)), fusion_ndcg))
    a = np.array(rows)
    delta = list(a[:, 0] - a[:, 3])
    return dict(tasks=len(rows), ndcg=float(a[:, 0].mean()), recall=float(a[:, 1].mean()), hit=float(a[:, 2].mean()),
                fusion_ndcg=float(a[:, 3].mean()), minus_fusion=dict(mean=float(np.mean(delta)), ci95=bootstrap_mean(delta)))


def main():
    start = time.time()
    with (ROOT / "data/papers.jsonl").open() as source:
        papers = [json.loads(line) for line in source if line.strip()]
    saved = np.load(ROOT / "artifacts/specter.npz")
    table = dict(zip(saved["ids"].tolist(), saved["vectors"]))
    split = [json.loads(line) for line in (ROOT / "artifacts/recommendation_split.jsonl").open() if line.strip()]
    e3_hidden = {(t["paper_id"], r) for t in split for r in t["relevant"]}
    without_e3 = [dict(p, references=[r for r in p.get("references", []) if (p["id"], r) not in e3_hidden])
                  for p in papers]

    # Train on graphs cut at 2010-2017; the 2018 graph is the validation set that picks the variant.
    train_x, train_y, counts = [], [], {}
    for cutoff in TRAIN_CUTOFFS:
        g = SemanticGraph(without_e3, cutoff, table)
        last = min(cutoff + WINDOW, 2020)
        masks = MASKS if cutoff != VALIDATION_CUTOFF else MASKS[:1]
        tasks = list(masked_tasks(without_e3, g, last, masks)) + list(author_tasks(without_e3, g, last))
        x, y = table_for(g, tasks)
        counts[cutoff] = len(tasks)
        if cutoff == VALIDATION_CUTOFF:
            val_x, val_y = x, y
        else:
            train_x += x
            train_y += y
        print(f"cutoff {cutoff}: {len(tasks)} tasks, {time.time() - start:.0f}s", flush=True)

    validation = {}
    for variant in VARIANTS:
        validation[variant] = ndcg_on(variant, fit(variant, train_x, train_y), val_x, val_y)
        print(f"validation {variant}: nDCG@10 {validation[variant]:.4f}", flush=True)
    chosen = max(validation, key=validation.get)
    model = fit(chosen, train_x + val_x, train_y + val_y)
    print(f"chosen {chosen}, refit on 2010-2018", flush=True)

    e2 = json.loads((ROOT / "artifacts/temporal_recovery.json").read_text())["observations"]
    e2b = json.loads((ROOT / "artifacts/author_next.json").read_text())["observations"]
    fusion_e3 = {r["paper_id"]: r["fusion"] for r in json.loads((ROOT / "artifacts/holdout_fusion.json").read_text())["rows"]}
    test_2020 = SemanticGraph(papers, 2020, table)
    test_e3 = SemanticGraph(without_e3, max(year(p) for p in papers), table)
    suites = {
        "E2": (test_2020, [(t["seeds"], t["targets"], t["methods"]["fusion"]["ndcg"], None) for t in e2]),
        "E2b": (test_2020, [(t["seeds"], t["targets"], t["methods"]["fusion"]["ndcg"], None) for t in e2b]),
        "E3": (test_e3, [(t["seeds"], t["relevant"], fusion_e3[t["paper_id"]], t["paper_id"]) for t in split]),
    }
    test = {suite: score(g, chosen, model, tasks) for suite, (g, tasks) in suites.items()}
    result = dict(features=FEATURES, rank_features=VARIANTS[chosen][1], tasks_per_cutoff=counts,
                  validation_cutoff=VALIDATION_CUTOFF, validation_ndcg=validation, chosen=chosen, test=test)
    (ROOT / "artifacts/decision_model.json").write_text(json.dumps(result, indent=2))
    with (ROOT / "artifacts/decision_model.pkl").open("wb") as out:
        pickle.dump(dict(variant=chosen, model=model, features=FEATURES), out)
    for suite, r in test.items():
        print(chosen, suite, f"nDCG {r['ndcg']:.3f} fusion {r['fusion_ndcg']:.3f} Hit {r['hit']:.3f}",
              f"delta {r['minus_fusion']['mean']:+.3f}", np.round(r["minus_fusion"]["ci95"], 3), flush=True)
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
