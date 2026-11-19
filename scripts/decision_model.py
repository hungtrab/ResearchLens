"""Decision model: should the reader open this paper next? A gradient-boosted classifier scores each candidate
from graph, SPECTER and fusion signals. It learns from self-made tasks on graphs cut at 2010-2017 (both protocols,
no human labels). The 2018 graph is the validation set that picks one of four variants; the winner is refit on
2010-2018 and tested once on the frozen E2, E2b and E3 tasks against fusion. Edges hidden by E3 are removed from
every training graph and label. Run embed_specter.py first."""
import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from researchlens.evaluation import bootstrap_mean, ranking_metrics
from scripts.evaluate_author_next import year
from researchlens.decision import (FEATURES, QUERY_FEATURES, fusion_scores, graph_features, query_retrieval,
                                   semantic_features, with_ranks)
from scripts.evaluate_author_next import authors
from scripts.evaluate_ltr import Graph, author_tasks

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CUTOFFS = range(2010, 2019)
VALIDATION_CUTOFF = 2018
WINDOW = 2
MASKS = ("a", "b", "c")


class SemanticGraph(Graph):
    """Graph signals plus the two SPECTER signals and the fusion score."""

    def __init__(self, papers, cutoff, table):
        super().__init__(papers, cutoff)
        self.vectors = np.stack([table[pid] for pid in self.engine.ids]).astype(np.float32)

    def candidates(self, seeds, query=None):
        """With a topic query, BM25 adds candidates and the two query signals are appended."""
        bm25, tfidf, extra = query_retrieval(self.engine, query) if query else (None, None, ())
        ids, rows = graph_features(self.engine, self.refs, self.citers, seeds, self.cutoff, extra)
        x = np.column_stack([rows, semantic_features(self.engine, self.vectors, ids, seeds),
                             fusion_scores(self.engine, seeds, ids)])
        if query:
            at = [self.engine.pos[pid] for pid in ids]
            x = np.column_stack([x, bm25[at], tfidf[at]])
        return ids, x


def text(paper):
    return paper["title"] + " " + (paper.get("abstract") or "")


def masked_tasks(papers, g, last, masks=None, with_query=False):
    """Several masks of the reference-recovery task, so one graph yields more training tasks.
    The topic query is the citing paper's own title and abstract, which its author has while writing."""
    for p in papers:
        if not g.cutoff < year(p) <= last:
            continue
        refs = sorted(set(g.engine.ids) & set(p.get("references", [])))
        if len(refs) >= 6:
            for mask in masks or MASKS:
                hidden = set(random.Random(f"{mask}:{p['id']}").sample(refs, len(refs) // 3))
                yield sorted(set(refs) - hidden), hidden, text(p) if with_query else None


def table_for(g, tasks):
    xs, ys = [], []
    for seeds, targets, query in tasks:
        ids, x = g.candidates(seeds, query)
        xs.append(x)
        ys.append(np.array([pid in targets for pid in ids]))
    return xs, ys


BASE = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0)
WITHOUT_FUSION = FEATURES[:-1]
# Every variant also sees each signal's rank inside the task. The 2018 validation graph picks one.
VARIANTS = {
    "gbdt_rank": (BASE, WITHOUT_FUSION),
    "gbdt_rank_fusion": (BASE, FEATURES),
    "gbdt_rank_fusion_large": (dict(BASE, max_iter=600, max_leaf_nodes=63), FEATURES),
    "gbdt_rank_fusion_fast": (dict(BASE, learning_rate=0.1, min_samples_leaf=50), FEATURES),
}


QUERY_VARIANTS = {"gbdt_rank_query": (BASE, WITHOUT_FUSION + QUERY_FEATURES)}


def inputs(variant, x):
    columns = [(FEATURES + QUERY_FEATURES).index(name) for name in {**VARIANTS, **QUERY_VARIANTS}[variant][1]]
    return with_ranks(x[:, columns])


def fit(variant, xs, ys):
    model = HistGradientBoostingClassifier(**{**VARIANTS, **QUERY_VARIANTS}[variant][0], random_state=42)
    return model.fit(np.vstack([inputs(variant, x) for x in xs]), np.concatenate(ys))


def ndcg_on(variant, model, xs, ys):
    out = []
    for x, y in zip(xs, ys):
        order = np.argsort(-model.predict_proba(inputs(variant, x))[:, 1], kind="stable")[:10]
        hits = set(np.flatnonzero(y))  # a task whose answers all miss the candidate pool scores 0
        out.append(ranking_metrics(list(order), hits)["ndcg_at_10"] if hits else 0.0)
    return float(np.mean(out))


def score(g, variant, model, tasks):
    """tasks: (seeds, targets, fusion nDCG@10, paper to leave out, topic query or None).
    Returns means and the paired delta vs fusion."""
    rows = []
    for seeds, targets, fusion_ndcg, source, query in tasks:
        ids, x = g.candidates(seeds, query)
        p = model.predict_proba(inputs(variant, x))[:, 1]
        order = [ids[i] for i in np.argsort(-p, kind="stable") if ids[i] != source]
        top, targets = order[:10], set(targets)
        m = ranking_metrics(top, targets)
        bm25 = 0.0
        if query:  # the retriever alone: candidates ordered by BM25 against the topic
            column = (FEATURES + QUERY_FEATURES).index("query_bm25")
            alone = [ids[i] for i in np.argsort(-x[:, column], kind="stable") if ids[i] != source][:10]
            bm25 = ranking_metrics(alone, targets)["ndcg_at_10"]
        rows.append((m["ndcg_at_10"], m["recall_at_10"], float(bool(set(top) & targets)), fusion_ndcg, bm25,
                     float(bool(set(top[:5]) & targets))))
    a = np.array(rows)
    delta = list(a[:, 0] - a[:, 3])
    result = dict(tasks=len(rows), ndcg=float(a[:, 0].mean()), recall=float(a[:, 1].mean()), hit=float(a[:, 2].mean()),
                  hit5=float(a[:, 5].mean()),
                  fusion_ndcg=float(a[:, 3].mean()), minus_fusion=dict(mean=float(np.mean(delta)), ci95=bootstrap_mean(delta)))
    if any(t[4] for t in tasks):
        result["bm25_only_ndcg"] = float(a[:, 4].mean())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, help="refit this variant instead of the validation winner")
    parser.add_argument("--query", action="store_true", help="train the model that also reads a topic query")
    args = parser.parse_args()
    variants = QUERY_VARIANTS if args.query else VARIANTS
    name = "decision_query_model" if args.query else "decision_model"
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
        tasks = list(masked_tasks(without_e3, g, last, masks, args.query))
        tasks += [(s, t, q if args.query else None) for s, t, q in author_tasks(without_e3, g, last, with_query=True)]
        x, y = table_for(g, tasks)
        counts[cutoff] = len(tasks)
        if cutoff == VALIDATION_CUTOFF:
            val_x, val_y = x, y
        else:
            train_x += x
            train_y += y
        print(f"cutoff {cutoff}: {len(tasks)} tasks, {time.time() - start:.0f}s", flush=True)

    validation = {}
    for variant in variants:
        validation[variant] = ndcg_on(variant, fit(variant, train_x, train_y), val_x, val_y)
        print(f"validation {variant}: nDCG@10 {validation[variant]:.4f}", flush=True)
    best_on_validation = max(validation, key=validation.get)
    chosen = args.variant or best_on_validation
    model = fit(chosen, train_x + val_x, train_y + val_y)
    print(f"chosen {chosen}, refit on 2010-2018", flush=True)

    by_id = {p["id"]: p for p in papers}
    later = {}
    for p in papers:
        if year(p) > 2020:
            for author in authors(p):
                later.setdefault(author, []).append(p["title"])
    topic = (lambda q: q) if args.query else (lambda q: None)
    e2 = json.loads((ROOT / "artifacts/temporal_recovery.json").read_text())["observations"]
    e2b = json.loads((ROOT / "artifacts/author_next.json").read_text())["observations"]
    fusion_e3 = {r["paper_id"]: r["fusion"] for r in json.loads((ROOT / "artifacts/holdout_fusion.json").read_text())["rows"]}
    test_2020 = SemanticGraph(papers, 2020, table)
    test_e3 = SemanticGraph(without_e3, max(year(p) for p in papers), table)
    suites = {
        "E2": (test_2020, [(t["seeds"], t["targets"], t["methods"]["fusion"]["ndcg"], None,
                            topic(text(by_id[t["source"]]))) for t in e2]),
        "E2b": (test_2020, [(t["seeds"], t["targets"], t["methods"]["fusion"]["ndcg"], None,
                             topic(" ".join(later[t["author"]]))) for t in e2b]),
        "E3": (test_e3, [(t["seeds"], t["relevant"], fusion_e3[t["paper_id"]], t["paper_id"],
                          topic(text(by_id[t["paper_id"]]))) for t in split]),
    }
    test = {suite: score(g, chosen, model, tasks) for suite, (g, tasks) in suites.items()}
    result = dict(features=variants[chosen][1], params=variants[chosen][0], tasks_per_cutoff=counts,
                  validation_cutoff=VALIDATION_CUTOFF, validation_ndcg=validation, best_on_validation=best_on_validation,
                  chosen=chosen, test=test)
    (ROOT / f"artifacts/{name}.json").write_text(json.dumps(result, indent=2))
    with (ROOT / f"artifacts/{name}.pkl").open("wb") as out:
        pickle.dump(dict(variant=chosen, model=model, features=variants[chosen][1]), out)
    for suite, r in test.items():
        print(chosen, suite, f"nDCG {r['ndcg']:.3f} fusion {r['fusion_ndcg']:.3f} Hit {r['hit']:.3f}",
              f"delta {r['minus_fusion']['mean']:+.3f}", np.round(r["minus_fusion"]["ci95"], 3), flush=True)
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
