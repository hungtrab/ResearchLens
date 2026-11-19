"""Learning-to-rank check. A logistic regression over eight named signals is trained on a 2018 graph
(tasks from 2019-2020 papers) and tested on the frozen tasks of E2, E2b and the E3 random holdout.
Nothing is tuned on the test tasks."""
import json
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from researchlens.decision import GRAPH_FEATURES as FEATURES, citation_links, graph_features
from researchlens.evaluation import bootstrap_mean, ranking_metrics
from researchlens.mining import MiningEngine
from scripts.evaluate_author_next import authors, year

ROOT = Path(__file__).resolve().parents[1]
class Graph:
    """A corpus cut at one year, with the citation links that the decision signals need."""

    def __init__(self, papers, cutoff):
        self.cutoff = cutoff
        self.engine = MiningEngine([p for p in papers if 0 < year(p) <= cutoff])
        self.refs, self.citers = citation_links(self.engine)

    def candidates(self, seeds):
        return graph_features(self.engine, self.refs, self.citers, seeds, self.cutoff)


def reference_tasks(papers, g, last):
    """E2 protocol: hide a third of a later paper's references."""
    for p in sorted(papers, key=lambda p: p["id"]):
        if not g.cutoff < year(p) <= last:
            continue
        refs = sorted(set(g.engine.ids) & set(p.get("references", [])))
        if len(refs) >= 6:
            hidden = set(random.Random(f"42:{p['id']}").sample(refs, len(refs) // 3))
            yield sorted(set(refs) - hidden), hidden


def author_tasks(papers, g, last, with_query=False):
    """E2b protocol: an author's library before the cutoff, first-time citations after it.
    with_query also yields the titles of the author's new papers, the topic they are working on."""
    by_author = {}
    for p in papers:
        for name in authors(p):
            by_author.setdefault(name, []).append(p)
    ids = set(g.engine.ids)
    for _, written in sorted(by_author.items()):
        before = [p for p in written if 0 < year(p) <= g.cutoff]
        after = [p for p in written if g.cutoff < year(p) <= last]
        if not before or not after:
            continue
        library = ({p["id"] for p in before} | {r for p in before for r in p.get("references", [])}) & ids
        targets = ({r for p in after for r in p.get("references", [])} & ids) - library - {p["id"] for p in written}
        if len(library) >= 3 and targets:
            yield (sorted(library), targets, " ".join(p["title"] for p in after)) if with_query else (sorted(library), targets)


def fit(g, tasks):
    xs, ys = [], []
    for seeds, targets in tasks:
        ids, x = g.candidates(seeds)
        xs.append(x)
        ys.append([pid in targets for pid in ids])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    return model.fit(np.vstack(xs), np.concatenate(ys)), len(xs)


def test(g, model, tasks):
    """tasks: (seeds, targets, fusion nDCG@10 on the same task, paper to leave out)."""
    rows = []
    for seeds, targets, fusion_ndcg, source in tasks:
        ids, x = g.candidates(seeds)
        order = [ids[i] for i in np.argsort(-model.predict_proba(x)[:, 1], kind="stable") if ids[i] != source]
        top, targets = order[:10], set(targets)
        m = ranking_metrics(top, targets)
        rows.append(dict(ndcg=m["ndcg_at_10"], recall=m["recall_at_10"], hit=float(bool(set(top) & targets)),
                         fusion_ndcg=fusion_ndcg))
    delta = [r["ndcg"] - r["fusion_ndcg"] for r in rows]
    return dict(tasks=len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in ("ndcg", "recall", "hit")},
                ltr_minus_fusion=dict(mean=float(np.mean(delta)), ci95=bootstrap_mean(delta)))


def main():
    with (ROOT / "data/papers.jsonl").open() as source:
        papers = [json.loads(line) for line in source if line.strip()]
    train_graph, test_graph = Graph(papers, 2018), Graph(papers, 2020)
    result, models = {}, {}
    for name, tasks, artifact in (("reference_recovery", reference_tasks, "temporal_recovery.json"),
                                  ("author_next", author_tasks, "author_next.json")):
        model, n_train = fit(train_graph, tasks(papers, train_graph, 2020))
        observations = json.loads((ROOT / "artifacts" / artifact).read_text())["observations"]
        frozen = [(t["seeds"], t["targets"], t["methods"]["fusion"]["ndcg"], None) for t in observations]
        result[name] = dict(train_tasks=n_train, weights=dict(zip(FEATURES, map(float, model[-1].coef_[0]))),
                            **test(test_graph, model, frozen))
        models[name] = model

    # E3: the reference-recovery model, unchanged, on the random holdout. Hidden edges leave the graph first.
    split = [json.loads(line) for line in (ROOT / "artifacts/recommendation_split.jsonl").open() if line.strip()]
    hidden = {(t["paper_id"], r) for t in split for r in t["relevant"]}
    kept = [dict(p, references=[r for r in p.get("references", []) if (p["id"], r) not in hidden]) for p in papers]
    holdout_graph = Graph(kept, max(year(p) for p in papers))
    fusion = {r["paper_id"]: r["fusion"] for r in json.loads((ROOT / "artifacts/holdout_fusion.json").read_text())["rows"]}
    frozen = [(t["seeds"], t["relevant"], fusion[t["paper_id"]], t["paper_id"]) for t in split]
    result["random_holdout"] = dict(model="reference_recovery", **test(holdout_graph, models["reference_recovery"], frozen))
    # One model for the app has to hold on every protocol, so also score the reference-recovery model on E2b.
    observations = json.loads((ROOT / "artifacts/author_next.json").read_text())["observations"]
    frozen = [(t["seeds"], t["targets"], t["methods"]["fusion"]["ndcg"], None) for t in observations]
    result["author_next_single_model"] = dict(model="reference_recovery",
                                              **test(test_graph, models["reference_recovery"], frozen))
    (ROOT / "artifacts/ltr.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
