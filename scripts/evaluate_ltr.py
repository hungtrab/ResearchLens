"""Learning-to-rank check. A logistic regression over eight named signals is trained on a 2018 graph
(tasks from 2019-2020 papers) and tested on the frozen 2020 tasks of E2 and E2b. Nothing is tuned on the test tasks."""
import json
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from researchlens.evaluation import bootstrap_mean, ranking_metrics
from researchlens.mining import MiningEngine
from scripts.evaluate_author_next import authors, year

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("co_citation_votes", "content", "personalized_pagerank", "cited_by_saved", "cites_saved",
            "shared_references", "citation_count", "age")
POOL = 300


class Graph:
    def __init__(self, papers, cutoff):
        self.cutoff = cutoff
        self.engine = MiningEngine([p for p in papers if 0 < year(p) <= cutoff])
        ids = set(self.engine.ids)
        self.refs = {p["id"]: set(p.get("references", [])) & ids for p in self.engine.papers}
        self.citers = {}
        for source, refs in self.refs.items():
            for ref in refs:
                self.citers.setdefault(ref, set()).add(source)

    def candidates(self, seeds):
        """Hybrid top POOL plus every paper linked to the saved set; returns ids and one feature row each."""
        e, saved = self.engine, set(seeds)
        eligible = np.array([pid not in saved for pid in e.ids])
        content, graph = e.content_graph_scores(seeds, eligible)
        votes = e.co_citation_witnesses(seeds)
        saved_refs = {}
        for s in saved:
            for ref in self.refs.get(s, ()):
                saved_refs[ref] = saved_refs.get(ref, 0) + 1
        pool = set(np.argsort(-(0.7 * content + 0.3 * graph))[:POOL]) | {e.pos[c] for c in votes}
        for s in saved:
            pool |= {e.pos[x] for x in self.refs.get(s, set()) | self.citers.get(s, set())}
        rows, ids = [], []
        for i in sorted(i for i in pool if eligible[i]):
            pid = e.ids[i]
            rows.append([np.log1p(len(votes.get(pid, []))), content[i], graph[i],
                         sum(pid in self.refs.get(s, ()) for s in saved),
                         sum(pid in self.citers.get(s, ()) for s in saved),
                         np.log1p(sum(saved_refs.get(r, 0) for r in self.refs.get(pid, ()))),
                         np.log1p(len(self.citers.get(pid, ()))), self.cutoff - year(e.papers[i])])
            ids.append(pid)
        return ids, np.array(rows, float)


def reference_tasks(papers, g, last):
    """E2 protocol: hide a third of a later paper's references."""
    for p in sorted(papers, key=lambda p: p["id"]):
        if not g.cutoff < year(p) <= last:
            continue
        refs = sorted(set(g.engine.ids) & set(p.get("references", [])))
        if len(refs) >= 6:
            hidden = set(random.Random(f"42:{p['id']}").sample(refs, len(refs) // 3))
            yield sorted(set(refs) - hidden), hidden


def author_tasks(papers, g, last):
    """E2b protocol: an author's library before the cutoff, first-time citations after it."""
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
            yield sorted(library), targets


def fit(g, tasks):
    xs, ys = [], []
    for seeds, targets in tasks:
        ids, x = g.candidates(seeds)
        xs.append(x)
        ys.append([pid in targets for pid in ids])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    return model.fit(np.vstack(xs), np.concatenate(ys)), len(xs)


def test(g, model, observations):
    rows = []
    for t in observations:
        ids, x = g.candidates(t["seeds"])
        top = [ids[i] for i in np.argsort(-model.predict_proba(x)[:, 1], kind="stable")[:10]]
        m = ranking_metrics(top, set(t["targets"]))
        rows.append(dict(ndcg=m["ndcg_at_10"], recall=m["recall_at_10"], hit=float(bool(set(top) & set(t["targets"]))),
                         fusion_ndcg=t["methods"]["fusion"]["ndcg"]))
    delta = [r["ndcg"] - r["fusion_ndcg"] for r in rows]
    return dict(tasks=len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in ("ndcg", "recall", "hit")},
                ltr_minus_fusion=dict(mean=float(np.mean(delta)), ci95=bootstrap_mean(delta)))


def main():
    with (ROOT / "data/papers.jsonl").open() as source:
        papers = [json.loads(line) for line in source if line.strip()]
    train_graph, test_graph = Graph(papers, 2018), Graph(papers, 2020)
    result = {}
    for name, tasks, artifact in (("reference_recovery", reference_tasks, "temporal_recovery.json"),
                                  ("author_next", author_tasks, "author_next.json")):
        model, n_train = fit(train_graph, tasks(papers, train_graph, 2020))
        observations = json.loads((ROOT / "artifacts" / artifact).read_text())["observations"]
        coef = model[-1].coef_[0]
        result[name] = dict(train_tasks=n_train, weights=dict(zip(FEATURES, map(float, coef))),
                            **test(test_graph, model, observations))
    (ROOT / "artifacts/ltr.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
