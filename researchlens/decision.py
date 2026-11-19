"""Decision model: given the saved papers, how likely is each candidate to be read next?

Candidates are the hybrid top 300 plus every paper linked to a saved paper. Each candidate gets ten signals.
A gradient-boosted classifier, trained offline by scripts/decision_model.py, turns them into a probability."""
import pickle

import numpy as np
from scipy.stats import rankdata

GRAPH_FEATURES = ("co_citation_votes", "content", "personalized_pagerank", "cited_by_saved", "cites_saved",
                  "shared_references", "citation_count", "age")
FEATURES = GRAPH_FEATURES + ("specter_profile", "specter_closest_saved")
POOL = 300


def year(paper):
    return int(paper.get("year") or 0)


def citation_links(engine):
    """References and citing papers inside the corpus, keyed by paper id."""
    ids = set(engine.ids)
    refs = {p["id"]: set(p.get("references", [])) & ids for p in engine.papers}
    citers = {}
    for source, targets in refs.items():
        for target in targets:
            citers.setdefault(target, set()).add(source)
    return refs, citers


def graph_features(engine, refs, citers, seeds, now):
    """The eight graph and text signals for every candidate; returns candidate ids and one row each."""
    saved = set(seeds)
    eligible = np.array([pid not in saved for pid in engine.ids])
    content, graph = engine.content_graph_scores(seeds, eligible)
    votes = engine.co_citation_witnesses(seeds)
    saved_refs = {}
    for s in saved:
        for ref in refs.get(s, ()):
            saved_refs[ref] = saved_refs.get(ref, 0) + 1
    pool = set(np.argsort(-(0.7 * content + 0.3 * graph))[:POOL]) | {engine.pos[c] for c in votes}
    for s in saved:
        pool |= {engine.pos[x] for x in refs.get(s, set()) | citers.get(s, set())}
    ids, rows = [], []
    for i in sorted(i for i in pool if eligible[i]):
        pid = engine.ids[i]
        rows.append([np.log1p(len(votes.get(pid, []))), content[i], graph[i],
                     sum(pid in refs.get(s, ()) for s in saved),
                     sum(pid in citers.get(s, ()) for s in saved),
                     np.log1p(sum(saved_refs.get(r, 0) for r in refs.get(pid, ()))),
                     np.log1p(len(citers.get(pid, ()))), now - year(engine.papers[i])])
        ids.append(pid)
    return ids, np.array(rows, float)


def semantic_features(engine, vectors, ids, seeds):
    """SPECTER cosine to the mean of the saved papers and to the closest saved paper."""
    saved = vectors[[engine.pos[s] for s in seeds]]
    mine = vectors[[engine.pos[pid] for pid in ids]]
    profile = saved.mean(axis=0)
    profile /= np.linalg.norm(profile) or 1
    return np.column_stack([mine @ profile, (mine @ saved.T).max(axis=1)])


def with_ranks(x):
    """Append each signal's percentile rank inside the task, which does not depend on how big the graph is."""
    return np.column_stack([x, rankdata(x, axis=0) / len(x)]) if len(x) else x


class DecisionModel:
    def __init__(self, model, uses_ranks, table):
        self.model, self.uses_ranks, self.table = model, uses_ranks, table
        self.dim = len(next(iter(table.values())))

    @classmethod
    def load(cls, model_path, vectors_path):
        with open(model_path, "rb") as source:
            saved = pickle.load(source)
        if tuple(saved["features"]) != FEATURES:
            raise ValueError("Decision model was trained on different features")
        vectors = np.load(vectors_path)
        return cls(saved["model"], saved["variant"].endswith("_rank"),
                   dict(zip(vectors["ids"].tolist(), vectors["vectors"].astype(np.float32))))

    def vectors(self, engine):
        """SPECTER vectors in engine order; a paper imported after embedding gets a zero vector."""
        zero = np.zeros(self.dim, np.float32)
        return np.stack([self.table.get(pid, zero) for pid in engine.ids])

    def rank(self, engine, seeds):
        """Candidate ids, probability of being read next, and the ten signals behind each probability."""
        refs, citers = citation_links(engine)
        ids, x = graph_features(engine, refs, citers, seeds, max(year(p) for p in engine.papers))
        x = np.column_stack([x, semantic_features(engine, self.vectors(engine), ids, seeds)])
        probability = self.model.predict_proba(with_ranks(x) if self.uses_ranks else x)[:, 1]
        return ids, probability, x
