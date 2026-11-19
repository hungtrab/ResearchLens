"""Decision model: given the saved papers, how likely is each candidate to be read next?

Candidates are the hybrid top 300 plus every paper linked to a saved paper. Each candidate gets eleven signals.
A gradient-boosted classifier, trained offline by scripts/decision_model.py, turns them and their ranks inside
the task into a score."""
import pickle
from functools import lru_cache

import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from researchlens.mining import co_citation_ranking, eligible_ranking, rank_positions, reciprocal_rank_fusion

GRAPH_FEATURES = ("co_citation_votes", "content", "personalized_pagerank", "cited_by_saved", "cites_saved",
                  "shared_references", "citation_count", "age")
FEATURES = GRAPH_FEATURES + ("specter_profile", "specter_closest_saved", "fusion")
QUERY_FEATURES = ("query_bm25", "query_tfidf")
QUERY_POOL = 100
POOL = 300


def year(paper):
    return int(paper.get("year") or 0)


@lru_cache(maxsize=4)
def citation_links(engine):
    """References and citing papers inside the corpus, keyed by paper id. An engine never changes after it is
    built (an import builds a new one), so the links are computed once per engine."""
    ids = set(engine.ids)
    refs = {p["id"]: set(p.get("references", [])) & ids for p in engine.papers}
    citers = {}
    for source, targets in refs.items():
        for target in targets:
            citers.setdefault(target, set()).add(source)
    return refs, citers


def base_scores(engine, seeds):
    """Content, graph and fusion score of every paper, and the co-citation witnesses. recommend() computes the
    same four for its own methods and passes them in, so a request computes them once."""
    saved = set(seeds)
    eligible = np.array([pid not in saved for pid in engine.ids])
    content, graph = engine.content_graph_scores(seeds, eligible)
    votes = engine.co_citation_witnesses(seeds)
    hybrid_rank = rank_positions(eligible_ranking(0.7 * content + 0.3 * graph, eligible))
    cocitation_rank = rank_positions(engine.pos[pid] for pid in co_citation_ranking(votes))
    return content, graph, reciprocal_rank_fusion([hybrid_rank, cocitation_rank], len(engine.ids)), votes


def graph_features(engine, refs, citers, seeds, now, extra=(), base=None):
    """The eight graph and text signals for every candidate; returns candidate ids and one row each.
    extra adds candidate indices from another retriever, such as a topic query."""
    saved = set(seeds)
    eligible = np.array([pid not in saved for pid in engine.ids])
    content, graph, _, votes = base or base_scores(engine, seeds)
    saved_refs = {}
    for s in saved:
        for ref in refs.get(s, ()):
            saved_refs[ref] = saved_refs.get(ref, 0) + 1
    pool = set(np.argsort(-(0.7 * content + 0.3 * graph))[:POOL]) | {engine.pos[c] for c in votes} | set(extra)
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


def query_retrieval(engine, query):
    """BM25 and TF-IDF scores of every paper against the reader's topic, and the BM25 top QUERY_POOL indices."""
    bm25, tfidf = engine.lexical_scores(query, "bm25"), engine.lexical_scores(query, "tfidf")
    return bm25 / (bm25.max(initial=0) or 1), tfidf, np.argsort(-bm25, kind="stable")[:QUERY_POOL]


def candidate_signals(engine, refs, citers, seeds, now, vectors, query=None, base=None):
    """Every candidate's signals in FEATURES order (eight graph, two SPECTER, fusion), then with a topic query
    the two QUERY_FEATURES; BM25 also adds its top QUERY_POOL papers to the candidates."""
    base = base or base_scores(engine, seeds)
    bm25, tfidf, extra = query_retrieval(engine, query) if query else (None, None, ())
    ids, x = graph_features(engine, refs, citers, seeds, now, extra, base)
    at = [engine.pos[pid] for pid in ids]
    columns = [x, semantic_features(engine, vectors, ids, seeds), base[2][at]]
    if query:
        columns += [bm25[at], tfidf[at]]
    return ids, np.column_stack(columns)


def with_ranks(x):
    """Append each signal's percentile rank inside the task, which does not depend on how big the graph is."""
    return np.column_stack([x, rankdata(x, axis=0) / len(x)]) if len(x) else x


class DecisionModel:
    def __init__(self, model, columns, table):
        self.model, self.table = model, table
        self.columns = [(FEATURES + QUERY_FEATURES).index(name) for name in columns]
        self.needs_query = bool(set(columns) & set(QUERY_FEATURES))
        self.dim = len(next(iter(table.values())))

    @classmethod
    def load(cls, model_path, vectors_path):
        with open(model_path, "rb") as source:
            saved = pickle.load(source)
        if not set(saved["features"]) <= set(FEATURES + QUERY_FEATURES):
            raise ValueError("Decision model was trained on different features")
        vectors = np.load(vectors_path)
        return cls(saved["model"], saved["features"],
                   dict(zip(vectors["ids"].tolist(), vectors["vectors"].astype(np.float32))))

    def vectors(self, engine):
        """SPECTER vectors in engine order; a paper imported after embedding gets a zero vector."""
        zero = np.zeros(self.dim, np.float32)
        return np.stack([self.table.get(pid, zero) for pid in engine.ids])

    def rank(self, engine, seeds, query=None, base=None):
        """Candidate ids, model score, and the signals behind each score. A topic query adds candidates and
        two signals; it needs a model trained with them. base is base_scores(engine, seeds) if already known."""
        if self.needs_query and not query:
            raise ValueError("This decision model needs a topic query")
        ids, x = candidate_signals(engine, *citation_links(engine), seeds, max(year(p) for p in engine.papers),
                                   self.vectors(engine), query if self.needs_query else None, base)
        # One OpenMP thread: a few hundred rows gain little from more, and right after the PageRank step the
        # OpenMP and BLAS pools spin against each other (p50 145 ms with all threads, 78 ms with one).
        with threadpool_limits(1, user_api="openmp"):
            probability = self.model.predict_proba(with_ranks(x[:, self.columns]))[:, 1]
        return ids, probability, x
