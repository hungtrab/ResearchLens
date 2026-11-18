"""Sparse lexical retrieval, citation mining and transparent recommendations."""
from __future__ import annotations

from collections import Counter
import math
import re
import shlex

import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize

TOKEN = re.compile(r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]+\b")
RECOMMEND_METHODS = ("fusion", "content", "graph", "hybrid", "popular", "cocitation")
RRF_K = 60


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def unit(values):
    x = np.asarray(values, dtype=float)
    m = x.max(initial=0)
    return x / m if m > 0 else np.zeros_like(x)


def contains(text, term):
    """Whole-word match of term in text; a space in term matches any whitespace."""
    pattern = r"(?<!\w)" + re.escape(term).replace(r"\ ", r"\s+") + r"(?!\w)"
    return re.search(pattern, text) is not None


def query_parts(query):
    """Quoted phrases, -exclusions and OR terms. Unquoted terms use OR semantics."""
    try:
        parts = shlex.split(query)
    except ValueError as exc:
        raise ValueError("Close the quotation marks in your query.") from exc
    positive = []
    excluded = []
    phrases = []
    for p in parts:
        if p.upper() in {"AND", "OR"}:
            continue
        if p.startswith("-") and len(p) > 1:
            excluded.append(p[1:].lower())
        else:
            positive.append(p)
            if " " in p:
                phrases.append(p.lower())
    and_mode = any(part.upper() == "AND" for part in parts)
    return " ".join(positive), excluded, phrases, and_mode


def graph_metrics(graph):
    if not graph:
        return {}, {}, {}, {}
    pagerank = nx.pagerank(graph, alpha=0.85, max_iter=300, tol=1e-10)
    ids = list(graph)
    a = nx.to_scipy_sparse_array(graph, nodelist=ids, dtype=float, format="csr")
    # Nonnegative power iteration avoids arbitrary signs from singular vectors.
    authority = np.ones(len(ids)) / math.sqrt(len(ids))
    for _ in range(200):
        h = np.asarray(a @ authority).ravel()
        h = h / (np.linalg.norm(h) or 1)
        new = np.asarray(a.T @ h).ravel()
        new = new / (np.linalg.norm(new) or 1)
        converged = np.linalg.norm(new - authority) < 1e-9
        authority = new
        hub = h
        if converged:
            break
    authority = authority / (authority.sum() or 1)
    hub = hub / (hub.sum() or 1)
    if graph.number_of_edges():
        groups = nx.community.louvain_communities(graph.to_undirected(), seed=42)
    else:
        groups = [{i} for i in ids]
    groups = sorted(groups, key=lambda c: (-len(c), min(c)))
    communities = {pid: i for i, g in enumerate(groups) for pid in g}
    return pagerank, dict(zip(ids, authority)), dict(zip(ids, hub)), communities


def eligible_ranking(scores, eligible):
    """Indices of eligible papers from best to worst score; ties keep corpus order."""
    return [i for i in np.argsort(-scores, kind="stable") if eligible[i]]


def co_citation_ranking(witnesses):
    """Candidate ids with the most witnesses first, ties broken by id."""
    return sorted(witnesses, key=lambda pid: (-len(witnesses[pid]), pid))


def rank_positions(order):
    """Map each item of an ordered sequence to its 1-based rank."""
    return {item: rank for rank, item in enumerate(order, start=1)}


def reciprocal_rank_fusion(rankings, size):
    """Sum 1 / (RRF_K + rank) over the rankings that hold each index, scaled so the best is 1 as MMR expects."""
    fused = np.zeros(size)
    for ranking in rankings:
        for i, rank in ranking.items():
            fused[i] += 1 / (RRF_K + rank)
    return unit(fused)


def fusion_reason(hybrid_rank, cocitation_rank, citers):
    """Name the content/graph rank and the co-citation rank that were fused."""
    suffix = f"reciprocal rank fusion k={RRF_K}."
    if cocitation_rank is None:
        return f"Content/graph rank {hybrid_rank}, not co-cited with a saved paper, {suffix}"
    papers = "1 paper cites" if citers == 1 else f"{citers} papers cite"
    return f"Co-citation rank {cocitation_rank} ({papers} it with a saved paper) + content/graph rank {hybrid_rank}, {suffix}"


class MiningEngine:
    def __init__(self, papers):
        if not papers:
            raise ValueError("At least one paper is required.")
        self.papers = papers
        self.ids = [p["id"] for p in papers]
        self.pos = {pid: i for i, pid in enumerate(self.ids)}
        if len(self.pos) != len(papers):
            raise ValueError("Paper IDs must be unique.")
        self.by_id = {p["id"]: p for p in papers}
        self.texts = [p["title"] + " " + p.get("abstract", "") for p in papers]
        weighted = [p["title"] + " " + p["title"] + " " + p.get("abstract", "") for p in papers]
        self.vectorizer = CountVectorizer(analyzer=tokens, max_features=60000)
        if not any(tokens(text) for text in weighted):
            raise ValueError("Corpus must contain indexable words.")
        self.counts = self.vectorizer.fit_transform(weighted).tocsr()
        self.transformer = TfidfTransformer(sublinear_tf=True)
        self.tfidf = self.transformer.fit_transform(self.counts).tocsr()
        df = np.asarray((self.counts > 0).sum(axis=0)).ravel()
        n = len(papers)
        self.idf = np.log1p((n - df + 0.5) / (df + 0.5))
        self.length = np.asarray(self.counts.sum(axis=1)).ravel()
        self.avg_length = self.length.mean() or 1
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.ids)
        self.graph.add_edges_from(
            (p["id"], r) for p in papers for r in p.get("references", []) if r in self.pos and r != p["id"]
        )
        self.pagerank, self.authority, self.hubs, self.communities = graph_metrics(self.graph)
        self.pr = np.array([self.pagerank[pid] for pid in self.ids])
        self.features = self.vectorizer.get_feature_names_out()

    def lexical_scores(self, query, method="bm25"):
        q = self.vectorizer.transform([query])
        columns = q.indices
        if not len(columns):
            return np.zeros(len(self.ids))
        if method == "tfidf":
            return np.asarray((self.tfidf @ self.transformer.transform(q).T).toarray()).ravel()
        tf = self.counts[:, columns].toarray()
        norm = 1.5 * (1 - 0.75 + 0.75 * self.length / self.avg_length)
        return (self.idf[columns] * tf * 2.5 / (tf + norm[:, None])).sum(axis=1)

    def search(self, query="", method="hybrid", year_min=None, year_max=None, topic=None, venue=None,
               open_access=False, offset=0, limit=20):
        if method not in {"bm25", "tfidf", "hybrid", "pagerank"}:
            raise ValueError("Unknown ranking method")
        clean, excluded, phrases, and_mode = query_parts(query)
        lexical = self.lexical_scores(clean, "tfidf" if method == "tfidf" else "bm25")
        score = lexical.copy()
        if method == "hybrid":
            score = 0.85 * unit(lexical) + 0.15 * unit(self.pr)
        if method == "pagerank" or not query.strip():
            score = self.pr.copy()
        wanted = set(tokens(clean))
        indices = []
        for i, p in enumerate(self.papers):
            text = self.texts[i].lower()
            if clean and (not wanted or lexical[i] <= 0):
                continue
            if any(not contains(text, x) for x in phrases) or any(contains(text, x) for x in excluded):
                continue
            if and_mode and not wanted.issubset(set(tokens(text))):
                continue
            year = p.get("year") or 0
            if year_min is not None and year < year_min:
                continue
            if year_max is not None and year > year_max:
                continue
            if topic and topic not in p.get("topics", []) and topic != p.get("collection_topic"):
                continue
            if venue and p.get("venue") != venue:
                continue
            if open_access and not p.get("open_access"):
                continue
            indices.append(i)
        indices.sort(key=lambda i: (-score[i], self.ids[i]))
        items = [self.summary(i, float(score[i]), float(lexical[i])) for i in indices[offset:offset + limit]]
        return {"total": len(indices), "items": items}

    def summary(self, i, score=None, lexical=None):
        p = self.papers[i]
        fields = ("id", "title", "abstract", "authors", "year", "venue", "topics", "collection_topic",
                  "citations", "source", "url", "open_access")
        result = {k: p.get(k) for k in fields}
        result.update(
            pagerank=float(self.pr[i]),
            authority=float(self.authority[p["id"]]),
            community=self.communities[p["id"]],
            local_citations=self.graph.in_degree(p["id"]),
        )
        if score is not None:
            result["score"] = score
        if lexical is not None:
            result["lexical_score"] = lexical
        return result

    def keywords(self, pid, n=8):
        row = self.tfidf.getrow(self.pos[pid])
        order = np.argsort(-row.data)[:n]
        return [{"term": str(self.features[row.indices[i]]), "weight": round(float(row.data[i]), 4)} for i in order]

    def co_citation_witnesses(self, seeds):
        """Map each candidate to the papers that cite it together with a seed; one vote per citing paper."""
        saved = set(seeds)
        citing_papers = {source for seed in saved for source in self.graph.predecessors(seed)}
        witnesses = {}
        for source in sorted(citing_papers):
            references = set(self.graph.successors(source))
            shared_seed = min(references & saved)
            for candidate in references - saved:
                witnesses.setdefault(candidate, []).append({"paper_id": source, "seed_id": shared_seed})
        return witnesses

    def titled_witnesses(self, evidence):
        """The first three witnesses with the citing paper and seed titles attached."""
        return [
            dict(w, paper_title=self.by_id[w["paper_id"]]["title"], seed_title=self.by_id[w["seed_id"]]["title"])
            for w in evidence[:3]
        ]

    def co_citation_item(self, pid, evidence):
        """A co-citation recommendation scored by its number of witnesses."""
        count = len(evidence)
        if count > 1:
            reason = f"{count} papers cite this paper alongside a saved paper."
        else:
            reason = "One paper cites this paper alongside a saved paper."
        summary = self.summary(self.pos[pid], float(count))
        return dict(summary, co_citation_count=count, witnesses=self.titled_witnesses(evidence), reason=reason)

    def content_graph_scores(self, seeds, eligible):
        """Content similarity and personalized PageRank, each scaled so the best non-seed candidate scores 1."""
        profile = normalize(np.asarray(self.tfidf[[self.pos[s] for s in seeds]].mean(axis=0)))
        content = np.asarray(self.tfidf @ profile.T).ravel()
        personalization = {pid: 1 / len(seeds) if pid in seeds else 0 for pid in self.ids}
        ppr = nx.pagerank(self.graph, alpha=0.85, personalization=personalization, max_iter=300, tol=1e-9)
        graph = np.array([ppr[pid] for pid in self.ids])
        content = content / (content[eligible].max(initial=0) or 1)
        graph = graph / (graph[eligible].max(initial=0) or 1)
        return content, graph

    def mmr_select(self, scores, eligible, limit, diversity):
        """Greedy MMR over the top 150: (1 - diversity) * score - diversity * similarity to earlier picks."""
        pool = eligible_ranking(scores, eligible)[:max(150, limit)]
        if diversity == 0:
            return pool[:limit]
        selected = []
        while pool and len(selected) < limit:
            if selected:
                similarity = self.tfidf[pool] @ self.tfidf[selected].T
                redundancy = np.asarray(similarity.max(axis=1).toarray()).ravel()
            else:
                redundancy = np.zeros(len(pool))
            mmr = (1 - diversity) * scores[pool] - diversity * redundancy
            selected.append(pool.pop(int(np.argmax(mmr))))
        return selected

    def score_reason(self, method, i, content, graph):
        """How the content, graph, hybrid or popular score of paper i was computed."""
        if method == "content":
            return f"Normalized content similarity {content[i]:.3f}."
        if method == "graph":
            return f"Normalized personalized PageRank {graph[i]:.3f}."
        if method == "hybrid":
            return f"Content 0.7 × {content[i]:.3f} + personalized PageRank 0.3 × {graph[i]:.3f}."
        return f"Global PageRank {self.pr[i]:.5f}."

    def recommend(self, seeds, limit=12, method="fusion", diversity=0.15):
        seeds = list(dict.fromkeys(seeds))
        if method not in RECOMMEND_METHODS:
            raise ValueError("Unknown recommendation method")
        if any(s not in self.pos for s in seeds):
            raise ValueError("Unknown seed paper")
        if not 0 <= diversity <= 1:
            raise ValueError("Diversity must be between zero and one")
        if not seeds:
            ordered = np.argsort(-self.pr, kind="stable")[:limit]
            reason = "Cold start: influential within this corpus."
            return {"cold_start": True, "items": [dict(self.summary(i, float(self.pr[i])), reason=reason) for i in ordered]}
        witnesses = self.co_citation_witnesses(seeds)
        cocitation_order = co_citation_ranking(witnesses)
        if method == "cocitation":
            items = [self.co_citation_item(pid, witnesses[pid]) for pid in cocitation_order[:limit]]
            return {"cold_start": False, "items": items}

        eligible = np.array([pid not in seeds for pid in self.ids])
        content, graph = self.content_graph_scores(seeds, eligible)
        hybrid = 0.7 * content + 0.3 * graph
        hybrid_rank = rank_positions(eligible_ranking(hybrid, eligible))
        cocitation_rank = rank_positions(self.pos[pid] for pid in cocitation_order)
        scores = {
            "content": content,
            "graph": graph,
            "hybrid": hybrid,
            "popular": unit(self.pr),
            "fusion": reciprocal_rank_fusion([hybrid_rank, cocitation_rank], len(self.ids)),
        }[method]
        seed_terms = set(tokens(" ".join(self.texts[self.pos[s]] for s in seeds)))
        items = []
        for i in self.mmr_select(scores, eligible, limit, diversity):
            pid = self.ids[i]
            shared = set(tokens(self.texts[i])) & seed_terms
            words = [x["term"] for x in self.keywords(pid, 20) if x["term"] in shared][:4]
            item = dict(self.summary(i, float(scores[i])), content_score=float(content[i]),
                        graph_score=float(graph[i]), shared_terms=words)
            if method == "fusion":
                evidence = witnesses.get(pid, [])
                item.update(co_citation_count=len(evidence), witnesses=self.titled_witnesses(evidence))
                reason = fusion_reason(hybrid_rank[i], cocitation_rank.get(i), len(evidence))
            else:
                reason = self.score_reason(method, i, content, graph)
            if words and method in {"content", "hybrid", "fusion"}:
                reason += " Shared terms: " + ", ".join(words) + "."
            if diversity:
                reason += f" Diversity reranking enabled (weight {diversity:g})."
            item["reason"] = reason
            items.append(item)
        return {"cold_start": False, "items": items}

    def subgraph(self, pid=None, limit=70):
        if pid:
            if pid not in self.pos:
                raise KeyError(pid)
            neighborhood = set(self.graph.predecessors(pid)) | set(self.graph.successors(pid))
            ids = [pid] + sorted(neighborhood, key=lambda p: -self.pagerank[p])[:limit - 1]
        else:
            ids = sorted(self.ids, key=lambda p: -self.pagerank[p])[:limit]
        sub = self.graph.subgraph(ids)
        layout = nx.spring_layout(sub.to_undirected(), seed=42, iterations=100, k=0.75)
        nodes = [
            {
                "id": p,
                "title": self.by_id[p]["title"],
                "year": self.by_id[p].get("year"),
                "pagerank": self.pagerank[p],
                "community": self.communities[p],
                "x": float(layout[p][0]),
                "y": float(layout[p][1]),
            }
            for p in ids
        ]
        return {
            "nodes": nodes,
            "edges": [{"source": a, "target": b} for a, b in sub.edges],
            "direction": "citing → cited",
            "total_nodes": len(self.graph),
            "total_edges": self.graph.number_of_edges(),
        }

    def stats(self):
        connected = list(nx.weakly_connected_components(self.graph))
        top = sorted(self.ids, key=lambda p: -self.pagerank[p])[:10]
        return {
            "papers": len(self.ids),
            "edges": self.graph.number_of_edges(),
            "communities": len(set(self.communities.values())),
            "isolates": len(list(nx.isolates(self.graph))),
            "largest_component": max(map(len, connected), default=0),
            "abstracts": sum(bool(p.get("abstract")) for p in self.papers),
            "sources": dict(Counter(p["source"] for p in self.papers)),
            "years": dict(sorted(Counter(p.get("year") for p in self.papers if p.get("year")).items())),
            "topics": dict(Counter(p["collection_topic"] for p in self.papers)),
            "top_papers": [self.summary(self.pos[p]) for p in top],
            "community_sizes": dict(Counter(self.communities.values())),
        }


ASPECTS = {
    "novelty": r"novel|original|contribut|incremental",
    "soundness": r"theor|proof|correct|assumption|rigor|sound",
    "evaluation": r"experiment|baseline|dataset|benchmark|evaluation|ablation",
    "clarity": r"clear|clarity|writing|written|readab|explain",
    "reproducibility": r"reproduc|code|implement|hyperparam|detail",
}
POS = {"good", "excellent", "strong", "novel", "clear", "interesting", "convincing", "thorough", "significant",
       "impressive", "useful", "well"}
NEG = {"weak", "unclear", "limited", "missing", "poor", "incorrect", "confusing", "insufficient", "flaw", "problem",
       "lack", "lacks"}
NEGATIONS = {"no", "not", "never", "hardly", "isn", "doesn", "wasn"}


def polarity(score):
    return "positive" if score > 0 else "critical" if score < 0 else "mixed"


def opinion(text):
    """Rule baseline with local negation and exact sentence evidence; not a factual verdict."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    findings = []
    for sentence in sentences:
        if not sentence:
            continue
        words = re.findall(r"[a-z]+", sentence.lower())
        score = 0
        for i, w in enumerate(words):
            s = 1 if w in POS else -1 if w in NEG else 0
            if any(x in NEGATIONS for x in words[max(0, i - 3):i]):
                s = -s
            score += s
        aspects = [a for a, pattern in ASPECTS.items() if re.search(pattern, sentence, re.I)]
        if aspects:
            findings.append({"text": sentence, "aspects": aspects, "polarity": polarity(score), "score": score})
    total = sum(x["score"] for x in findings)
    return {
        "label": polarity(total),
        "score": total,
        "evidence": findings,
        "method": "Aspect keywords + polarity lexicon with three-token negation; sentence-level heuristic.",
    }


def extract(text):
    patterns = {
        "datasets": r"\b(?:ImageNet|CIFAR[- ]?(?:10|100)|MNIST|SVHN|SQuAD|GLUE|Cora|CiteSeer|PubMed|MovieLens|COCO|WikiText|MS MARCO)\b",
        "methods": r"\b(?:BERT|GPT[- ]?\d*|Transformer[s]?|GCN|GAT|GraphSAGE|LSTM|CNN|SVM|PageRank|BM25|TF[- ]?IDF|reinforcement learning|contrastive learning)\b",
        "doi": r'10\.\d{4,9}/[^\s<>"\]]+',
        "arxiv": r"\b\d{4}\.\d{4,5}(?:v\d+)?\b",
    }
    return {
        kind: [{"text": m.group(), "start": m.start(), "end": m.end(), "method": "pattern"}
               for m in re.finditer(pattern, text, re.I)]
        for kind, pattern in patterns.items()
    }
