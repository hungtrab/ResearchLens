"""U3: does a SPECTER1 profile beat the TF-IDF profile as the content signal? Same frozen E2 and E2b tasks,
same app recommend(), only content_graph_scores swaps TF-IDF cosine for SPECTER cosine. Run embed_specter.py first."""
import json
from pathlib import Path

import numpy as np

from researchlens.evaluation import bootstrap_mean, ranking_metrics
from researchlens.mining import MiningEngine
from scripts.evaluate_author_next import year

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = 2020
METHODS = ("content", "hybrid", "fusion")


def use_specter(engine, table):
    """Replace only the content half of content_graph_scores; PageRank and scaling stay as in the app."""
    vectors = np.stack([table[pid] for pid in engine.ids]).astype(np.float32)
    tfidf_scores = engine.content_graph_scores

    def scores(seeds, eligible):
        _, graph = tfidf_scores(seeds, eligible)
        profile = vectors[[engine.pos[s] for s in seeds]].mean(axis=0)
        content = vectors @ (profile / (np.linalg.norm(profile) or 1))
        return content / (content[eligible].max(initial=0) or 1), graph

    engine.content_graph_scores = scores


def main():
    with (ROOT / "data/papers.jsonl").open() as source:
        papers = [json.loads(line) for line in source if line.strip()]
    saved = np.load(ROOT / "artifacts/specter.npz")
    table = dict(zip(saved["ids"].tolist(), saved["vectors"]))
    engine = MiningEngine([p for p in papers if 0 < year(p) <= CUTOFF])
    use_specter(engine, table)

    result = {}
    for name, artifact in (("reference_recovery", "temporal_recovery.json"), ("author_next", "author_next.json")):
        observations = json.loads((ROOT / "artifacts" / artifact).read_text())["observations"]
        run = {}
        for method in METHODS:
            ndcg, delta = [], []
            for t in observations:
                ranked = [p["id"] for p in engine.recommend(t["seeds"], limit=10, method=method, diversity=0)["items"]]
                ndcg.append(ranking_metrics(ranked, set(t["targets"]))["ndcg_at_10"])
                delta.append(ndcg[-1] - t["methods"][method]["ndcg"])
            run[method] = dict(ndcg=float(np.mean(ndcg)),
                               specter_minus_tfidf=dict(mean=float(np.mean(delta)), ci95=bootstrap_mean(delta)))
        result[name] = run
    (ROOT / "artifacts/specter_eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
