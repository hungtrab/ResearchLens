"""Recover old references of later papers using only the older induced graph."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from researchlens.evaluation import bootstrap_mean, ranking_metrics
from researchlens.mining import MiningEngine

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = 2020
METHODS = ("popular", "content", "graph", "hybrid", "cocitation", "fusion")


def paired_delta(observations, treatment, baseline):
    delta = [t["methods"][treatment]["ndcg"] - t["methods"][baseline]["ndcg"] for t in observations]
    return dict(mean=float(np.mean(delta)), ci95=bootstrap_mean(delta))


def main():
    path = ROOT / "data/papers.jsonl"
    with path.open() as source:
        papers = [json.loads(line) for line in source if line.strip()]
    historical = [p for p in papers if 0 < (p.get("year") or 0) <= CUTOFF]
    engine = MiningEngine(historical)
    candidates = set(engine.ids)
    observations = []
    for paper in sorted(papers, key=lambda p: p["id"]):
        if (paper.get("year") or 0) <= CUTOFF:
            continue
        references = sorted(candidates.intersection(paper.get("references", [])))
        if len(references) < 6:
            continue
        rng = random.Random(f"42:{paper['id']}")
        hidden = set(rng.sample(references, len(references) // 3))
        seeds = sorted(set(references) - hidden)
        task = dict(source=paper["id"], year=paper["year"], seeds=seeds, targets=sorted(hidden), methods={})
        for method in METHODS:
            result = engine.recommend(seeds, limit=10, method=method, diversity=0)
            ranked = [p["id"] for p in result["items"]]
            assert not set(ranked).intersection(seeds) and set(ranked) <= candidates
            metrics = ranking_metrics(ranked, hidden)
            task["methods"][method] = dict(top10=ranked, ndcg=metrics["ndcg_at_10"], recall=metrics["recall_at_10"])
        observations.append(task)

    summary = {}
    for method in METHODS:
        rows = [t["methods"][method] for t in observations]
        summary[method] = {k: float(np.mean([r[k] for r in rows])) for k in ("ndcg", "recall")}
        summary[method]["queries_with_results"] = sum(bool(r["top10"]) for r in rows)
    result = dict(cutoff_year=CUTOFF, corpus_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  historical_papers=len(historical), historical_edges=engine.graph.number_of_edges(),
                  tasks=len(observations), hidden_edges=sum(len(t["targets"]) for t in observations),
                  summary=summary,
                  paired_cocitation_minus_ppr=paired_delta(observations, "cocitation", "graph"),
                  paired_fusion_minus_hybrid=paired_delta(observations, "fusion", "hybrid"),
                  paired_fusion_minus_cocitation=paired_delta(observations, "fusion", "cocitation"),
                  observations=observations)
    (ROOT / "artifacts/temporal_recovery.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "observations"}, indent=2))


if __name__ == "__main__":
    main()
