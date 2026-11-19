"""E2b: what does an author read next? Library = the author's papers up to the cutoff and their references.
Targets = papers the author cites for the first time after the cutoff. The graph stops at the cutoff."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from researchlens.evaluation import bootstrap_mean, ranking_metrics
from researchlens.mining import MiningEngine

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = 2020
MIN_LIBRARY = 3
METHODS = ("popular", "content", "graph", "hybrid", "cocitation", "fusion")


def year(paper):
    return int(paper.get("year") or 0)


def authors(paper):
    value = paper.get("authors") or []
    return ast.literal_eval(value) if isinstance(value, str) else value


def main():
    path = ROOT / "data/papers.jsonl"
    with path.open() as source:
        papers = [json.loads(line) for line in source if line.strip()]
    historical = [p for p in papers if 0 < year(p) <= CUTOFF]
    engine = MiningEngine(historical)
    candidates = set(engine.ids)

    by_author = {}
    for paper in papers:
        for name in authors(paper):
            by_author.setdefault(name, []).append(paper)

    observations = []
    for name, written in sorted(by_author.items()):
        before = [p for p in written if 0 < year(p) <= CUTOFF]
        after = [p for p in written if year(p) > CUTOFF]
        if not before or not after:
            continue
        own = {p["id"] for p in written}
        library = {p["id"] for p in before} | {r for p in before for r in p.get("references", [])}
        library &= candidates
        new_reads = {r for p in after for r in p.get("references", [])} & candidates
        targets = new_reads - library - own
        if len(library) < MIN_LIBRARY or not targets:
            continue
        seeds = sorted(library)
        task = dict(author=name, seeds=seeds, targets=sorted(targets), methods={})
        for method in METHODS:
            ranked = [p["id"] for p in engine.recommend(seeds, limit=10, method=method, diversity=0)["items"]]
            metrics = ranking_metrics(ranked, targets)
            task["methods"][method] = dict(top10=ranked, ndcg=metrics["ndcg_at_10"], recall=metrics["recall_at_10"],
                                           hit=float(bool(set(ranked) & targets)))
        observations.append(task)

    summary = {m: {k: float(np.mean([t["methods"][m][k] for t in observations])) for k in ("ndcg", "recall", "hit")}
               for m in METHODS}
    delta = lambda a, b: (lambda d: dict(mean=float(np.mean(d)), ci95=bootstrap_mean(d)))(
        [t["methods"][a]["ndcg"] - t["methods"][b]["ndcg"] for t in observations])
    result = dict(cutoff_year=CUTOFF, corpus_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  authors=len(observations), targets=sum(len(t["targets"]) for t in observations),
                  median_library=float(np.median([len(t["seeds"]) for t in observations])),
                  summary=summary,
                  paired_fusion_minus_hybrid=delta("fusion", "hybrid"),
                  paired_fusion_minus_cocitation=delta("fusion", "cocitation"),
                  paired_cocitation_minus_content=delta("cocitation", "content"),
                  observations=observations)
    out = ROOT / "artifacts/author_next.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != "observations"}, indent=2))


if __name__ == "__main__":
    main()
