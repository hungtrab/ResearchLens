"""Re-check fusion on the frozen random-holdout split, using the app's recommend()."""
import json
from pathlib import Path

import numpy as np

from researchlens.evaluation import bootstrap_mean, ranking_metrics
from researchlens.mining import MiningEngine

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("content", "graph", "cocitation", "hybrid", "fusion")


def main():
    papers = [json.loads(line) for line in (ROOT / "data/papers.jsonl").open() if line.strip()]
    tasks = [json.loads(line) for line in (ROOT / "artifacts/recommendation_split.jsonl").open() if line.strip()]
    hidden = {(t["paper_id"], r) for t in tasks for r in t["relevant"]}
    train = MiningEngine([dict(p, references=[r for r in p.get("references", []) if (p["id"], r) not in hidden])
                          for p in papers])

    rows = []
    for task in tasks:
        row = {"paper_id": task["paper_id"]}
        for method in METHODS:
            items = train.recommend(task["seeds"], limit=11, method=method, diversity=0)["items"]
            ranked = [item["id"] for item in items if item["id"] != task["paper_id"]][:10]
            row[method] = ranking_metrics(ranked, set(task["relevant"]))["ndcg_at_10"]
            row[f"{method}_hit5"] = float(bool(set(ranked[:5]) & set(task["relevant"])))
            row[f"{method}_hit10"] = float(bool(set(ranked) & set(task["relevant"])))
        rows.append(row)

    summary = {m: float(np.mean([r[m] for r in rows])) for m in METHODS}
    hits = {m: {k: float(np.mean([r[f"{m}_{k}"] for r in rows])) for k in ("hit5", "hit10")} for m in METHODS}
    paired = {f"fusion_minus_{base}": dict(mean=float(np.mean(d)), ci95=bootstrap_mean(d))
              for base in ("hybrid", "cocitation")
              for d in [[r["fusion"] - r[base] for r in rows]]}
    result = dict(tasks=len(rows), ndcg_at_10=summary, hits=hits, paired=paired, rows=rows)
    (ROOT / "artifacts/holdout_fusion.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
