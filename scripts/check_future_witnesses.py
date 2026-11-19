"""Random holdout (E3): how many co-citation hits rely on witnesses published after the source paper?"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from researchlens.evaluation import ranking_metrics
from researchlens.mining import MiningEngine

papers = [json.loads(l) for l in (ROOT / "data/papers.jsonl").open() if l.strip()]
year = {p["id"]: p.get("year") or 0 for p in papers}
tasks = [json.loads(l) for l in (ROOT / "artifacts/recommendation_split.jsonl").open() if l.strip()]
hidden = {(t["paper_id"], r) for t in tasks for r in t["relevant"]}
train = MiningEngine([dict(p, references=[r for r in p.get("references", []) if (p["id"], r) not in hidden]) for p in papers])

hits = later_only = 0
past_only = {"cocitation": [], "fusion": []}
for t in tasks:
    src_year = year[t["paper_id"]]
    w = train.co_citation_witnesses(t["seeds"])
    for pid in t["relevant"]:
        if pid in w:
            hits += 1
            later_only += all(year[x["paper_id"]] > src_year for x in w[pid])
    # Re-run co-citation keeping only witnesses not newer than the source (what a reader could know then).
    pw = {c: [x for x in ev if year[x["paper_id"]] <= src_year] for c, ev in w.items()}
    pw = {c: ev for c, ev in pw.items() if ev and c != t["paper_id"]}
    co = sorted(pw, key=lambda c: (-len(pw[c]), c))
    past_only["cocitation"].append(ranking_metrics(co[:10], set(t["relevant"]))["ndcg_at_10"])
print("hidden refs with >=1 witness:", hits, " of which every witness is newer than the source:", later_only)
print("co-citation nDCG@10 using only witnesses not newer than source:", round(float(np.mean(past_only["cocitation"])), 3))
