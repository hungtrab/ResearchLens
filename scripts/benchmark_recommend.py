"""Warm in-process latency of each recommender on the full corpus, the way the API calls it.

Saved papers are the seed sets of the first 50 E2 tasks; the topic, when used, is the source paper's title.
Each set runs three times after one warm-up; limit 8, no MMR, as in My library. Excludes HTTP and the browser.
Run: uv run python -m scripts.benchmark_recommend"""
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from researchlens.decision import DecisionModel
from researchlens.mining import MiningEngine

ROOT = Path(__file__).resolve().parents[1]
SETS, REPEATS = 50, 3


def main():
    papers = [json.loads(line) for line in (ROOT / "data/papers.jsonl").read_text().splitlines() if line.strip()]
    by_id = {p["id"]: p for p in papers}
    engine = MiningEngine(papers)
    vectors = ROOT / "artifacts/specter.npz"
    plain = DecisionModel.load(ROOT / "artifacts/decision_model.pkl", vectors)
    topical = DecisionModel.load(ROOT / "artifacts/decision_query_model.pkl", vectors)
    tasks = json.loads((ROOT / "artifacts/temporal_recovery.json").read_text())["observations"][:SETS]
    runs = {"fusion": lambda t: engine.recommend(t["seeds"], 8, "fusion", 0),
            "cocitation": lambda t: engine.recommend(t["seeds"], 8, "cocitation", 0),
            "hybrid": lambda t: engine.recommend(t["seeds"], 8, "hybrid", 0),
            "decision": lambda t: engine.recommend(t["seeds"], 8, "decision", 0, plain),
            "decision_topic": lambda t: engine.recommend(t["seeds"], 8, "decision", 0, topical,
                                                         by_id[t["source"]]["title"])}
    summary = {}
    for name, run in runs.items():
        run(tasks[0])
        times = []
        for _ in range(REPEATS):
            for task in tasks:
                started = time.perf_counter()
                run(task)
                times.append(1000 * (time.perf_counter() - started))
        summary[name] = dict(runs=len(times), p50_ms=float(np.percentile(times, 50)), p95_ms=float(np.percentile(times, 95)))
        print(name, {k: round(v, 1) for k, v in summary[name].items()}, flush=True)
    (ROOT / "artifacts/latency_recommend.json").write_text(json.dumps(dict(
        scope="warm in-process engine.recommend, limit 8, no MMR; excludes HTTP and browser",
        machine=platform.processor() or platform.machine(), load_average_1min=os.getloadavg()[0], papers=len(papers),
        summary=summary), indent=2))


if __name__ == "__main__":
    main()
