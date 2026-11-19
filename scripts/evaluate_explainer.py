"""Speed and grounding of the SLM + RAG explanations, for three sizes of Qwen2.5 (GGUF, Q4_K_M, CPU).

Round 3: the model summarizes only the recommended paper's abstract; the app states how it links to the saved
papers. Rounds 1 and 2 (artifacts/explainer_round1.json, _round2.json) used E2 tasks 0-29; this round uses tasks
30-59 so the prompt is not tuned to the sentences we read. The 3B model is left out: its round-2 median was
7.3 s. Each library gets the decision model's top 3 on the full corpus, so 90 explanations per model. first_token
times a one-token answer on the first 10 prompts, which is reading the prompt. Pre-registered choice: the default model is the one with the highest
share of answers that pass the check, among models with median generation time at most 3 s; ties go to the
faster model. Run: uv run --extra slm python -m scripts.evaluate_explainer"""
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from researchlens.decision import DecisionModel
from researchlens.explain import Explainer, passages
from researchlens.mining import MiningEngine

ROOT = Path(__file__).resolve().parents[1]
MODELS = sorted(p for p in glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-*-Instruct-GGUF/snapshots/*/*.gguf"))
                if "-3b-" not in p)
TASKS, TOP, BUDGET = slice(30, 60), 3, 3.0


def percentiles(values):
    return dict(p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)))


def main():
    papers = [json.loads(line) for line in (ROOT / "data/papers.jsonl").read_text().splitlines() if line.strip()]
    engine = MiningEngine(papers)
    decision = DecisionModel.load(ROOT / "artifacts/decision_model.pkl", ROOT / "artifacts/specter.npz")
    tasks = json.loads((ROOT / "artifacts/temporal_recovery.json").read_text())["observations"][TASKS]
    jobs, retrieval = [], []
    for task in tasks:
        items = engine.recommend(task["seeds"], TOP, "decision", 0, decision)["items"]
        for item in items:
            started = time.perf_counter()
            evidence, fact = passages(engine, task["seeds"], item["id"], decision.vectors(engine))
            retrieval.append(time.perf_counter() - started)
            jobs.append(dict(source=task["source"], paper_id=item["id"], evidence=evidence, fact=fact,
                             title=item["title"]))
    results, load_average = {}, [os.getloadavg()[0]]
    for path in MODELS:
        started = time.perf_counter()
        explainer = Explainer.load(path)
        load = time.perf_counter() - started
        explainer.explain(jobs[0]["evidence"], jobs[0]["fact"], jobs[0]["title"])  # warm-up, not counted
        outputs = [dict(job, **{k: v for k, v in explainer.explain(job["evidence"], job["fact"], job["title"]).items()
                                if k not in ("evidence", "fact")}) for job in jobs]
        del explainer
        one_token = Explainer.load(path, max_tokens=1)
        first = []
        for job in jobs[:10]:
            first.append(one_token.explain(job["evidence"], job["fact"], job["title"])["seconds"])
        del one_token
        load_average.append(os.getloadavg()[0])
        checks = [o["checks"] for o in outputs]
        name = Path(path).name
        results[name] = dict(
            load_seconds=load, generation_seconds=percentiles([o["seconds"] for o in outputs]),
            first_token_seconds=percentiles(first),
            words=float(np.mean([len(o["draft"].split()) for o in outputs])),
            **{k: float(np.mean([c[k] for c in checks])) for k in ("shown", "valid_citations", "cites_recommended",
                                                                 "grounded_numbers", "beyond_title", "novel_words")},
            outputs=outputs)
        print(name, f"load {load:.1f}s", {k: round(v, 3) if isinstance(v, float) else v
                                          for k, v in results[name].items() if k != "outputs"}, flush=True)
    fast = [n for n, r in results.items() if r["generation_seconds"]["p50"] <= BUDGET]
    chosen = max(fast, key=lambda n: (results[n]["shown"], -results[n]["generation_seconds"]["p50"]))
    out = dict(rule="highest shown rate with median generation <= 3 s; ties to the faster model",
               explanations=len(jobs), retrieval_seconds=percentiles(retrieval),
               load_average_1min=load_average, threads=8, chosen=chosen, models=results)
    (ROOT / "artifacts/explainer.json").write_text(json.dumps(out, indent=2))
    print("retrieval", percentiles(retrieval), "chosen", chosen)


if __name__ == "__main__":
    main()
