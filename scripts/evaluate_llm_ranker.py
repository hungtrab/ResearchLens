"""A large language model behind an API ranks papers: the LLM + RAG baseline against the decision model.

Same retrieval and tasks as evaluate_slm_ranker.py: fusion's top 20 candidates on the frozen E2, E2b and E3 tasks.
Listwise, as in RankGPT (Sun et al., 2023): one call per task shows the saved papers' titles and the 20 numbered
candidates, and asks for all 20 numbers in order. Candidates the answer leaves out keep fusion's order after the
ones it names. Answers are cached in artifacts/llm_ranker_cache.jsonl, so a rerun does not call the API again.
Pre-registered: the LLM beats the decision model only if its nDCG@10 is higher with a paired 95% CI excluding 0 on
at least two of the three suites.
Needs an OpenAI-compatible endpoint: LLM_BASE_URL (ending in /v1), LLM_API_KEY, LLM_MODEL.
Run: uv run python -m scripts.evaluate_llm_ranker [--limit N] [--workers 4]"""
import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

from scripts.evaluate_slm_ranker import POOL, SEEDS_SHOWN, baselines, compare, ndcg, retrieval_jobs

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts/llm_ranker_cache.jsonl"
SYSTEM = "You help a researcher decide which paper to read next."


def prompt(job):
    library = "\n".join(f"- {t}" for t in job["titles"])
    candidates = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(job["candidates"]))
    return (f"The researcher has saved these papers:\n{library}\n\nCandidate papers:\n{candidates}\n\n"
            f"Rank all {len(job['candidates'])} candidates by how useful each is for this researcher to read next, "
            "most useful first. Answer only with the numbers, like: [3] > [1] > [2]")


def order_from(answer, n):
    """Indices named in the answer, first mention wins, then the rest in fusion order."""
    named = list(dict.fromkeys(int(k) - 1 for k in re.findall(r"\[?(\d+)\]?", answer) if 1 <= int(k) <= n))
    return named + [i for i in range(n) if i not in named], len(named)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="first N tasks per suite, for a smoke run")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    base, key, model = (os.environ[k] for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"))
    jobs = retrieval_jobs(args.limit)
    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            row = json.loads(line)
            cache[(row["model"], row["prompt_sha256"])] = row
    lock = threading.Lock()
    client = httpx.Client(base_url=base.rstrip("/"), headers={"Authorization": f"Bearer {key}"}, timeout=900)

    def ask(job):
        text = prompt(job)
        digest = hashlib.sha256((SYSTEM + text).encode()).hexdigest()
        if (model, digest) in cache:
            return cache[(model, digest)]
        for attempt in range(5):
            started = time.perf_counter()
            try:
                response = client.post("/chat/completions", json=dict(
                    model=model, temperature=0, max_tokens=8192,  # GLM reasons first; reasoning counts toward the limit
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]))
            except httpx.TransportError:
                time.sleep(2 ** attempt * 5)
                continue
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt * 5)
                continue
            response.raise_for_status()
            body = response.json()
            row = dict(model=model, prompt_sha256=digest, answer=body["choices"][0]["message"]["content"] or "",
                       seconds=time.perf_counter() - started, usage=body.get("usage"))
            with lock:
                with CACHE.open("a") as out:
                    out.write(json.dumps(row) + "\n")
            return row
        raise RuntimeError("API kept failing after 5 attempts")

    with ThreadPoolExecutor(args.workers) as pool:
        answers = list(pool.map(ask, jobs))
    rows = []
    for job, answer in zip(jobs, answers):
        indices, named = order_from(answer["answer"], len(job["pool"]))
        order = [job["pool"][i] for i in indices]
        rows.append(dict(suite=job["suite"], llm=ndcg(order, job["targets"]),
                         llm_hit=float(bool(set(order[:10]) & set(job["targets"]))), named=named,
                         decision=job["decision"], fusion=job["fusion"], seconds=answer["seconds"]))
    per_suite = compare(rows, "llm")
    seconds = [r["seconds"] for r in rows]
    tokens = [a["usage"] for a in answers if a.get("usage")]
    result = dict(rule="LLM beats decision model if nDCG@10 delta CI95 > 0 on at least 2 of 3 suites",
                  model=model, pool=POOL, seeds_shown=SEEDS_SHOWN, system=SYSTEM, baselines=baselines(jobs),
                  suites=per_suite, beats_decision=sum(s["minus_decision"]["ci95"][0] > 0 for s in per_suite.values()) >= 2,
                  answers_naming_all=float(np.mean([r["named"] == POOL for r in rows])),
                  seconds_per_request=dict(p50=float(np.percentile(seconds, 50)), p95=float(np.percentile(seconds, 95))),
                  tokens=dict(prompt=sum(t.get("prompt_tokens", 0) for t in tokens),
                              completion=sum(t.get("completion_tokens", 0) for t in tokens)),
                  rows=rows)
    out = ROOT / ("artifacts/llm_ranker.json" if not args.limit else "artifacts/llm_ranker_smoke.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
