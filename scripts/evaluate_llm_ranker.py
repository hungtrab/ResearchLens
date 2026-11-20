"""A large language model behind an API ranks papers: the LLM + RAG baseline against the decision model.

Same retrieval and tasks as evaluate_slm_ranker.py: fusion's top 20 candidates on the frozen E2, E2b and E3 tasks.
With --graph (graph RAG), each candidate also carries the citation evidence retrieved for it: papers citing it
together with a saved paper, citation links with the saved papers, and its citation count in the collection.
Listwise, as in RankGPT (Sun et al., 2023): one call per task shows the saved papers' titles and the 20 numbered
candidates, and asks for all 20 numbers in order. Candidates the answer leaves out keep fusion's order after the
ones it names. Answers are cached in artifacts/llm_ranker_cache.jsonl, so a rerun does not call the API again.
Pre-registered: the LLM beats the decision model only if its nDCG@10 is higher with a paired 95% CI excluding 0 on
at least two of the three suites.
Needs an OpenAI-compatible endpoint: LLM_BASE_URL (ending in /v1), LLM_API_KEY, LLM_MODEL.
Run: uv run python -m scripts.evaluate_llm_ranker [--graph] [--limit N] [--workers 4]"""
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
MAX_TOKENS = 16384  # GLM reasons first and reasoning counts toward the limit; at 8192, 31% of graph prompts ran out


def prompt(job, graph=False):
    """graph=True is the graph RAG variant: each candidate also carries its retrieved citation evidence."""
    library = "\n".join(f"- {t}" for t in job["titles"])
    lines = [f"{c}\n    Citation evidence: {e}" if graph else c for c, e in zip(job["candidates"], job["graph"])]
    candidates = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(lines))
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
    parser.add_argument("--graph", action="store_true", help="add each candidate's citation evidence to the prompt")
    args = parser.parse_args()
    base, key, model = (os.environ[k] for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"))
    jobs = retrieval_jobs(args.limit)
    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            row = json.loads(line)
            cache[(row["model"], row["prompt_sha256"])] = row
    lock = threading.Lock()
    client = httpx.Client(base_url=base.rstrip("/"), headers={"Authorization": f"Bearer {key}"}, timeout=600)  # the slowest answer so far took 430 s

    def ask(job):
        text = prompt(job, args.graph)
        digest = hashlib.sha256((SYSTEM + text).encode()).hexdigest()
        cached = cache.get((model, digest))
        if cached and order_from(cached["answer"], POOL)[1]:  # an empty answer was cut off mid-reasoning: ask again
            return cached
        for attempt in range(8):  # the shared endpoint is sometimes overloaded for minutes
            started = time.perf_counter()
            answer, usage = [], None
            try:  # streamed, so the gateway does not time out (HTTP 504) while the model reasons for minutes
                with client.stream("POST", "/chat/completions", json=dict(
                        model=model, temperature=0, max_tokens=MAX_TOKENS, stream=True,
                        stream_options={"include_usage": True},
                        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}])) as response:
                    if response.status_code in (429, 500, 502, 503, 504):
                        print(f"attempt {attempt}: HTTP {response.status_code}", flush=True)
                        time.sleep(min(2 ** attempt * 10, 120))
                        continue
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        chunk = json.loads(line[6:])
                        usage = chunk.get("usage") or usage
                        for choice in chunk.get("choices", []):
                            answer.append(choice.get("delta", {}).get("content") or "")
            except httpx.TransportError as error:
                print(f"attempt {attempt}: {type(error).__name__}", flush=True)
                time.sleep(min(2 ** attempt * 10, 120))
                continue
            row = dict(model=model, prompt_sha256=digest, answer="".join(answer),
                       seconds=time.perf_counter() - started, usage=usage, max_tokens=MAX_TOKENS)
            with lock:
                with CACHE.open("a") as out:
                    out.write(json.dumps(row) + "\n")
                print(time.strftime("%H:%M"), "new answer", f"{row['seconds']:.0f}s",
                      "named" if order_from(row["answer"], POOL)[1] else "EMPTY", flush=True)
            return row
        raise RuntimeError("API kept failing after 8 attempts")

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
    result = dict(graph_evidence=args.graph, rule="LLM beats decision model if nDCG@10 delta CI95 > 0 on at least 2 of 3 suites",
                  model=model, pool=POOL, seeds_shown=SEEDS_SHOWN, system=SYSTEM, baselines=baselines(jobs),
                  suites=per_suite, beats_decision=sum(s["minus_decision"]["ci95"][0] > 0 for s in per_suite.values()) >= 2,
                  answers_naming_all=float(np.mean([r["named"] == POOL for r in rows])),
                  seconds_per_request=dict(p50=float(np.percentile(seconds, 50)), p95=float(np.percentile(seconds, 95))),
                  tokens=dict(prompt=sum(t.get("prompt_tokens", 0) for t in tokens),
                              completion=sum(t.get("completion_tokens", 0) for t in tokens)),
                  rows=rows)
    name = "llm_ranker_graph" if args.graph else "llm_ranker"
    out = ROOT / f"artifacts/{name}{'_smoke' if args.limit else ''}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
