"""Can a small language model rank papers instead of the decision model? The SLM + RAG baseline.

Retrieval: fusion's top 20 candidates. Ranking: the model reads the saved papers' titles (up to 30) and one
candidate's title and abstract, and is asked whether the reader should read it next; the score is the logit of
"Yes" minus "No". The saved papers come first in the prompt, so llama.cpp reuses them across the 20 candidates.
Same frozen tasks as decision_model.py: E2, E2b and E3. Decision model and fusion are rerun here on the same
graphs, so the deltas are paired per task. The decision model is scored twice: on the same 20 candidates
("decision_pool", the fair comparison) and on its own candidates as the app runs it ("decision"), where about a
third of its top 10 lies outside fusion's 20.
Pre-registered: the SLM ranker beats the decision model only if its nDCG@10 is higher with a paired 95% CI
excluding 0 on at least two of the three suites.
Run: uv run --extra slm python -m scripts.evaluate_slm_ranker [--models 0.5b 1.5b] [--limit N]
--rescore recomputes the baselines and deltas of slm_ranker.json from its stored per-task scores."""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from researchlens.decision import DecisionModel, citation_links
from researchlens.evaluation import bootstrap_mean, ranking_metrics
from researchlens.explain import describe
from researchlens.mining import MiningEngine
from scripts.evaluate_author_next import year

ROOT = Path(__file__).resolve().parents[1]
POOL, SEEDS_SHOWN, THREADS = 20, 30, 8
BASELINES = ("decision_pool", "decision", "fusion")  # decision_pool: the decision model on the same 20 candidates
SYSTEM = "You help a researcher decide which paper to read next."
RULE = "beats the decision model if nDCG@10 delta CI95 > 0 on at least 2 of 3 suites"


class Ranker:
    """Scores one candidate at a time with a Yes/No question; the shared prefix stays in the KV cache."""

    def __init__(self, path):
        from llama_cpp import Llama
        self.llm = Llama(str(path), n_ctx=4096, n_threads=THREADS, n_threads_batch=THREADS, verbose=False)
        self.yes, self.no = (self.llm.tokenize(w.encode(), add_bos=False)[0] for w in ("Yes", "No"))
        self.name = Path(path).name

    def score(self, saved_titles, candidate):
        """The library part is read once and kept; each candidate is read right after it, so a candidate's score
        does not depend on which candidate came before (reusing a longer shared prefix changed logits by ~0.1)."""
        library = "\n".join(f"- {t}" for t in saved_titles)
        tokenize = lambda s: self.llm.tokenize(s.encode(), add_bos=False, special=True)
        prefix = tokenize(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
                          f"The researcher has saved these papers:\n{library}\n\nCandidate paper:")
        suffix = tokenize(f" {candidate}\n\nShould the researcher read the candidate paper next? "
                          "Answer Yes or No.<|im_end|>\n<|im_start|>assistant\n")
        llm = self.llm
        if llm.n_tokens < len(prefix) or llm.input_ids[:len(prefix)].tolist() != prefix:
            llm.n_tokens = 0
            llm.eval(prefix)
        llm.n_tokens = len(prefix)
        llm.eval(suffix)  # drops the cache after the prefix, then reads the candidate
        logits = np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm.n_vocab(),))
        return float(logits[self.yes] - logits[self.no])


def suites(papers, without_e3, split):
    """(name, engine, tasks) with tasks as (seeds, targets, paper to leave out)."""
    e2 = json.loads((ROOT / "artifacts/temporal_recovery.json").read_text())["observations"]
    e2b = json.loads((ROOT / "artifacts/author_next.json").read_text())["observations"]
    cut = MiningEngine([p for p in papers if 0 < year(p) <= 2020])
    full = MiningEngine([p for p in without_e3 if year(p) > 0])
    return [("E2", cut, [(t["seeds"], t["targets"], None) for t in e2]),
            ("E2b", cut, [(t["seeds"], t["targets"], None) for t in e2b]),
            ("E3", full, [(t["seeds"], t["relevant"], t["paper_id"]) for t in split])]


def graph_evidence(pid, seeds, refs, citers, votes):
    """What the citation graph says about one candidate, in words, for the LLM + graph RAG prompt."""
    saved = set(seeds)
    return (f"{len(votes.get(pid, []))} papers cite it together with a saved paper; "
            f"{sum(pid in refs.get(s, ()) for s in saved)} saved papers cite it; "
            f"it cites {sum(pid in citers.get(s, ()) for s in saved)} saved papers; "
            f"{len(citers.get(pid, ()))} papers in the collection cite it.")


def ndcg(order, targets):
    return ranking_metrics(order[:10], set(targets))["ndcg_at_10"]


def summary(rows, key):
    values = np.array([r[key] for r in rows])
    return dict(ndcg=float(values.mean()), hit10=float(np.mean([r[key + "_hit"] for r in rows])))


def retrieval_jobs(limit=None):
    """Every frozen task with fusion's top-20 pool and the decision model and fusion scores on it."""
    papers =[json.loads(line) for line in (ROOT / "data/papers.jsonl").read_text().splitlines() if line.strip()]
    split = [json.loads(line) for line in (ROOT / "artifacts/recommendation_split.jsonl").open() if line.strip()]
    hidden = {(t["paper_id"], r) for t in split for r in t["relevant"]}
    without_e3 = [dict(p, references=[r for r in p.get("references", []) if (p["id"], r) not in hidden])
                  for p in papers]
    decision = DecisionModel.load(ROOT / "artifacts/decision_model.pkl", ROOT / "artifacts/specter.npz")

    # Retrieval and the two baselines, once.
    jobs = []
    for name, engine, tasks in suites(papers, without_e3, split):
        refs, citers = citation_links(engine)
        for seeds, targets, source in tasks[:limit]:
            drop = lambda items: [i["id"] for i in items if i["id"] != source]
            pool = drop(engine.recommend(seeds, POOL + 1, "fusion", 0)["items"])[:POOL]
            ranked = drop(engine.recommend(seeds, 11, "decision", 0, decision)["items"])
            probability = dict(zip(*decision.rank(engine, seeds)[:2]))  # every fusion top-20 paper is scored
            on_pool = sorted(pool, key=lambda pid: -probability[pid])
            titles = [engine.by_id[s]["title"] for s in sorted(seeds)[:SEEDS_SHOWN]]
            votes = engine.co_citation_witnesses(seeds)
            hit = lambda order: float(bool(set(order[:10]) & set(targets)))
            jobs.append(dict(suite=name, seeds=seeds, targets=sorted(targets), pool=pool, titles=titles,
                             candidates=[describe(engine.by_id[pid], 40) for pid in pool],
                             graph=[graph_evidence(pid, seeds, refs, citers, votes) for pid in pool],
                             fusion=ndcg(pool, targets), fusion_hit=hit(pool),
                             decision=ndcg(ranked, targets), decision_hit=hit(ranked),
                             decision_pool=ndcg(on_pool, targets), decision_pool_hit=hit(on_pool),
                             pool_recall=len(set(pool) & set(targets)) / len(targets)))
        print(name, "retrieval done", flush=True)
    return jobs


def baselines(jobs):
    out = {}
    for name in ("E2", "E2b", "E3"):
        sub = [j for j in jobs if j["suite"] == name]
        if sub:
            out[name] = dict(tasks=len(sub), pool_recall=float(np.mean([j["pool_recall"] for j in sub])),
                             **{key: summary(sub, key) for key in BASELINES})
    return out


def compare(rows, key):
    """Per suite: the ranker's nDCG@10 and Hit@10 and its paired deltas vs each baseline."""
    per_suite = {}
    for name in ("E2", "E2b", "E3"):
        sub = [r for r in rows if r["suite"] == name]
        if sub:
            per_suite[name] = dict(tasks=len(sub), **summary(sub, key), **{
                f"minus_{base}": dict(mean=float(np.mean([r[key] - r[base] for r in sub])),
                                      ci95=bootstrap_mean([r[key] - r[base] for r in sub]))
                for base in BASELINES})
    return per_suite


def beats(per_suite):
    """The pre-registered rule, against the decision model on the same candidates and as the app runs it."""
    return {base: sum(s[f"minus_{base}"]["ci95"][0] > 0 for s in per_suite.values()) >= 2
            for base in ("decision_pool", "decision")}


def with_baselines(rows, jobs):
    """Per-task rows with every baseline's score on the same task (rows and jobs are in the same order)."""
    assert len(rows) == len(jobs) and all(r["suite"] == j["suite"] for r, j in zip(rows, jobs))
    return [dict(r, **{k: j[k] for k in BASELINES}) for r, j in zip(rows, jobs)]


def rescore(jobs):
    """Recompute slm_ranker.json's baselines and deltas without rerunning the language models."""
    path = ROOT / "artifacts/slm_ranker.json"
    result = json.loads(path.read_text())
    for model in result["models"].values():
        model["rows"] = with_baselines(model["rows"], jobs)
        model["suites"] = compare(model["rows"], "slm")
        model["beats_decision"] = beats(model["suites"])
    result.update(rule=RULE, baselines=baselines(jobs))
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({n: m["suites"] for n, m in result["models"].items()}, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["0.5b", "1.5b", "3b"])
    parser.add_argument("--limit", type=int, help="first N tasks per suite, for a smoke run")
    parser.add_argument("--rescore", action="store_true", help="recompute baselines from stored per-task scores")
    args = parser.parse_args()
    jobs = retrieval_jobs(args.limit)
    if args.rescore:
        return rescore(jobs)
    models = {}
    paths = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-*-Instruct-GGUF/snapshots/*/*.gguf"))
    for size in args.models:
        path = next(p for p in paths if f"-{size}-" in p)
        ranker = Ranker(path)
        ranker.score(jobs[0]["titles"], jobs[0]["candidates"][0])  # warm-up
        seconds, rows, load = [], [], [os.getloadavg()[0]]
        for k, job in enumerate(jobs):
            started = time.perf_counter()
            scores = [ranker.score(job["titles"], c) for c in job["candidates"]]
            seconds.append(time.perf_counter() - started)
            order = [job["pool"][i] for i in np.argsort(-np.array(scores), kind="stable")]
            rows.append(dict(suite=job["suite"], slm=ndcg(order, job["targets"]),
                             slm_hit=float(bool(set(order[:10]) & set(job["targets"])))))
            if k % 50 == 0:
                print(size, k, f"{np.median(seconds):.2f}s/task", flush=True)
        load.append(os.getloadavg()[0])
        rows = with_baselines(rows, jobs)
        per_suite = compare(rows, "slm")
        models[ranker.name] = dict(suites=per_suite, beats_decision=beats(per_suite),
                                   seconds_per_request=dict(p50=float(np.percentile(seconds, 50)),
                                                            p95=float(np.percentile(seconds, 95))),
                                   load_average_1min=load, rows=rows)
        print(ranker.name, json.dumps(per_suite, indent=1), flush=True)
        del ranker

    result = dict(rule=RULE,
                  pool=POOL, seeds_shown=SEEDS_SHOWN, threads=THREADS, system=SYSTEM, baselines=baselines(jobs),
                  models=models)
    out = ROOT / ("artifacts/slm_ranker.json" if not args.limit else "artifacts/slm_ranker_smoke.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["baselines"], indent=1))


if __name__ == "__main__":
    main()
