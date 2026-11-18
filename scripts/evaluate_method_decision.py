"""Apply one frozen Laya technical-match decision to the shared fusion shortlist."""

import argparse
import json
from pathlib import Path

import numpy as np
from evaluate_method_retrieval import STUDY, bootstrap, checksum
from method_datasets import acl_arc
from reading_decisions import MODEL, REVISION

QUESTION = {
    "type": "noul",
    "instructions": "Does the candidate describe a concrete technique, dataset, or experiment that matches the specific research method or task sought in the goal? Shared field or vocabulary alone is insufficient.",
}


def score_queries(model, rows, inputs, threshold):
    out = []
    for row in rows:
        goal, docs, grades = inputs[row["query"]]
        shortlist = row["fusion_top20"]
        values = []
        for key in shortlist:
            state = json.dumps(
                {"goal": goal, "candidate": docs[key]}, ensure_ascii=False
            )
            result = model.predict(state, {"match": QUESTION})
            value = float(result["answers"]["match"]["noul"])
            if not np.isfinite(value):
                raise ValueError("Invalid decision score")
            values.append(value)
        order = np.argsort(-np.array(values), kind="stable")
        first = [shortlist[i] for i in order[:3]]
        slots = sum(grades.get(k, 0) >= threshold for k in first)
        out.append(
            dict(
                query=row["query"],
                cluster=row["cluster"],
                scores=values,
                top3=first,
                slots3=slots,
                hit3=int(slots > 0),
                fusion_hit3=row["arms"]["fusion"]["hit3"],
            )
        )
    return out


def main():
    import laya
    import torch
    from huggingface_hub import snapshot_download
    from evaluate_passages import source
    from researchlens.passages import split_sentences

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--csfcube-cache", required=True, type=Path)
    args = parser.parse_args()
    parent = STUDY / "method_retrieval.json"
    runs = json.loads(parent.read_bytes())["queries"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = laya.load(snapshot_download(MODEL, revision=REVISION), device=device)
    docs, queries, _ = acl_arc(args.cache)
    inputs = {q.key: (q.text, docs, q.grades) for q in queries}
    rows = {"acl_arc": score_queries(model, runs["acl_arc"], inputs, 1)}
    papers = {
        r["paper_id"]: r
        for r in map(
            json.loads,
            source(args.csfcube_cache, "abstracts-csfcube-preds.jsonl").splitlines(),
        )
    }
    prev = json.loads((STUDY / "passage_comparison.json").read_bytes())
    inputs = {}
    for r in prev["queries"]["method"]:
        pid = r["query"]
        seed = papers[pid]
        sent = split_sentences(seed["abstract"][r["selected_source_sentence"]])[0]
        docs = {
            k: papers[k]["title"]
            + ". "
            + " ".join(split_sentences(" ".join(papers[k]["abstract"])))
            for k, _ in r["pool"]
        }
        inputs[pid] = sent + " " + seed["title"], docs, dict(r["pool"])
    rows["csfcube"] = score_queries(model, runs["csfcube"], inputs, 2)
    summaries = {}
    for name, values in rows.items():
        summaries[name] = dict(
            queries=len(values),
            pairs=sum(len(r["scores"]) for r in values),
            hit3=sum(r["hit3"] for r in values),
            slots3=sum(r["slots3"] for r in values),
            hit3_vs_fusion=bootstrap(
                [r["hit3"] - r["fusion_hit3"] for r in values],
                [r["cluster"] for r in values],
            ),
        )
    result = dict(
        model=[MODEL, REVISION],
        question=QUESTION,
        parent_sha256=checksum(parent),
        script_sha256=checksum(Path(__file__)),
        protocol_sha256=checksum(STUDY / "method_decision_protocol.md"),
        summary=summaries,
        queries=rows,
    )
    (STUDY / "method_decision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
