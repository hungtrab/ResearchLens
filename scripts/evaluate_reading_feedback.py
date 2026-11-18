"""Evaluate a single positive reading signal without exposing any unseen judgment."""

import argparse
import json
from pathlib import Path
import hashlib

import numpy as np
from build_passage_index import encode, load_policy
from decision_features import DATA_REV, DEST, DecisionFeatures
from evaluate_passages import FILES, source
from researchlens.catalog import title_key
from researchlens.passages import split_sentences


def next_pages(original, clicked_scores, first):
    """Rank previously unseen IDs for the same click under three fixed rules."""
    candidates = np.ones(len(original), dtype=bool)
    candidates[first] = False
    indices = np.flatnonzero(candidates)
    score_sets = {
        "continue": original,
        "clicked_only": clicked_scores,
        "centroid": (original + clicked_scores) / 2,
    }
    return {
        name: indices[np.argsort(-values[indices], kind="stable")[:3]]
        for name, values in score_sets.items()
    }


def summarize(rows):
    by_arm = {
        m: sum(r["next_hits"][m] for r in rows)
        for m in ["continue", "clicked_only", "centroid"]
    }
    return dict(
        tasks=len(rows),
        distinct_papers=len({r["query"] for r in rows}),
        slots=len(rows) * 3,
        hits=by_arm,
    )


def run(cache):
    cache.mkdir(parents=True, exist_ok=True)
    FILES["test-pid2anns-csfcube-background.json"] = (
        "2015cbf965377374e020b0ec133351ec1feb731b9a9c40da0e446ef6eee4fa5e"
    )
    papers = {
        p["paper_id"]: p
        for p in map(
            json.loads, source(cache, "abstracts-csfcube-preds.jsonl").splitlines()
        )
    }
    splits = json.loads(source(cache, "evaluation_splits.json"))
    excluded = {k.split("_")[0] for k in splits["method"]["fold1_dev"]}
    enc = DecisionFeatures(include_laya=False)
    policy, policy_sha = load_policy()
    w = np.array(policy["weights"]) / policy["scale"]
    ids = sorted(papers)
    ix = {pid: i for i, pid in enumerate(ids)}
    documents = [
        papers[pid]["title"]
        + ". "
        + " ".join(split_sentences(" ".join(papers[pid]["abstract"])))
        for pid in ids
    ]
    docs = encode(enc, documents).astype(np.float32)
    lexical = enc.lex.transform(documents)
    rows = []
    for f in ("method", "result", "background"):
        judged = json.loads(source(cache, f"test-pid2anns-csfcube-{f}.json"))
        for key in sorted(splits[f]["fold1_dev"] + splits[f]["fold1_test"]):
            q = key.split("_")[0]
            if q in excluded:
                continue
            p = papers[q]
            tagged = [
                s
                for s, tag in zip(p["abstract"], p["pred_labels"], strict=True)
                if tag == f + "_label"
            ]
            chosen = split_sentences((tagged or p["abstract"])[0])[0]
            parts = split_sentences(" ".join(p["abstract"]))
            chosen = next((s for s in parts if chosen in s or s in chosen), parts[0])
            text = chosen + " " + p["title"]
            pool = []
            grades = []
            seen = {title_key(p["title"])}
            for pid, g in sorted(
                zip(judged[q]["cands"], judged[q]["relevance_adju"], strict=True)
            ):
                name = title_key(papers[pid]["title"])
                if name in seen or not " ".join(papers[pid]["abstract"]).strip():
                    continue
                seen.add(name)
                pool.append(pid)
                grades.append(g)
            which = [ix[pid] for pid in pool]
            lex = lexical[which]
            sem = docs[which]
            distances = w[0] * (
                lex @ enc.lex.transform([text]).T
            ).toarray().ravel() + w[1] * (
                sem @ encode(enc, [text])[0].astype(np.float32)
            )
            first = np.argsort(-distances, kind="stable")[:3]
            click = next((i for i in first if grades[i] >= 2), None)
            clicked = (
                (
                    w[0] * (lex @ lex[click].T).toarray().ravel()
                    + w[1] * (sem @ sem[click])
                )
                if click is not None
                else distances
            )
            choices = next_pages(distances, clicked, first)
            rows.append(
                dict(
                    query=q,
                    facet=f,
                    first=[pool[i] for i in first],
                    click=pool[click] if click is not None else None,
                    pool=list(zip(pool, grades)),
                    next={
                        m: [pool[i] for i in positions]
                        for m, positions in choices.items()
                    },
                    next_hits={
                        m: sum(grades[i] >= 2 for i in positions)
                        for m, positions in choices.items()
                    },
                )
            )
    clicked = [r for r in rows if r["click"]]
    result = dict(
        all=summarize(rows),
        clicked=summarize(clicked),
        no_click=summarize([r for r in rows if not r["click"]]),
        by_facet={
            f: summarize([r for r in rows if r["facet"] == f])
            for f in ["method", "result", "background"]
        },
    )
    for name, subset in [("all", rows), ("clicked", clicked)]:
        blocks = []
        for q in sorted({r["query"] for r in subset}):
            group = [r for r in subset if r["query"] == q]
            blocks.append(
                [
                    sum(
                        r["next_hits"]["centroid"] - r["next_hits"]["continue"]
                        for r in group
                    ),
                    3 * len(group),
                ]
            )
        boot = np.asarray(blocks)[
            np.random.default_rng(20261118).integers(
                len(blocks), size=(10000, len(blocks))
            )
        ].sum(1)
        result[name]["delta_p3_ci"] = np.quantile(
            boot[:, 0] / boot[:, 1], [0.025, 0.975]
        ).tolist()
    return dict(
        dataset_revision=DATA_REV,
        script_sha256={
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("evaluate_reading_feedback.py", "build_passage_index.py")
        },
        protocol_sha=hashlib.sha256(
            (DEST / "feedback_protocol.md").read_bytes()
        ).hexdigest(),
        policy_sha=policy_sha,
        summary=result,
        deploy=result["all"]["hits"]["centroid"] > result["all"]["hits"]["continue"],
        rows=rows,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    result = run(args.cache)
    (DEST / "feedback_comparison.json").write_text(json.dumps(result, allow_nan=False))
    print(json.dumps(result["summary"], indent=2))
