"""Select a sentence ranker on development topics and evaluate on separate query papers."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import ndcg_score
from build_passage_index import encode, load_policy
from contextual_rank import MODEL, REVISION, contextual_vectors, fused_scores, nearest
from decision_features import DATA_REV, DEST, DecisionFeatures
from evaluate_passages import FILES, source
from fit_reading_decision import score
from researchlens.catalog import title_key
from researchlens.passages import split_sentences

BACKGROUND = "test-pid2anns-csfcube-background.json"
WEIGHTS = (0, 0.25, 0.5, 0.75, 1)


def topics(splits, previous):
    method_dev = sorted(splits["method"]["fold1_dev"])
    seen = {k.split("_")[0] for k in method_dev}
    groups = {"dev": method_dev, "method": [], "result": [], "transfer": []}
    for facet, rows in previous["queries"].items():
        groups[facet] = sorted(r["query"] + "_" + facet for r in rows)
        seen.update(r["query"] for r in rows)
    for facet in ("background", "result"):
        for phase in ("fold1_dev", "fold1_test"):
            groups["transfer"].extend(
                sorted(k for k in splits[facet][phase] if k.split("_")[0] not in seen)
            )
    return groups


def metrics(grades, scores):
    order = np.argsort(-scores, kind="stable")
    return dict(
        hits=int((grades[order[:3]] >= 2).sum()),
        ndcg=float(ndcg_score([2**grades - 1], [scores], k=10)),
    )


def summary(rows, arm):
    return dict(
        queries=len(rows),
        pairs=sum(len(r["pool"]) for r in rows),
        slots=3 * len(rows),
        hits=sum(r["arms"][arm]["hits"] for r in rows),
        ndcg=float(np.mean([r["arms"][arm]["ndcg"] for r in rows])),
    )


def paired(rows, arm):
    groups = sorted({r["query"] for r in rows})
    deltas = []
    for g in groups:
        items = [r for r in rows if r["query"] == g]
        deltas.append(
            [
                sum(
                    (r["arms"][arm]["hits"] - r["arms"]["0"]["hits"]) / 3 for r in items
                ),
                sum(r["arms"][arm]["ndcg"] - r["arms"]["0"]["ndcg"] for r in items),
                len(items),
            ]
        )
    sampled = np.asarray(deltas)[
        np.random.default_rng(20261118).integers(len(groups), size=(10000, len(groups)))
    ].sum(1)
    ci = np.quantile(sampled[:, :2] / sampled[:, 2:], [0.025, 0.975], axis=0)
    return dict(
        clusters=len(groups),
        p3_delta_ci=ci[:, 0].tolist(),
        ndcg_delta_ci=ci[:, 1].tolist(),
    )


def run(cache, model_path):
    cache.mkdir(parents=True, exist_ok=True)
    FILES[BACKGROUND] = (
        "2015cbf965377374e020b0ec133351ec1feb731b9a9c40da0e446ef6eee4fa5e"
    )
    papers = {
        p["paper_id"]: p
        for p in map(
            json.loads, source(cache, "abstracts-csfcube-preds.jsonl").splitlines()
        )
    }
    splits = json.loads(source(cache, "evaluation_splits.json"))
    previous = json.loads((DEST / "passage_comparison.json").read_text())
    groups = topics(splits, previous)
    qrels = {
        f: json.loads(source(cache, f"test-pid2anns-csfcube-{f}.json"))
        for f in ("method", "background", "result")
    }
    records = {}
    queries = {}
    for keys in groups.values():
        for k in keys:
            q, f = k.split("_")
            p = papers[q]
            parts = split_sentences(" ".join(p["abstract"]))
            tagged = [
                s
                for s, t in zip(p["abstract"], p["pred_labels"], strict=True)
                if t == f + "_label"
            ]
            # The reader can only select a displayed sentence. Keep every eligible topic.
            selected = split_sentences(tagged[0])[0] if tagged else parts[0]
            ix = next(
                (i for i, s in enumerate(parts) if selected in s or s in selected), 0
            )
            records[q] = (p["title"], parts)
            seen = {title_key(p["title"])}
            pool = []
            judged = qrels[f][q]
            for pid, grade in sorted(
                zip(judged["cands"], judged["relevance_adju"], strict=True)
            ):
                paper = papers[pid]
                sentences = split_sentences(" ".join(paper["abstract"]))
                name = title_key(paper["title"])
                if name in seen or not sentences:
                    continue
                seen.add(name)
                records[pid] = (paper["title"], sentences)
                pool.append((pid, grade))
            queries[k] = (q, f, ix, pool)
    contextual = contextual_vectors(records, model_path)
    enc = DecisionFeatures(include_laya=False)
    policy, policy_sha = load_policy()
    ids = list(records)
    docs = [t + ". " + " ".join(parts) for t, parts in records.values()]
    vecs = encode(enc, docs).astype(np.float32)
    lexical = enc.lex.transform(docs)
    where = {pid: i for i, pid in enumerate(ids)}
    report = {}
    for split, keys in groups.items():
        report[split] = []
        for key in keys:
            q, f, ix, pool = queries[key]
            title, parts = records[q]
            context = parts[ix] + " " + title
            indices = [where[p] for p, g in pool]
            overlaps = (
                (lexical[indices] @ enc.lex.transform([context]).T).toarray().ravel()
            )
            sims = vecs[indices] @ encode(enc, [context])[0].astype(np.float32)
            old = score(
                [dict(lexical=l, semantic=s) for l, s in zip(overlaps, sims)], policy
            )
            asp = nearest(contextual[q][ix], [contextual[p] for p, g in pool])
            grades = np.array([g for p, g in pool])
            arms = {}
            for w in WEIGHTS:
                values = fused_scores(old, asp, w)
                arms[str(w)] = metrics(grades, values)
                arms[str(w)]["top3"] = [
                    pool[i][0] for i in np.argsort(-values, kind="stable")[:3]
                ]
            report[split].append(
                dict(
                    query=q,
                    facet=f,
                    sentence=ix,
                    pool=pool,
                    baseline=old.tolist(),
                    contextual=asp.tolist(),
                    arms=arms,
                )
            )
        if split == "dev":
            dev = {w: summary(report["dev"], str(w)) for w in WEIGHTS}
            selected = max(WEIGHTS, key=lambda w: (dev[w]["hits"], dev[w]["ndcg"], -w))
    arm = str(selected)
    overall = {
        name: {
            "baseline": summary(rows, "0"),
            "selected": summary(rows, arm),
            "aspire": summary(rows, "1"),
            "paired": paired(rows, arm),
        }
        for name, rows in report.items()
    }
    transfer = overall["transfer"]
    deploy = (
        transfer["selected"]["hits"] > transfer["baseline"]["hits"]
        and transfer["selected"]["ndcg"] > transfer["baseline"]["ndcg"]
        and overall["method"]["selected"]["hits"]
        >= overall["method"]["baseline"]["hits"]
    )
    return dict(
        dataset_revision=DATA_REV,
        script_sha256={
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "evaluate_contextual.py",
                "contextual_rank.py",
                "build_passage_index.py",
            )
        },
        model=MODEL,
        revision=REVISION,
        policy_sha=policy_sha,
        protocol_sha256=hashlib.sha256(
            (DEST / "context_protocol.md").read_bytes()
        ).hexdigest(),
        selected_weight=selected,
        deploy=deploy,
        dev_grid=dev,
        summary=overall,
        transfer_by_facet={
            f: {
                a: summary([r for r in report["transfer"] if r["facet"] == f], w)
                for a, w in [("selected", arm), ("baseline", "0")]
            }
            for f in ["background", "result"]
        },
        rows=report,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--model-path", type=Path)
    args = p.parse_args()
    if args.model_path:
        from contextual_rank import verify_checkpoint

        verify_checkpoint(args.model_path)
    result = run(args.cache, args.model_path)
    (DEST / "context_comparison.json").write_text(json.dumps(result, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
