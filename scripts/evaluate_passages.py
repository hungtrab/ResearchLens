"""Paired comparison of sentence-guided reading suggestions in expert-judged pools."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.metrics import ndcg_score
from build_passage_index import build_arrays, encode, load_policy
from decision_features import CSFCUBE_FILES, DATA_REV, DEST, DecisionFeatures
from fit_reading_decision import score
from researchlens.catalog import title_key
from researchlens.passages import passage_scores, split_sentences

RESULT_FILE = "test-pid2anns-csfcube-result.json"
FILES = {
    **CSFCUBE_FILES,
    RESULT_FILE: "5ed7a63bc20e3deb725f8ec82abfcb1c5d79673befc942509b48effceb3be031",
}


def source(cache, name):
    path = cache / name
    if not path.exists():
        url = f"https://raw.githubusercontent.com/iesl/CSFCube/{DATA_REV}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FILES[name]:
        raise ValueError(f"Unexpected source for {name}")
    return raw


def summarize(scores):
    methods = scores[0]["arms"]
    out = {}
    for m in methods:
        runs = [r["arms"][m] for r in scores]
        hits = sum(s["useful_in_three"] for s in runs)
        out[m] = dict(
            relevant_slots=hits,
            slots=3 * len(runs),
            precision_at_3=hits / (3 * len(runs)),
            has_useful=sum(s["useful_in_three"] > 0 for s in runs),
            ndcg_at_10=float(np.mean([s["ndcg_at_10"] for s in runs])),
        )
    delta = np.array(
        [
            (
                r["arms"]["passage_mix"]["useful_in_three"]
                - r["arms"]["text_policy"]["useful_in_three"]
            )
            / 3
            for r in scores
        ]
    )
    interval = np.quantile(
        np.random.default_rng(20261118).choice(delta, (10000, len(delta))).mean(1),
        [0.025, 0.975],
    )
    return dict(
        queries=len(scores),
        pairs=sum(len(r["pool"]) for r in scores),
        arms=out,
        paired_p3_vs_text=dict(
            delta=float(delta.mean()), bootstrap_95=interval.tolist()
        ),
    )


def evaluate(cache):
    cache.mkdir(exist_ok=True, parents=True)
    papers = {
        r["paper_id"]: r
        for r in map(
            json.loads, source(cache, "abstracts-csfcube-preds.jsonl").splitlines()
        )
    }
    splits = json.loads(source(cache, "evaluation_splits.json"))
    method_ids = {
        key.split("_")[0]
        for split in ["fold1_dev", "fold1_test"]
        for key in splits["method"][split]
    }
    encoder = DecisionFeatures(include_laya=False)
    policy, policy_sha = load_policy()
    previous = json.loads((DEST / "decision_comparison.json").read_text())
    old_scores = dict(
        zip(
            map(tuple, previous["prediction_order"]["external_test"]),
            previous["predictions"]["external_test"]["text_only"],
            strict=True,
        )
    )
    grouped = {}
    for facet in ("method", "result"):
        qrels = json.loads(source(cache, f"test-pid2anns-csfcube-{facet}.json"))
        grouped[facet] = []
        for key in sorted(splits[facet]["fold1_test"]):
            q = key.split("_")[0]
            if facet == "result" and q in method_ids:
                continue
            paper = papers[q]
            # Match a reader picking one sentence. Automatic tags are provided with the dataset.
            snippet = next(
                s
                for s, label in zip(
                    paper["abstract"], paper["pred_labels"], strict=True
                )
                if label == facet + "_label"
            )
            chosen = split_sentences(snippet)[0]
            assert chosen in split_sentences(" ".join(paper["abstract"]))
            seen = {title_key(paper["title"])}
            pool = []
            for pid, grade in sorted(
                zip(qrels[q]["cands"], qrels[q]["relevance_adju"], strict=True)
            ):
                candidate = papers[pid]
                parts = split_sentences(" ".join(candidate["abstract"]))
                name = title_key(candidate["title"])
                if not parts or name in seen:
                    continue
                seen.add(name)
                pool.append((pid, grade, candidate["title"], parts))
            documents, sentences, offsets, _ = build_arrays(
                encoder, [(t, ss) for _, _, t, ss in pool]
            )
            documents = documents.astype(np.float32)
            sentences = sentences.astype(np.float32)
            query = encode(encoder, [chosen])[0].astype(np.float32)
            combined, local = passage_scores(query, documents, sentences, offsets)
            contextual = chosen + " " + paper["title"]
            context_vector = encode(encoder, [contextual])[0].astype(np.float32)
            texts = [t + ". " + " ".join(ss) for _, _, t, ss in pool]
            vocab = encoder.lex.transform([contextual] + texts)
            lex = (vocab[1:] @ vocab[0].T).toarray().ravel()
            baseline = score(
                [
                    dict(lexical=l, semantic=s)
                    for l, s in zip(lex, documents @ context_vector)
                ],
                policy,
            )
            arms = dict(
                text_policy=baseline,
                document_only=documents @ query,
                best_sentence=np.maximum.reduceat(local, offsets[:-1]),
                passage_mix=combined,
            )
            if facet == "method":
                arms["previous_full_goal"] = [
                    old_scores[q, pid] for pid, _, _, _ in pool
                ]
            grades = np.array([g for _, g, _, _ in pool])
            runs = {}
            for m, values in arms.items():
                best = np.argsort(-np.asarray(values), kind="stable")[:3]
                runs[m] = dict(
                    useful_in_three=int(sum(grades[best] >= 2)),
                    ndcg_at_10=float(ndcg_score([2**grades - 1], [values], k=10)),
                    top3=[pool[i][0] for i in best],
                    scores=[round(float(v), 7) for v in values],
                )
            grouped[facet].append(
                dict(
                    query=q,
                    selected_source_sentence=paper["abstract"].index(snippet),
                    pool=[[p, g] for p, g, _, _ in pool],
                    arms=runs,
                )
            )
    return dict(
        dataset_revision=DATA_REV,
        policy_sha256=policy_sha,
        encoder=policy["source"]["encoder"],
        corpus_sha256=policy["source"]["corpus_sha256"],
        script_sha256={
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in [
                ("evaluation", Path(__file__)),
                ("encoding", Path(__file__).with_name("build_passage_index.py")),
                ("scoring", DEST.parents[1] / "researchlens/passages.py"),
            ]
        },
        method_weight=0.5,
        positive_grade=2,
        split="fold1_test",
        summary={facet: summarize(rows) for facet, rows in grouped.items()},
        queries=grouped,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.cache)
    (DEST / "passage_comparison.json").write_text(
        json.dumps(result, allow_nan=False, indent=2)
    )
    print(json.dumps(result["summary"], indent=2))
