"""Compare a frozen Reader, BM25, fusion and a top-20 cross-encoder on three tasks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.feature_extraction.text import CountVectorizer

from method_datasets import ARCHIVES, Query, acl_arc, scierc

STUDY = Path(__file__).resolve().parents[1] / "evaluation/decision-study"
MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
FILES = {
    "config.json": "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc",
    "model.safetensors": "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae",
    "special_tokens_map.json": "3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "tokenizer_config.json": "a5c2e5a7b1a29a0702cd28c08a399b5ecc110c263009d17f7e3b415f25905fd8",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


def bm25_scores(documents, queries):
    shape = (len(queries), len(documents))
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2))
    if not all(shape) or not any(vectorizer.build_tokenizer()(d) for d in documents):
        return np.zeros(shape), np.zeros(shape)
    matrix = vectorizer.fit_transform(documents).tocoo().astype(float)
    unigram = np.array([" " not in t for t in vectorizer.get_feature_names_out()])
    length = np.bincount(
        matrix.row, weights=matrix.data * unigram[matrix.col], minlength=len(documents)
    )
    df = np.bincount(matrix.col, minlength=matrix.shape[1])
    idf = np.log1p((len(documents) - df + 0.5) / (df + 0.5))
    norm = 1.2 * (0.25 + 0.75 * length / length.mean())
    matrix.data = idf[matrix.col] * matrix.data * 2.2 / (matrix.data + norm[matrix.row])
    q = vectorizer.transform(queries)
    q.data.fill(1)
    weighted = matrix.tocsr()
    return (q[:, unigram] @ weighted[:, unigram].T).toarray(), (
        q @ weighted.T
    ).toarray()


def fusion_scores(reader, phrase):
    reader, phrase = np.asarray(reader), np.asarray(phrase)
    if reader.ndim != 1 or reader.shape != phrase.shape:
        raise ValueError("Both score vectors must cover the same candidates")
    return 1 / (60 + rankdata(-reader)) + 1 / (60 + rankdata(-phrase))


def rerank_order(order, logits):
    order = np.asarray(order)
    n = len(logits)
    return np.concatenate(
        [order[:n][np.argsort(-np.asarray(logits), kind="stable")], order[n:]]
    )


def metrics(ordered_grades, positive):
    grades = np.asarray(ordered_grades)
    found = np.flatnonzero(grades >= positive)
    discounts = 1 / np.log2(np.arange(min(10, len(grades))) + 2)
    gains = np.exp2(grades.astype(float)) - 1
    ideal = np.sort(gains)[::-1][:10] @ discounts
    return dict(
        first=int(found[0] + 1) if len(found) else None,
        slots3=int(sum(grades[:3] >= positive)),
        hit3=int(any(found < 3)),
        hit20=int(any(found < 20)),
        ndcg10=float(gains[:10] @ discounts / ideal) if ideal else 0.0,
    )


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Reranker:
    def __init__(self, location, device):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from huggingface_hub import snapshot_download

        if not (location / "model.safetensors").is_file():
            location = Path(
                snapshot_download(MODEL, revision=REVISION, allow_patterns=list(FILES))
            )
        if any(checksum(location / name) != sha for name, sha in FILES.items()):
            raise ValueError("Cross-encoder artifact differs from its pinned revision")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(location)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(location)
            .to(device)
            .eval()
        )
        self.torch = torch

    def __call__(self, query, docs):
        inputs = self.tokenizer(
            [query] * len(docs),
            docs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        with self.torch.inference_mode():
            return self.model(**inputs).logits.flatten().cpu().numpy()


def ranked_rows(docs, queries, reader, reranker, threshold=1, saved_reader=None):
    from build_passage_index import encode

    ids = sorted(docs)
    texts = [docs[k] for k in ids]
    qtexts = [q.text for q in queries]
    uni, phrase = bm25_scores(texts, qtexts)
    if saved_reader is None:
        encoder, w = reader
        sparse_scores = (
            encoder.lex.transform(qtexts) @ encoder.lex.transform(texts).T
        ).toarray()
        vectors = encode(encoder, texts).astype(np.float32)
        dense_scores = encode(encoder, qtexts).astype(np.float32) @ vectors.T
        saved_reader = w[0] * sparse_scores + w[1] * dense_scores
    result = []
    for i, q in enumerate(queries):
        eligible = np.array([j for j, key in enumerate(ids) if key not in q.excluded])
        keys = [ids[j] for j in eligible]
        grades = np.array([q.grades.get(k, 0) for k in keys])
        if not (grades >= threshold).any() and threshold == 1:
            raise ValueError(f"Missing known target {q.key}")
        scores = dict(
            reader=saved_reader[i][eligible],
            unigram=uni[i, eligible],
            phrase=phrase[i, eligible],
        )
        scores["fusion"] = fusion_scores(scores["reader"], scores["phrase"])
        orders = {k: np.argsort(-v, kind="stable") for k, v in scores.items()}
        top = orders["fusion"][:20]
        logits = reranker(q.text, [docs[keys[j]] for j in top])
        orders["cross"] = rerank_order(orders["fusion"], logits)
        arms = {}
        for name, order in orders.items():
            m = metrics(grades[order], threshold)
            m["top3"] = [keys[j] for j in order[:3]]
            m["first_target"] = keys[order[m["first"] - 1]] if m["first"] else None
            arms[name] = m
        result.append(
            dict(
                query=q.key,
                cluster=q.cluster,
                pool=len(keys),
                fusion_top20=[keys[j] for j in top],
                positives=int(sum(grades >= threshold)),
                arms=arms,
            )
        )
    return result


def bootstrap(delta, cluster, seed=20261118):
    _, inverse = np.unique(cluster, return_inverse=True)
    total = np.bincount(inverse, weights=delta)
    size = np.bincount(inverse)
    indices = np.random.default_rng(seed).integers(len(size), size=(10000, len(size)))
    values = total[indices].sum(axis=1) / size[indices].sum(axis=1)
    return dict(
        delta=float(np.mean(delta)), ci95=np.quantile(values, [0.025, 0.975]).tolist()
    )


def summary(rows):
    cluster = [r["cluster"] for r in rows]
    report = {}
    for arm in rows[0]["arms"]:
        scores = [r["arms"][arm] for r in rows]
        changes = [
            s["hit3"] - r["arms"]["reader"]["hit3"]
            for s, r in zip(scores, rows, strict=True)
        ]
        report[arm] = {
            k: sum(s[k] for s in scores) for k in ("hit3", "hit20", "slots3")
        }
        report[arm].update(
            ndcg10=float(np.mean([s["ndcg10"] for s in scores])),
            hit3_vs_reader=bootstrap(changes, cluster),
            wins=sum(v > 0 for v in changes),
            losses=sum(v < 0 for v in changes),
        )
    return dict(queries=len(rows), clusters=len(set(cluster)), arms=report)


def csfcube(cache, reranker):
    from evaluate_passages import source
    from researchlens.passages import split_sentences

    cache.mkdir(parents=True, exist_ok=True)
    papers = {
        r["paper_id"]: r
        for r in map(
            json.loads, source(cache, "abstracts-csfcube-preds.jsonl").splitlines()
        )
    }
    archived = json.loads((STUDY / "passage_comparison.json").read_bytes())
    rows = []
    for r in archived["queries"]["method"]:
        key = r["query"]
        seed = papers[key]
        sent = split_sentences(seed["abstract"][r["selected_source_sentence"]])[0]
        query = Query(key, key, sent + " " + seed["title"], set(), dict(r["pool"]))
        baseline = dict(
            zip(query.grades, r["arms"]["text_policy"]["scores"], strict=True)
        )
        docs = {
            k: papers[k]["title"]
            + ". "
            + " ".join(split_sentences(" ".join(papers[k]["abstract"])))
            for k in query.grades
        }
        scores = np.array([[baseline[k] for k in sorted(docs)]])
        rows.extend(ranked_rows(docs, [query], None, reranker, 2, scores))
    return rows


def main():
    from build_passage_index import load_policy
    from decision_features import DecisionFeatures

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--csfcube-cache", type=Path)
    parser.add_argument("--output", type=Path, default=STUDY / "method_retrieval.json")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    policy, policy_sha = load_policy()
    if policy["columns"] != ["lexical", "semantic"]:
        raise ValueError("Expected the frozen lexical–semantic policy")
    encoder = DecisionFeatures(include_laya=False)
    reader = (encoder, np.array(policy["weights"]) / policy["scale"])
    cross = Reranker(args.cache / "reranker", encoder.device)
    docs, visible, masked, sci_audit = scierc(args.cache)
    runs = {}
    for name, queries in [("scierc_visible", visible), ("scierc_masked", masked)]:
        print(name, len(queries), flush=True)
        runs[name] = ranked_rows(docs, queries, reader, cross)
    docs, queries, arc_audit = acl_arc(args.cache)
    print("acl_arc", len(queries), flush=True)
    runs["acl_arc"] = ranked_rows(docs, queries, reader, cross)
    runs["csfcube"] = csfcube(args.csfcube_cache or args.cache / "csfcube", cross)
    paired = list(zip(runs["scierc_visible"], runs["scierc_masked"], strict=True))
    if any(a["query"] != b["query"] for a, b in paired):
        raise ValueError("Mask comparison requires paired source queries")
    mask_effect = {
        arm: bootstrap(
            [b["arms"][arm]["hit3"] - a["arms"][arm]["hit3"] for a, b in paired],
            [a["cluster"] for a, b in paired],
        )
        for arm in paired[0][0]["arms"]
    }
    report = dict(
        masked_minus_visible_hit3=mask_effect,
        protocol_sha256=checksum(STUDY / "method_retrieval_protocol.md"),
        code_sha256={
            p: checksum(Path(__file__).with_name(p))
            for p in ("evaluate_method_retrieval.py", "method_datasets.py")
        },
        archive_sha256={k: v[1] for k, v in ARCHIVES.items()},
        policy_sha256=policy_sha,
        reused_pools_sha256=checksum(STUDY / "passage_comparison.json"),
        reranker=[MODEL, REVISION],
        encoder=policy["source"]["encoder"],
        audit=dict(scierc=sci_audit, acl_arc=arc_audit),
        summary={name: summary(rows) for name, rows in runs.items()},
        queries=runs,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
