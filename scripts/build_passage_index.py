"""Encode the snapshot offline. Serving only needs NumPy and the bundled index."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import expit
from researchlens.catalog import title_key
from decision_features import DecisionFeatures, ENCODER, ENCODER_REV, DEST, provenance
from researchlens.passages import split_sentences

ROOT = Path(__file__).resolve().parents[1]


def encode(encoder, texts):
    import torch

    # Sort by length to avoid padding short sentences to the longest document.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    output = np.empty(
        (len(order), encoder.encoder.config.hidden_size), dtype=np.float16
    )
    with torch.inference_mode():
        for pos in range(0, len(order), 32):
            batch = order[pos : pos + 32]
            tokens = encoder.tokenizer(
                [texts[i] for i in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(encoder.device)
            hidden = encoder.encoder(**tokens).last_hidden_state
            mask = tokens.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            output[batch] = torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()
    return output


def build_arrays(encoder, records):
    docs, texts, offsets = [], [], [0]
    for title, parts in records:
        if not parts:
            raise ValueError("Each indexed paper must have an excerpt.")
        docs.append(title + ". " + " ".join(parts))
        texts.extend(parts)
        offsets.append(len(texts))
    return encode(encoder, docs), encode(encoder, texts), np.asarray(offsets), texts


def distinct_three(scores, titles, excluded):
    seen = set(excluded)
    picks = []
    for j in np.argsort(-scores, kind="stable"):
        if titles[j] in seen:
            continue
        seen.add(titles[j])
        picks.append(j)
        if len(picks) == 3:
            break
    return picks


def policy_shortlist(encoder, rows, document_vectors, policy):
    """Compute three distinct titles per selectable passage using the frozen reading policy."""
    titles = [title_key(t) for _, t, _ in rows]
    documents = [t + ". " + " ".join(parts) for _, t, parts in rows]
    owners = [i for i, (_, _, parts) in enumerate(rows) for _ in parts]
    queries = [sentence + " " + t for _, t, parts in rows for sentence in parts]
    contexts = encode(encoder, queries).astype(np.float32)
    lexical_documents = encoder.lex.transform(documents).T
    semantic_documents = document_vectors.astype(np.float32).T
    if policy["columns"] != ["lexical", "semantic"]:
        raise ValueError("Expected the two-signal text policy.")
    weights = np.array(policy["weights"]) / policy["scale"]
    bias = policy["bias"] - np.dot(weights, policy["center"])
    picks = np.full((len(queries), 3), -1, dtype=np.int32)
    relevance = np.zeros(picks.shape, dtype=np.float32)
    # Every possible first-page click can be evaluated offline on this snapshot.
    followups = np.full((len(queries), 4, 3), -1, dtype=np.int32)
    followup_scores = np.zeros(followups.shape, dtype=np.float32)
    doc_logits = weights[0] * (lexical_documents.T @ lexical_documents).toarray()
    doc_logits += weights[1] * (semantic_documents.T @ semantic_documents) + bias
    for a in range(0, len(queries), 128):
        b = min(a + 128, len(queries))
        lexical = (encoder.lex.transform(queries[a:b]) @ lexical_documents).toarray()
        logits = (
            bias
            + weights[0] * lexical
            + weights[1] * (contexts[a:b] @ semantic_documents)
        )
        for i, original in enumerate(logits, a):
            seed_title = titles[owners[i]]
            first = distinct_three(original, titles, {seed_title})
            picks[i, : len(first)] = first
            relevance[i, : len(first)] = expit(original[first])
            seen = {seed_title, *[titles[j] for j in first]}
            for slot, values in enumerate(
                [original, *[(original + doc_logits[j]) / 2 for j in first]]
            ):
                new = distinct_three(values, titles, seen)
                followups[i, slot, : len(new)] = new
                followup_scores[i, slot, : len(new)] = expit(values[new])
    return picks, relevance, followups, followup_scores


def load_policy():
    policy_bytes = (DEST / "text_policy.json").read_bytes()
    policy = json.loads(policy_bytes)
    if any(policy["source"][k] != v for k, v in provenance().items()):
        raise ValueError(
            "Build the index using the corpus and features from the frozen policy."
        )
    return policy, hashlib.sha256(policy_bytes).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data")
    data = p.parse_args().data
    corpus = data / "papers.jsonl"
    raw = corpus.read_bytes()
    rows = [
        (r["id"], r["title"], split_sentences(r.get("abstract")))
        for r in map(json.loads, raw.splitlines())
    ]
    rows = [r for r in rows if r[2]]
    enc = DecisionFeatures(include_laya=False)
    docs, sentences, offsets, texts = build_arrays(enc, [(t, s) for _, t, s in rows])
    policy, policy_sha = load_policy()
    picks, relevance, followups, followup_scores = policy_shortlist(
        enc, rows, docs, policy
    )
    destination = data / "passages.npz"
    with tempfile.NamedTemporaryFile(dir=data, suffix=".npz", delete=False) as temp:
        tmp = Path(temp.name)
    try:
        np.savez_compressed(
            tmp,
            version=2,
            shortlist=picks,
            followups=followups,
            followup_scores=followup_scores,
            relevance=relevance,
            policy_sha=policy_sha,
            corpus_sha=hashlib.sha256(raw).hexdigest(),
            encoder=ENCODER,
            revision=ENCODER_REV,
            ids=[p for p, _, _ in rows],
            titles=[t for _, t, _ in rows],
            texts_utf8=np.frombuffer(
                json.dumps(texts, ensure_ascii=False).encode(), dtype=np.uint8
            ),
            offsets=offsets,
            documents=docs,
            sentences=sentences,
        )
        os.chmod(tmp, 0o644)
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)
    print(
        len(rows),
        "papers,",
        len(texts),
        "sentences,",
        destination.stat().st_size,
        "bytes",
    )


if __name__ == "__main__":
    main()
