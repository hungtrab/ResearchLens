"""Scientific sentence embeddings and reciprocal-rank fusion for passage retrieval."""

import hashlib
from pathlib import Path

import numpy as np

MODEL = "allenai/aspire-contextualsentence-singlem-compsci"
REVISION = "8cba94d6199519804674b7695fbb79ea0ae7bfe4"


WEIGHTS_SHA256 = "1b1698557ff51f257412cc664b90dd43d8c32595653cb5b0ec8ff98815575be0"


def verify_checkpoint(path):
    with (Path(path) / "pytorch_model.bin").open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != WEIGHTS_SHA256:
            raise ValueError(
                "The local checkpoint differs from the pinned ASPIRE release."
            )


def windows(title_ids, passages, sep, cls):
    # Repeat the title in long abstracts so every selectable sentence gets a vector.
    prefix = [cls, *title_ids[:96], sep]
    ids = prefix.copy()
    spans = []
    for i, tokens in enumerate(passages):
        if not tokens:
            raise ValueError("Cannot encode an empty passage.")
        tokens = tokens[: 511 - len(prefix)]
        if len(ids) + len(tokens) > 511:
            yield ids + [sep], spans
            ids, spans = prefix.copy(), []
        spans.append((i, len(ids), len(ids) + len(tokens)))
        ids.extend(tokens)
    if spans:
        yield ids + [sep], spans


def contextual_vectors(records, model_path=None):
    """Return one unnormalized mean of contextual token states per sentence."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    location = str(model_path) if model_path else MODEL
    tokenizer = AutoTokenizer.from_pretrained(location, revision=REVISION)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(location, revision=REVISION).eval().to(device)
    jobs = []
    result = {}
    for pid, (title, parts) in records.items():
        token_rows = tokenizer(
            [title, *parts], add_special_tokens=False, verbose=False
        )["input_ids"]
        result[pid] = np.empty((len(parts), model.config.hidden_size), np.float32)
        for ids, spans in windows(
            token_rows[0],
            token_rows[1:],
            tokenizer.sep_token_id,
            tokenizer.cls_token_id,
        ):
            jobs.append((pid, ids, spans))
    jobs.sort(key=lambda x: len(x[1]))
    with torch.inference_mode():
        for i in range(0, len(jobs), 8):
            batch = jobs[i : i + 8]
            inputs = tokenizer.pad(
                [{"input_ids": j[1]} for j in batch], padding=True, return_tensors="pt"
            ).to(device)
            hidden = model(**inputs).last_hidden_state.cpu().numpy()
            for mat, (pid, _, spans) in zip(hidden, batch):
                for j, a, b in spans:
                    result[pid][j] = mat[a:b].mean(axis=0)
    return result


def reciprocal(scores):
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return 1 / (60 + ranks)


def fused_scores(baseline, contextual, weight):
    return (1 - weight) * reciprocal(baseline) + weight * reciprocal(contextual)


def nearest(query, candidate_vectors):
    return np.array(
        [-np.min(np.sum((m - query) ** 2, axis=1)) for m in candidate_vectors]
    )
