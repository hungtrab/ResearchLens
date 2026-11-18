"""Rank indexed papers from one selected abstract sentence, with a source excerpt."""

import hashlib
import json
import re

import numpy as np

from researchlens.catalog import title_key


def split_sentences(text):
    # Slice the original string so an excerpt can be located verbatim in the abstract.
    text = (text or "").strip()
    if not text:
        return []
    start = 0
    parts = []
    for boundary in re.finditer(r'(?<=[.!?])\s+(?=[A-Z0-9“"(])|\n\s*\n', text):
        if re.search(
            r"(?:\b(?:al|Fig|Eq|Dr|Prof|vs)|\b[A-Z]|e\.g|i\.e)\.$",
            text[max(start, boundary.start() - 12) : boundary.start()],
        ):
            continue
        sentence = text[start : boundary.start()].strip()
        if sentence:
            parts.append(sentence)
        start = boundary.end()
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def passage_scores(query, documents, sentences, offsets):
    # Balance global topic and the best supporting passage. Both weights are fixed.
    local = np.einsum("ij,j->i", sentences, query)
    global_scores = np.einsum("ij,j->i", documents, query)
    return (global_scores + np.maximum.reduceat(local, offsets[:-1])) / 2, local


def validate_shortlist(titles, offsets, picks, scores):
    if (
        picks.shape != (offsets[-1], 3)
        or scores.shape != picks.shape
        or not np.issubdtype(picks.dtype, np.integer)
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
        or np.any((picks < -1) | (picks >= len(titles)))
    ):
        raise ValueError("Invalid policy shortlist.")
    keys = [title_key(t) for t in titles]
    for i in range(len(titles)):
        for row in picks[offsets[i] : offsets[i + 1]]:
            seen = {keys[i]}
            ended = False
            for j in row:
                if j == -1:
                    ended = True
                elif ended or keys[j] in seen:
                    raise ValueError("Invalid or repeated paper in policy shortlist.")
                else:
                    seen.add(keys[j])


def validate_followups(titles, offsets, first, picks, scores):
    if (
        picks.shape != (len(first), 4, 3)
        or scores.shape != picks.shape
        or not np.issubdtype(picks.dtype, np.integer)
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
        or np.any((picks < -1) | (picks >= len(titles)))
    ):
        raise ValueError("Invalid reading feedback index.")
    keys = [title_key(t) for t in titles]
    for owner in range(len(keys)):
        for row in range(offsets[owner], offsets[owner + 1]):
            excluded = {keys[owner], *[keys[j] for j in first[row] if j >= 0]}
            for slot, indices in enumerate(picks[row]):
                valid = indices[indices >= 0]
                if np.any(indices[: len(valid)] < 0) or (
                    slot and first[row, slot - 1] < 0 and len(valid)
                ):
                    raise ValueError("Invalid feedback page or padding.")
                names = {keys[j] for j in valid}
                if names & excluded or len(names) != len(valid):
                    raise ValueError("Feedback results repeat a paper title.")


class PassageIndex:
    def __init__(
        self,
        ids,
        titles,
        texts,
        offsets,
        documents,
        sentences,
        shortlist=None,
        relevance=None,
        followups=None,
        followup_scores=None,
    ):
        self.ids = list(ids)
        self.positions = {pid: i for i, pid in enumerate(self.ids)}
        self.title_keys = [title_key(t) for t in titles]
        self.texts = list(texts)
        self.offsets = offsets
        self.documents = documents
        self.sentences = sentences
        self.shortlist = shortlist
        self.relevance = relevance
        self.followups = followups
        self.followup_scores = followup_scores

    @classmethod
    def load(cls, path, corpus_path):
        with np.load(path, allow_pickle=False) as bundle:
            if (
                bundle["version"].item() not in (1, 2)
                or bundle["corpus_sha"].item()
                != hashlib.sha256(corpus_path.read_bytes()).hexdigest()
            ):
                raise ValueError("Passage index does not match the corpus.")
            ids, titles = (bundle[k].tolist() for k in ("ids", "titles"))
            texts = json.loads(bundle["texts_utf8"].tobytes())
            offsets = bundle["offsets"]
            documents, sentences = (
                bundle[k].astype(np.float32) for k in ("documents", "sentences")
            )
            if (
                len(ids) != len(set(ids))
                or len(titles) != len(ids)
                or offsets.shape != (len(ids) + 1,)
                or not np.issubdtype(offsets.dtype, np.integer)
                or offsets[0] != 0
                or offsets[-1] != len(texts)
                or np.any(np.diff(offsets) <= 0)
                or documents.ndim != 2
                or not np.isfinite(documents).all()
                or not np.isfinite(sentences).all()
                or documents.shape[0] != len(ids)
                or sentences.shape != (len(texts), documents.shape[1])
            ):
                raise ValueError("Invalid passage index dimensions.")
            shortlist = relevance = followups = followup_scores = None
            if bundle["version"].item() == 2:
                if not {"shortlist", "relevance"}.issubset(bundle.files):
                    raise ValueError("Missing policy shortlist.")
                shortlist, relevance = bundle["shortlist"], bundle["relevance"]
                validate_shortlist(titles, offsets, shortlist, relevance)
                if {"followups", "followup_scores"}.intersection(bundle.files):
                    if not {"followups", "followup_scores"}.issubset(bundle.files):
                        raise ValueError("Incomplete reading feedback index.")
                    followups, followup_scores = (
                        bundle["followups"],
                        bundle["followup_scores"],
                    )
                    validate_followups(
                        titles, offsets, shortlist, followups, followup_scores
                    )
        return cls(
            ids,
            titles,
            texts,
            offsets,
            documents,
            sentences,
            shortlist,
            relevance,
            followups,
            followup_scores,
        )

    def choices(self, pid):
        i = self.positions[pid]
        return self.texts[self.offsets[i] : self.offsets[i + 1]]

    def search(self, pid, sentence_index, limit=3, after=None):
        i = self.positions[pid]
        start, end = self.offsets[i : i + 2]
        if not 0 <= sentence_index < end - start:
            raise ValueError("Select a sentence from this abstract.")
        if limit < 1:
            raise ValueError("Limit must be positive.")
        if after is not None and self.followups is None:
            raise ValueError("Feedback pages are unavailable in this index.")
        if self.shortlist is not None:
            if limit > 3:
                raise ValueError("At most three policy suggestions are indexed.")
            row = start + sentence_index
            query = self.sentences[row]
            picks, scores = self.shortlist[row], self.relevance[row]
            if after is not None:
                slot = 0
                if after != "next":
                    chosen = self.positions.get(after, -2)
                    hits = np.flatnonzero(picks == chosen)
                    if not len(hits):
                        raise ValueError(
                            "Choose a paper from the first three suggestions."
                        )
                    slot = int(hits[0]) + 1
                    query = (query + self.documents[chosen]) / 2
                picks, scores = (
                    self.followups[row, slot],
                    self.followup_scores[row, slot],
                )
            items = []
            for j, score in zip(picks[:limit], scores[:limit]):
                if j < 0:
                    break
                a, b = self.offsets[j : j + 2]
                best = a + np.argmax(self.sentences[a:b] @ query)
                items.append(
                    dict(id=self.ids[j], quote=self.texts[best], score=float(score))
                )
            return items
        scores, local = passage_scores(
            self.sentences[start + sentence_index],
            self.documents,
            self.sentences,
            self.offsets,
        )
        seen = {self.title_keys[i]}
        items = []
        for j in np.argsort(-scores, kind="stable"):
            if j == i or self.title_keys[j] in seen:
                continue
            seen.add(self.title_keys[j])
            a, b = self.offsets[j : j + 2]
            evidence = a + np.argmax(local[a:b])
            items.append(
                dict(id=self.ids[j], quote=self.texts[evidence], score=float(scores[j]))
            )
            if len(items) == limit:
                break
        return items
