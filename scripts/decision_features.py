"""Compute evidence for a typed reading decision. Heavy models load only on demand."""

import argparse
import hashlib
import json
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from reading_decisions import GOALS, MODEL, REVISION, STUDY

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DATA_REV = "7ffe012c1d92c470f185fc086086caf5b8e05ef3"
DEST = STUDY.parent / "decision-study"
SAVE = {
    "type": "choice",
    "instructions": "Would this reader save the candidate to help carry out the stated task?",
    "criteria": {
        "save": "Offers a directly useful method, dataset, experiment or component.",
        "pass": "Only general background or unrelated to the task.",
    },
}
CSFCUBE_FILES = {
    "abstracts-csfcube-preds.jsonl": "9d2d9414574b6e91d2487d3396d3256577149dad859e9f6eb231b95c2ba5577c",
    "test-pid2anns-csfcube-method.json": "4d9b48f4b6a7a4fc24a30e0c0e74d1cc53eb4214ec4b855e3cd98e86829323a2",
    "evaluation_splits.json": "1d23a1b2110e874809c9516f65c70e3c6e4b8fe3bd748ff36ff05c4875226c88",
}


def provenance():
    return dict(
        laya=[MODEL, REVISION],
        encoder=[ENCODER, ENCODER_REV],
        data_revision=DATA_REV,
        feature_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        corpus_sha256=hashlib.sha256(
            (STUDY.parents[1] / "data/papers.jsonl").read_bytes()
        ).hexdigest(),
    )


def records(cache):
    files = (
        "abstracts-csfcube-preds.jsonl",
        "test-pid2anns-csfcube-method.json",
        "evaluation_splits.json",
    )
    cache.mkdir(parents=True, exist_ok=True)
    for name in files:
        path = cache / name
        if (
            not path.exists()
            or hashlib.sha256(path.read_bytes()).hexdigest() != CSFCUBE_FILES[name]
        ):
            url = f"https://raw.githubusercontent.com/iesl/CSFCube/{DATA_REV}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
                if hashlib.sha256(content).hexdigest() != CSFCUBE_FILES[name]:
                    raise ValueError(f"Unexpected content for CSFCube file {name}")
                path.write_bytes(content)
    papers = {
        p["paper_id"]: p
        for p in map(json.loads, (cache / files[0]).read_text().splitlines())
    }
    judged = json.loads((cache / files[1]).read_text())
    splits = json.loads((cache / files[2]).read_text())["method"]
    for split in ("dev", "test"):
        for key in sorted(splits["fold1_" + split]):
            group = key.split("_")[0]
            query = papers[group]
            method = " ".join(
                s
                for s, tag in zip(query["abstract"], query["pred_labels"])
                if tag == "method_label"
            )
            goal = method or " ".join(query["abstract"])
            for pid, grade in zip(
                judged[group]["cands"], judged[group]["relevance_adju"], strict=True
            ):
                paper = papers[pid]
                yield dict(
                    split=split,
                    group=group,
                    pid=pid,
                    grade=grade,
                    goal=goal,
                    seed=query["title"],
                    title=paper["title"],
                    abstract=" ".join(paper["abstract"]),
                )
    packet = json.loads((STUDY / "blind_packet.json").read_text())
    pilot = json.loads((STUDY / "assistant_pilot.json").read_text())
    assert packet["study"] == pilot["study"]
    for card in packet["cards"]:
        grade = pilot["answers"][card["code"]]["grade"]
        yield dict(
            split="pilot",
            group=card["topic"],
            pid=card["id"],
            code=card["code"],
            grade=None if grade == "U" else int(grade),
            goal=GOALS[card["topic"]],
            seed=card["seed"],
            title=card["title"],
            abstract=card["abstract"],
        )


class DecisionFeatures:
    def __init__(self, include_laya=True):
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.laya = None
        if include_laya:
            import laya

            self.laya = laya.load(
                snapshot_download(MODEL, revision=REVISION), device=self.device
            )
        location = snapshot_download(
            ENCODER,
            revision=ENCODER_REV,
            allow_patterns=["*.json", "model.safetensors", "vocab.txt"],
        )
        self.encoder = AutoModel.from_pretrained(location).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(location)
        # IDF comes from the separate workspace corpus, without benchmark relevance labels.
        with (STUDY.parents[1] / "data/papers.jsonl").open() as f:
            self.lex = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(
                p["title"] + " " + (p.get("abstract") or "") for p in map(json.loads, f)
            )

    @lru_cache(maxsize=8192)
    def embed(self, text):
        import torch

        inputs = self.tokenizer(
            text, truncation=True, max_length=256, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            hidden = self.encoder(**inputs).last_hidden_state
            vector = hidden.mean(dim=1)[0]  # one unpadded example
            return torch.nn.functional.normalize(vector, dim=0).cpu().numpy()

    def compute(self, goal, seed, title, abstract):
        query = goal + " " + seed
        document = title + ". " + abstract
        vectors = self.lex.transform([query, document])
        lexical = float(vectors[0].multiply(vectors[1]).sum())
        semantic = float(self.embed(query) @ self.embed(document))
        if self.laya is None:
            return dict(lexical=lexical, semantic=semantic)
        state = json.dumps(
            dict(
                goal=goal,
                seed_title=seed,
                candidate_title=title,
                candidate_abstract=abstract or "(not available)",
            ),
            ensure_ascii=False,
        )
        # Ask for observable technical overlap, decoupled from the final action.
        specific = {
            "type": "noul",
            "instructions": "Does the candidate describe a concrete technique, dataset, or experiment that matches the specific research method or task sought in the goal? Shared field or vocabulary alone is insufficient.",
        }
        answers = self.laya.predict(state, {"reading": SAVE, "match": specific})[
            "answers"
        ]
        return dict(
            lexical=lexical,
            semantic=semantic,
            original=answers["reading"]["probabilities"]["save"],
            specific=answers["match"]["noul"],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    extractor = DecisionFeatures()
    # Resume only identical feature code, model pins, input text and IDF corpus.
    sources = provenance()
    signature = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    cached = args.cache / ("features-" + signature + ".jsonl")
    rows = {}
    if cached.exists():
        for line in cached.read_text().splitlines():
            key, features = json.loads(line)
            rows[key] = features
    output = []
    for idx, r in enumerate(records(args.cache)):
        text = {k: r[k] for k in ("goal", "seed", "title", "abstract")}
        key = hashlib.sha256(json.dumps(text, sort_keys=True).encode()).hexdigest()
        if key not in rows:
            rows[key] = extractor.compute(**text)
            with cached.open("a") as f:
                f.write(json.dumps([key, rows[key]]) + "\n")
        output.append({**{k: v for k, v in r.items() if k not in text}, **rows[key]})
        if idx % 200 == 0:
            print(idx, flush=True)
    DEST.mkdir(parents=True, exist_ok=True)
    feature_file = DEST / "features.jsonl"
    feature_file.write_text("".join(json.dumps(r) + "\n" for r in output))
    sources["features_sha256"] = hashlib.sha256(feature_file.read_bytes()).hexdigest()
    (DEST / "feature_provenance.json").write_text(json.dumps(sources, indent=2))


if __name__ == "__main__":
    main()
