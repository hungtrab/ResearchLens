"""Check every selectable passage and its three suggestions against the snapshot."""

import hashlib
import json
from pathlib import Path

import numpy as np
from researchlens.passages import PassageIndex, split_sentences

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "data/papers.jsonl"
    path = ROOT / "data/passages.npz"
    policy = ROOT / "evaluation/decision-study/text_policy.json"
    papers = {p["id"]: p for p in map(json.loads, source.read_text().splitlines())}
    index = PassageIndex.load(path, source)
    with np.load(path, allow_pickle=False) as bundle:
        assert bundle["version"].item() == 2
        assert (
            bundle["policy_sha"].item()
            == hashlib.sha256(policy.read_bytes()).hexdigest()
        )
    assert set(index.ids) == {
        p for p, r in papers.items() if (r.get("abstract") or "").strip()
    }
    count = 0
    follow_count = 0
    for pid in index.ids:
        assert index.choices(pid) == split_sentences(papers[pid]["abstract"])
        for i in range(len(index.choices(pid))):
            matches = index.search(pid, i)
            assert len(matches) == 3
            assert list(sorted(matches, key=lambda m: -m["score"])) == matches
            for m in matches:
                assert m["quote"] and m["quote"] in papers[m["id"]]["abstract"]
            count += len(matches)
            seen = {pid, *[m["id"] for m in matches]}
            for after in ["next", *[m["id"] for m in matches]]:
                following = index.search(pid, i, after=after)
                assert len(following) == 3 and not seen.intersection(
                    m["id"] for m in following
                )
                for m in following:
                    assert m["quote"] and m["quote"] in papers[m["id"]]["abstract"]
                follow_count += len(following)
    print(
        f"{len(index.ids)} papers, {len(index.texts)} selectable passages, {count} first-page and {follow_count} follow-up source excerpts verified."
    )


if __name__ == "__main__":
    main()
