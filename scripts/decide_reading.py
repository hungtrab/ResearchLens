"""Score a ResearchLens paper for method similarity using the frozen decision policy."""

import argparse
import json

from decision_features import DEST, DecisionFeatures, provenance
from fit_reading_decision import score


def decide(goal, seed, paper, policy, extractor_factory=DecisionFeatures):
    if not (paper.get("abstract") or "").strip():
        return {
            "action": "ask_reader",
            "message_vi": "Bài này chỉ có tiêu đề. Bạn hãy xem bản gốc trước khi quyết định.",
        }
    features = extractor_factory(include_laya="original" in policy["columns"]).compute(
        goal, seed, paper["title"], paper["abstract"]
    )
    probability = float(score([features], policy)[0])
    suggested = probability >= policy["threshold"]
    return {
        "action": "prioritize" if suggested else "defer",
        "message_vi": (
            "Nên xem phần phương pháp của bài này trước."
            if suggested
            else "Có thể để đọc sau khi tìm bài cùng phương pháp."
        ),
        "score": probability,
        "threshold": policy["threshold"],
        "target": policy["target"],
        "signals": features,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--goal", required=True, help="Research method or task to find, in English"
    )
    parser.add_argument(
        "--seed-title", required=True, help="English title of a reference paper"
    )
    parser.add_argument(
        "--paper-id", required=True, help="Candidate ID from data/papers.jsonl"
    )
    parser.add_argument(
        "--mode",
        choices=["text", "laya"],
        default="text",
        help="text uses TF-IDF and MiniLM; laya adds typed decision scores",
    )
    args = parser.parse_args()
    if not args.goal.strip() or not args.seed_title.strip():
        parser.error("Supply both a research method and a reference title.")
    corpus = DEST.parents[1] / "data/papers.jsonl"
    filename = "reading_policy.json" if args.mode == "laya" else "text_policy.json"
    policy = json.loads((DEST / filename).read_text())
    for key, value in provenance().items():
        if policy["source"][key] != value:
            parser.error(f"{key} differs from training. Rebuild features and refit.")
    with corpus.open() as stream:
        paper = next(
            (p for p in map(json.loads, stream) if p["id"] == args.paper_id), None
        )
    if paper is None:
        parser.error(f"Unknown paper ID: {args.paper_id}")
    answer = decide(args.goal.strip(), args.seed_title.strip(), paper, policy)
    print(
        json.dumps(
            {
                "paper_id": paper["id"],
                "title": paper["title"],
                "mode": args.mode,
                **answer,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
