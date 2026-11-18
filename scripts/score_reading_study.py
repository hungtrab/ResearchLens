"""Score blinded, graded reading judgments. Incomplete topics are left pending."""
import argparse
import itertools
import json
import math
from pathlib import Path
from statistics import mean
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1] / "evaluation/reading-study"


def read(path):
    return json.loads(Path(path).read_text())


def load_sheet(path, packet):
    sheet = read(path)
    valid_codes = {card["code"] for card in packet["cards"]}
    if sheet["study"] != packet["study"] or not set(sheet["answers"]) <= valid_codes:
        raise ValueError("This sheet does not match the frozen study.")
    if sheet.get("status") == "calibration_only":
        raise ValueError("Calibration answers are excluded from scoring.")
    if sheet.get("kind") not in {"human", "assistant_pilot"} or not sheet.get("reviewer"):
        raise ValueError("Missing assessor provenance.")
    grades = {k: str(v["grade"]) for k, v in sheet["answers"].items() if str(v["grade"]) != ""}
    if not set(grades.values()) <= {"0", "1", "2", "3", "U"}:
        raise ValueError("Unrecognized grade.")
    return sheet, grades


def dcg(grades):
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def score(packet, runs, grades):
    results, pending = {}, {}
    for topic, rankings in runs["rankings"].items():
        codes = sorted({k for ranking in rankings.values() for k in ranking})
        missing = [k for k in codes if k not in grades]
        if missing:
            pending[topic] = {"unanswered": len(missing)}
            continue
        unknown = [k for k in codes if grades[k] == "U"]
        # A large uncertain topic is left pending instead of exploding the enumeration.
        if len(unknown) > 5:
            pending[topic] = {"uncertain": len(unknown)}
            continue
        bounds = {m: [1.0, 0.0] for m in rankings}
        for assignment in itertools.product(range(4), repeat=len(unknown)):
            values = {k: int(grades[k]) for k in codes if k not in unknown}
            values.update(zip(unknown, assignment))
            ideal = dcg(sorted(values.values(), reverse=True)[:runs["depth"]])
            for method, ranking in rankings.items():
                value = dcg([values[k] for k in ranking]) / ideal if ideal else 0.0
                bounds[method] = [min(bounds[method][0], value), max(bounds[method][1], value)]
        results[topic] = bounds
    overall = {method: [mean(row[method][i] for row in results.values()) for i in (0, 1)]
               for method in next(iter(results.values()), {})}
    return dict(fully_answered_topics=list(results), pending_topics=pending,
                pooled_ndcg_at_3_bounds_by_topic=results, macro_bounds_on_scored_topics=overall)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sheet", type=Path)
    parser.add_argument("--compare-human", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    packet, runs = read(ROOT/"blind_packet.json"), read(ROOT/"frozen_runs.json")
    if runs["study"] != packet["study"]:
        raise ValueError("Frozen runs and packet differ.")
    sheet, grades = load_sheet(args.sheet, packet)
    result = dict(study=sheet["study"], assessor_kind=sheet["kind"], reviewer=sheet["reviewer"],
                  n_answered=len(grades), n_uncertain=sum(v == "U" for v in grades.values()), **score(packet, runs, grades))
    if args.compare_human:
        other, labels = load_sheet(args.compare_human, packet)
        if sheet["kind"] != "human" or other["kind"] != "human" or sheet["reviewer"] == other["reviewer"]:
            raise ValueError("Agreement requires two different human assessors.")
        common = sorted(k for k in grades.keys() & labels.keys() if "U" not in (grades[k], labels[k]))
        first = [int(grades[k]) for k in common]
        second = [int(labels[k]) for k in common]
        kappa = None
        if len(set(first+second)) > 1:
            kappa = float(cohen_kappa_score(first,second,labels=[0,1,2,3],weights="linear"))
        result["agreement"] = dict(n_comparable=len(common), linear_weighted_kappa=kappa,
                                   exact=sum(a==b for a,b in zip(first,second)), disagreements=[k for k in common if grades[k]!=labels[k]])
    payload = json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    if args.output:
        args.output.write_text(payload)
    print(payload)

if __name__ == "__main__":
    main()
