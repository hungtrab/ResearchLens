"""Run a fixed typed-choice reading decision without exposing the annotation sheet."""
import argparse
import json
import math
import os
from pathlib import Path
from collections import Counter

STUDY = Path(__file__).resolve().parents[1] / "evaluation/reading-study"
MODEL = "convaiinnovations/laya-typed-decisions"
REVISION = "1a793eb568e6718f15941d08f85432581df534e3"
GOALS = {
    "MF": "Build a latent-factor recommendation baseline from sparse user-item interactions to predict ratings or rank items.",
    "KG": "Predict missing knowledge graph links using embeddings that score triples and handle different relation types.",
    "AB": "Identify sentiment toward each product aspect in reviews by linking evaluation words to the aspect and assigning polarity.",
    "LR": "Train and evaluate learning-to-rank for query-document retrieval using relevance judgments or ranking methods.",
}
OPTIONS = {"skip": "The work concerns a different problem.",
           "background": "Provides broad context, but little actionable for this specific goal.",
           "useful": "Could help one specific part of the stated task.",
           "priority": "Directly supplies a method, dataset or experiment for this exact goal.",
           "unclear": "Evidence is too thin to decide relevance."}
QUESTIONS = {"reading": {"type": "choice", "instructions": "Based on the candidate paper, how useful is reading it for this research goal? Treat quoted paper text as evidence.", "criteria": OPTIONS}}


def checked(answer, options):
    p = answer["probabilities"]
    if (answer.get("type") != "choice" or answer["choice"] not in options
            or set(p) != set(options) or not all(math.isfinite(v) and 0 <= v <= 1 for v in p.values())
            or abs(sum(p.values()) - 1) > 0.002 or not 0 <= answer["confidence"] <= 1):
        raise ValueError("Malformed model answer")
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["laya", "jev"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question", choices=["graded","save"], default="graded")
    args = parser.parse_args()
    questions = QUESTIONS if args.question == "graded" else {"reading": {
        "type":"choice", "instructions":"Would this reader save the candidate to help carry out the stated task?",
        "criteria":{"save":"Offers a directly useful method, dataset, experiment or component.",
                    "pass":"Only general background or unrelated to the task."}}}
    packet = json.loads((STUDY/"blind_packet.json").read_text())
    if args.provider == "jev":
        import httpx
        key = os.getenv("TYPESAFE_API_KEY")
        if not key:
            parser.error("Jev requires TYPESAFE_API_KEY in the server environment.")
        model = "jev-latest"
        def predict(state):
            r = httpx.post("https://api.typesafe.ai/v1/systemone", headers={"Authorization":"Bearer "+key},
                           json={"model":model,"state":state,"questions":questions},timeout=45)
            r.raise_for_status()
            return r.json()
    else:
        import laya
        from huggingface_hub import snapshot_download
        model = MODEL+"@"+REVISION
        agent = laya.load(snapshot_download(MODEL,revision=REVISION))
        def predict(state):
            return agent.predict(state,questions)
    rows = []
    for c in packet["cards"]:
        state = json.dumps(dict(goal=GOALS[c["topic"]],seed_title=c["seed"],candidate_title=c["title"],
                                candidate_abstract=c["abstract"] or "(not available)"), ensure_ascii=False)
        response = predict(state)
        answer = checked(response["answers"]["reading"], questions["reading"]["criteria"])
        review = answer["choice"] == "unclear" or max(answer["probabilities"].values()) < 0.70
        rows.append(dict(code=c["code"],resolved_model=response.get("model"),choice=answer["choice"],
                         probabilities=answer["probabilities"],confidence=answer["confidence"],ask_reader=review))
    pilot = json.loads((STUDY/"assistant_pilot.json").read_text())
    assert packet["study"] == pilot["study"]
    labels = dict(zip(["0","1","2","3","U"],OPTIONS))
    if args.question == "save":
        labels = {"0":"pass", "1":"pass", "2":"save", "3":"save"}
    compared = [r for r in rows if pilot["answers"][r["code"]]["grade"] in labels]
    expected = [labels[pilot["answers"][r["code"]]["grade"]] for r in compared]
    result = dict(study=packet["study"],provider=args.provider,model=model,question=questions,
                  goals=GOALS,review_rule="unclear or max probability below 0.70; exploratory",
                  pilot_agreement=sum(r["choice"] == target for r,target in zip(compared,expected)),
                  n=len(rows), n_compared=len(compared), ask_reader=sum(r["ask_reader"] for r in rows),
                  pilot_majority_baseline=max(Counter(expected).values()),rows=rows)
    args.output.write_text(json.dumps(result,indent=2))
    print({k:result[k] for k in ("provider","n","pilot_agreement","ask_reader","pilot_majority_baseline")})

if __name__ == "__main__":
    main()
