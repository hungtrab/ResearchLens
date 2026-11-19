"""Every recommendation number quoted in the report and slides must come from an artifact."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = (ROOT / "report/main.tex").read_text()
SLIDES = (ROOT / "slides/main.tex").read_text()


def load(name):
    return json.loads((ROOT / "artifacts" / name).read_text())


def vn(x, digits=3):
    """Vietnamese decimal as LaTeX prints it; negatives appear as $-0{,}046$, so match the magnitude."""
    text = f"{abs(x):.{digits}f}".replace(".", ",")
    return text if x >= 0 else text.replace(",", "{,}")


def interval(stats):
    low, high = stats["ci95"]
    return f"[{vn(low)}; {vn(high)}]" if low >= 0 else f"[$-{vn(low)}$; {vn(high)}]"


TEMPORAL, AUTHOR, LTR, HOLDOUT = load("temporal_recovery.json"), load("author_next.json"), load("ltr.json"), load("holdout_fusion.json")
SPECTER = load("specter_eval.json")
DECISION = load("decision_model.json")
ROUND1, ROUND2, TOPIC = load("decision_model_round1.json"), load("decision_model_round2.json"), load("decision_query_model.json")


def temporal_claims():
    s = TEMPORAL["summary"]
    yield from (vn(s[m][k]) for m in s for k in ("ndcg", "recall"))
    yield vn(s["fusion"]["ndcg_app_default"])
    yield vn(s["hybrid"]["ndcg_app_default"])
    for key in ("paired_fusion_minus_hybrid", "paired_fusion_minus_cocitation", "paired_cocitation_minus_ppr"):
        yield "+" + vn(TEMPORAL[key]["mean"])
        yield interval(TEMPORAL[key])
    obs = TEMPORAL["observations"]
    halves = [sum(t["methods"]["fusion"]["ndcg"] for t in h) / len(h) for h in (obs[:110], obs[110:])]
    yield f"({vn(halves[0])} và {vn(halves[1])})"


def hit(task_list, method):
    return sum(bool(set(t["methods"][method]["top10"]) & set(t["targets"])) for t in task_list) / len(task_list)


def author_claims():
    for m, s in AUTHOR["summary"].items():
        yield from (vn(s[k]) for k in ("ndcg", "recall", "hit"))
        yield vn(hit(TEMPORAL["observations"], m))
    for key in ("paired_fusion_minus_hybrid", "paired_cocitation_minus_content"):
        yield "+" + vn(AUTHOR[key]["mean"])
        yield interval(AUTHOR[key])
    for name, run in LTR.items():
        yield vn(run["ndcg"])
        yield "+" + vn(run["ltr_minus_fusion"]["mean"]) + " " + interval(run["ltr_minus_fusion"])
        if name in ("reference_recovery", "author_next"):
            yield from (vn(run[k]) for k in ("recall", "hit"))
    weights = LTR["author_next"]["weights"]
    yield vn(weights["age"], 2)
    yield vn(weights["citation_count"], 2)


def holdout_claims():
    yield from (vn(v) for v in HOLDOUT["ndcg_at_10"].values())
    yield "+" + vn(HOLDOUT["paired"]["fusion_minus_hybrid"]["mean"])
    yield vn(HOLDOUT["paired"]["fusion_minus_cocitation"]["mean"])


def specter_claims():
    for protocol in ("reference_recovery", "author_next"):
        delta = SPECTER[protocol]["content"]["specter_minus_tfidf"]
        yield "+" + vn(delta["mean"]) + " " + interval(delta)
    yield vn(SPECTER["reference_recovery"]["fusion"]["specter_minus_tfidf"]["mean"])


@pytest.mark.parametrize("claim", [*temporal_claims(), *author_claims(), *holdout_claims(), *specter_claims()])
def test_report_quotes_artifact(claim):
    assert claim in REPORT, f"report/main.tex should quote {claim!r} from artifacts/; rerun the script and update the text"


@pytest.mark.parametrize("method", ["popular", "content", "graph", "hybrid", "cocitation", "fusion"])
def test_slides_quote_temporal_table(method):
    assert vn(TEMPORAL["summary"][method]["ndcg"]) in SLIDES


def test_slides_quote_ltr():
    for run in (LTR["reference_recovery"], LTR["author_next"]):
        assert vn(run["ndcg"]) in SLIDES
        assert interval(run["ltr_minus_fusion"]) in SLIDES


ROW_NAMES = {"popular": "Global PageRank", "content": "Content", "graph": "Personalized PageRank", "hybrid": "Hybrid",
             "cocitation": "Đồng trích dẫn", "fusion": "Fusion"}


@pytest.mark.parametrize("method", ROW_NAMES)
def test_e2_e2b_table_rows_match_artifacts(method):
    """A whole row must match, so a wrong cell cannot hide behind the same number quoted elsewhere."""
    t, a = TEMPORAL["summary"][method], AUTHOR["summary"][method]
    cells = [t["ndcg"], t["recall"], hit(TEMPORAL["observations"], method), a["ndcg"], a["recall"], a["hit"]]
    row = " & ".join([ROW_NAMES[method], *map(vn, cells)]) + "\\\\"
    assert row in REPORT, f"table tab:author should contain {row!r}"


def test_ltr_table_row_matches_artifact():
    e2, e2b = LTR["reference_recovery"], LTR["author_next"]
    cells = [e2["ndcg"], e2["recall"], e2["hit"], e2b["ndcg"], e2b["recall"], e2b["hit"]]
    assert "Hồi quy logistic & " + " & ".join(f"\\textbf{{{vn(c)}}}" for c in cells) in REPORT


@pytest.mark.parametrize("suite", ["E2", "E2b", "E3"])
def test_decision_table_row_matches_artifact(suite):
    r = DECISION["test"][suite]
    cells = f"{vn(r['fusion_ndcg'])} & \\textbf{{{vn(r['ndcg'])}}} & +{vn(r['minus_fusion']['mean'])} {interval(r['minus_fusion'])} & {vn(r['hit'])}\\\\"
    assert cells in REPORT, f"table tab:decision should contain {cells!r}"
    assert "+" + vn(r["minus_fusion"]["mean"]) + " " + interval(r["minus_fusion"]) in SLIDES


def test_decision_validation_numbers_are_quoted():
    for run in (ROUND1, ROUND2, TOPIC):
        for variant, value in run["validation_ndcg"].items():
            if variant in (run["chosen"], run.get("best_on_validation", run["chosen"])) or run is ROUND1:
                assert vn(value) in REPORT, (variant, vn(value))
    shipped = {suite: {k: v for k, v in r.items() if k != "hit5"} for suite, r in DECISION["test"].items()}
    assert shipped == ROUND1["test"], "the app model must be the round-1 model the report describes"


def test_round2_numbers_are_quoted():
    for suite in ("E2", "E2b", "E3"):
        assert vn(ROUND2["test"][suite]["ndcg"]) in REPORT


@pytest.mark.parametrize("suite", ["E2", "E2b", "E3"])
def test_topic_table_row_matches_artifacts(suite):
    t, d = TOPIC["test"][suite], DECISION["test"][suite]
    row = f"{suite} & {vn(t['fusion_ndcg'])} & {vn(t['bm25_only_ndcg'])} & {vn(d['ndcg'])} & \\textbf{{{vn(t['ndcg'])}}} & {vn(t['hit'])}\\\\"
    assert row in REPORT, f"table tab:topic should contain {row!r}"
    delta = "+" + vn(t["minus_fusion"]["mean"]) + " " + interval(t["minus_fusion"])
    assert delta in REPORT and delta in SLIDES
    assert f"{sum(DECISION['tasks_per_cutoff'].values()):,}".replace(",", ".") + " tác vụ" in REPORT


def hit_at(task_list, method, k):
    return sum(bool(set(t["methods"][method]["top10"][:k]) & set(t["targets"])) for t in task_list) / len(task_list)


def test_hit_table_matches_artifacts():
    fusion = {"E2": (hit_at(TEMPORAL["observations"], "fusion", 10), hit_at(TEMPORAL["observations"], "fusion", 5)),
              "E2b": (hit_at(AUTHOR["observations"], "fusion", 10), hit_at(AUTHOR["observations"], "fusion", 5)),
              "E3": (HOLDOUT["hits"]["fusion"]["hit10"], HOLDOUT["hits"]["fusion"]["hit5"])}
    for suite, (f10, f5) in fusion.items():
        d, t = DECISION["test"][suite], TOPIC["test"][suite]
        cells = [f10, d["hit"], t["hit"], f5, d["hit5"], t["hit5"]]
        row = f"{suite} & " + " & ".join(map(vn, cells)) + "\\\\"
        assert row in REPORT, f"table tab:hit should contain {row!r}"
        assert f"{round(100 * t['hit'])}\\%" in REPORT  # the abstract quotes the topic model in percent


SLM = {r: load(f) for r, f in (("1", "explainer_round1.json"), ("2", "explainer_round2.json"), ("3", "explainer.json"))}
SLM_AUDIT, SLM_AUDIT2 = load("explainer_audit.json"), load("explainer_round2_audit.json")
LATENCY = load("latency_recommend.json")["summary"]
SIZES = ("0.5b", "1.5b", "3b")


def s2(x):
    return f"{x:.2f}".replace(".", ",")


@pytest.mark.parametrize("round_", ["1", "2", "3"])
def test_slm_table_row_matches_artifacts(round_):
    models = SLM[round_]["models"]
    names = [next(n for n in models if f"-{size}-" in n) for size in SIZES if any(f"-{size}-" in n for n in models)]
    times = [s2(models[n]["generation_seconds"]["p50"]) for n in names]
    shown = [f"{round(100 * models[n]['shown'])}\\%" for n in names]
    if round_ == "3":
        times[0] = f"\\textbf{{{times[0]}}}"
    assert " / ".join(times) in REPORT and " / ".join(shown) in REPORT, (round_, times, shown)


def test_slm_choice_and_manual_audit_match_artifacts():
    assert SLM["3"]["chosen"] == SLM_AUDIT["model"] and SLM_AUDIT["passed"]
    assert f"\\textbf{{{SLM_AUDIT['correct']}/{SLM_AUDIT['read']}}}" in REPORT
    assert f"{SLM_AUDIT['distinct_papers_correct']}/{SLM_AUDIT['distinct_papers']} bài khác nhau" in REPORT
    small, mid = (SLM_AUDIT2["models"][n] for n in sorted(SLM_AUDIT2["models"]))
    assert f"{small['correct']}/{small['read']}; {mid['correct']}/{mid['read']}" in REPORT
    chosen = SLM["3"]["models"][SLM["3"]["chosen"]]
    assert f"{s2(chosen['first_token_seconds']['p50'])}\\,s để đọc prompt" in REPORT
    assert f"(p95 {s2(chosen['generation_seconds']['p95'])}\\,s)" in REPORT
    assert f"Retrieval mất {round(1000 * SLM['3']['retrieval_seconds']['p50'])}\\,ms" in REPORT


@pytest.mark.parametrize("name, label", [("cocitation", "Đồng trích dẫn"), ("hybrid", "Hybrid"), ("fusion", "Fusion"),
                                         ("decision", "Decision model"), ("decision_topic", "Decision model + đề tài")])
def test_recommend_latency_row_matches_artifact(name, label):
    row = LATENCY[name]
    assert f"{label} & {row['p50_ms']:.1f} & {row['p95_ms']:.1f}\\\\".replace(".", ",") in REPORT


def test_slm_latency_row_and_http_numbers_match_artifacts():
    chosen = SLM["3"]["models"][SLM["3"]["chosen"]]["generation_seconds"]
    assert f"Giải thích SLM 0.5B (một gợi ý) & {round(1000 * chosen['p50'])} & {round(1000 * chosen['p95'])}\\\\" in REPORT
    http = load("latency_http.json")
    assert f"decision model mất {round(http['recommend_ms']['p50'])}\\,ms ở trung vị" in REPORT
    assert f"trung vị là {s2(http['explain_seconds']['p50'])}\\,s" in REPORT
    assert f"một lượt gợi ý mất {round(LATENCY['decision']['p50_ms'])}\\,ms ở trung vị" in REPORT


SLM_RANKER, LLM_RANKER = load("slm_ranker.json"), load("llm_ranker.json")


def signed(x):
    return f"$-{vn(x)}$" if x < 0 else vn(x)


def test_language_model_ranker_table_matches_artifacts():
    """tab:llmrank and its slide: same 20 candidates, decision model and fusion rerun on the same tasks."""
    base = SLM_RANKER["baselines"]
    assert base == LLM_RANKER["baselines"]
    for suite in ("E2", "E2b", "E3"):  # the rerun reproduces the decision model's test numbers
        assert abs(base[suite]["decision"]["ndcg"] - DECISION["test"][suite]["ndcg"]) < 1e-9
    suites = ("E2", "E2b", "E3")
    rows = {"Fusion": [base[s]["fusion"]["ndcg"] for s in suites],
            "Decision": [base[s]["decision"]["ndcg"] for s in suites]}
    for name, model in SLM_RANKER["models"].items():
        size = "0.5B" if "-0.5b-" in name else "1.5B"
        cells = [model["suites"][s]["ndcg"] for s in suites]
        seconds = s2(model["seconds_per_request"]["p50"])
        assert f"SLM {size} & " + " & ".join(map(vn, cells)) + f" & {vn(model['suites']['E2']['hit10'])} & {seconds}\\,s\\\\" in REPORT
        assert f"SLM {size} & " + " & ".join(map(vn, cells)) + f" & {seconds}\\,s\\\\" in SLIDES
        assert not model["beats_decision"]
    llm = [LLM_RANKER["suites"][s]["ndcg"] for s in suites]
    seconds = s2(LLM_RANKER["seconds_per_request"]["p50"])
    assert "LLM GLM-5.3-flash & " + " & ".join(map(vn, llm)) + f" & {vn(LLM_RANKER['suites']['E2']['hit10'])} & {seconds}\\,s\\\\" in REPORT
    assert "LLM GLM-5.3 & " + " & ".join(map(vn, llm)) + f" & {seconds}\\,s\\\\" in SLIDES
    assert not LLM_RANKER["beats_decision"]
    for name, cells in rows.items():
        assert " & ".join(map(vn, cells)) in REPORT.replace("\\textbf{", "").replace("}", "")
    assert ", ".join(vn(base[s]["pool_recall"]) for s in suites[:2]) + f" và {vn(base['E3']['pool_recall'])} số bài đúng" in REPORT
    mid = next(m for n, m in SLM_RANKER["models"].items() if "-1.5b-" in n)["suites"]
    for s in ("E2", "E3"):
        d = mid[s]["minus_decision"]
        assert f"{signed(d['mean'])} [{signed(d['ci95'][0])}; {signed(d['ci95'][1])}]" in REPORT
    assert ", ".join(signed(LLM_RANKER["suites"][s]["minus_fusion"]["mean"]) for s in suites[:2]) in REPORT
    assert ", ".join(signed(LLM_RANKER["suites"][s]["minus_decision"]["mean"]) for s in suites[:2]) in REPORT
    empty = sum(r["named"] == 0 for r in LLM_RANKER["rows"])
    assert f"{empty}/{len(LLM_RANKER['rows'])} câu trả lời" in REPORT
