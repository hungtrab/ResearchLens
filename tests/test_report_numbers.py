"""Every recommendation number quoted in the report (English decimals) and slides (Vietnamese decimals) must come from an artifact."""
import json
from pathlib import Path

import numpy as np
import pytest

from researchlens.evaluation import bootstrap_mean

ROOT = Path(__file__).resolve().parents[1]
REPORT = (ROOT / "report/main.tex").read_text()
SLIDES = (ROOT / "slides/main.tex").read_text()


def load(name):
    return json.loads((ROOT / "artifacts" / name).read_text())


def vn(x, digits=3):
    """Vietnamese decimal as LaTeX prints it; negatives appear as $-0{,}046$, so match the magnitude."""
    text = f"{abs(x):.{digits}f}".replace(".", ",")
    return text if x >= 0 else text.replace(",", "{,}")


def en(x, digits=3):
    """English decimal; negatives are typeset as $-0.046$, so match the magnitude."""
    return f"{abs(x):.{digits}f}"


def signed_en(x):
    return f"$-{en(x)}$" if x < 0 else en(x)


def interval_en(stats):
    low, high = stats["ci95"]
    return f"[{signed_en(low)}, {signed_en(high)}]"


def span_en(d):
    return f"{signed_en(d['mean'])} [{signed_en(d['ci95'][0])}, {signed_en(d['ci95'][1])}]"


def interval(stats):
    low, high = stats["ci95"]
    return f"[{vn(low)}; {vn(high)}]" if low >= 0 else f"[$-{vn(low)}$; {vn(high)}]"


TEMPORAL, AUTHOR, LTR, HOLDOUT = load("temporal_recovery.json"), load("author_next.json"), load("ltr.json"), load("holdout_fusion.json")
SPECTER = load("specter_eval.json")
DECISION = load("decision_model.json")
ROUND1, ROUND2, TOPIC = load("decision_model_round1.json"), load("decision_model_round2.json"), load("decision_query_model.json")


def temporal_claims():
    s = TEMPORAL["summary"]
    yield from (en(s[m][k]) for m in s for k in ("ndcg", "recall"))
    yield en(s["fusion"]["ndcg_app_default"])
    yield en(s["hybrid"]["ndcg_app_default"])
    for key in ("paired_fusion_minus_hybrid", "paired_fusion_minus_cocitation", "paired_cocitation_minus_ppr"):
        yield "+" + en(TEMPORAL[key]["mean"]) + " " + interval_en(TEMPORAL[key])
    obs = TEMPORAL["observations"]
    halves = [sum(t["methods"]["fusion"]["ndcg"] for t in h) / len(h) for h in (obs[:110], obs[110:])]
    yield f"({en(halves[0])} and {en(halves[1])})"


def hit(task_list, method):
    return sum(bool(set(t["methods"][method]["top10"]) & set(t["targets"])) for t in task_list) / len(task_list)


def author_claims():
    for m, s in AUTHOR["summary"].items():
        yield from (en(s[k]) for k in ("ndcg", "recall", "hit"))
        yield en(hit(TEMPORAL["observations"], m))
    for key in ("paired_fusion_minus_hybrid", "paired_cocitation_minus_content"):
        yield "+" + en(AUTHOR[key]["mean"]) + " " + interval_en(AUTHOR[key])
    for name, run in LTR.items():
        yield en(run["ndcg"])
        yield "+" + en(run["ltr_minus_fusion"]["mean"]) + " " + interval_en(run["ltr_minus_fusion"])
        if name in ("reference_recovery", "author_next"):
            yield from (en(run[k]) for k in ("recall", "hit"))
    weights = LTR["author_next"]["weights"]
    yield f"age (${'-' if weights['age'] < 0 else ''}{en(weights['age'], 2)}$)"
    yield f"citation count ({en(weights['citation_count'], 2)})"


def holdout_claims():
    yield from (en(v) for v in HOLDOUT["ndcg_at_10"].values())
    yield "+" + en(HOLDOUT["paired"]["fusion_minus_hybrid"]["mean"]) + " " + interval_en(HOLDOUT["paired"]["fusion_minus_hybrid"])
    behind = HOLDOUT["paired"]["fusion_minus_cocitation"]
    assert behind["ci95"][1] < 0
    yield f"beats fusion by {en(behind['mean'])} [{en(behind['ci95'][1])}, {en(behind['ci95'][0])}]"


def specter_claims():
    for protocol in ("reference_recovery", "author_next"):
        delta = SPECTER[protocol]["content"]["specter_minus_tfidf"]
        yield "+" + en(delta["mean"]) + " " + interval_en(delta)
    worse = SPECTER["reference_recovery"]["fusion"]["specter_minus_tfidf"]
    yield f"on E2 by {en(worse['mean'])} [{en(worse['ci95'][1])}, {en(worse['ci95'][0])}]"


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
             "cocitation": "Cocitation", "fusion": "Fusion"}


@pytest.mark.parametrize("method", ROW_NAMES)
def test_e2_e2b_table_rows_match_artifacts(method):
    """A whole row must match, so a wrong cell cannot hide behind the same number quoted elsewhere."""
    t, a = TEMPORAL["summary"][method], AUTHOR["summary"][method]
    cells = [t["ndcg"], t["recall"], hit(TEMPORAL["observations"], method), a["ndcg"], a["recall"], a["hit"]]
    row = " & ".join([ROW_NAMES[method], *map(en, cells)]) + "\\\\"
    assert row in REPORT, f"table tab:author should contain {row!r}"


def test_ltr_table_row_matches_artifact():
    e2, e2b = LTR["reference_recovery"], LTR["author_next"]
    cells = [e2["ndcg"], e2["recall"], e2["hit"], e2b["ndcg"], e2b["recall"], e2b["hit"]]
    assert "Logistic regression & " + " & ".join(f"\\textbf{{{en(c)}}}" for c in cells) in REPORT


@pytest.mark.parametrize("suite", ["E2", "E2b", "E3"])
def test_decision_table_row_matches_artifact(suite):
    r = DECISION["test"][suite]
    cells = f"{en(r['fusion_ndcg'])} & \\textbf{{{en(r['ndcg'])}}} & +{en(r['minus_fusion']['mean'])} {interval_en(r['minus_fusion'])} & {en(r['hit'])}\\\\"
    assert cells in REPORT, f"table tab:decision should contain {cells!r}"
    assert "+" + vn(r["minus_fusion"]["mean"]) + " " + interval(r["minus_fusion"]) in SLIDES


def test_decision_validation_numbers_are_quoted():
    for run in (ROUND1, ROUND2, TOPIC):
        for variant, value in run["validation_ndcg"].items():
            if variant in (run["chosen"], run.get("best_on_validation", run["chosen"])) or run is ROUND1:
                assert en(value) in REPORT, (variant, en(value))
    shipped = {suite: {k: v for k, v in r.items() if k != "hit5"} for suite, r in DECISION["test"].items()}
    assert shipped == ROUND1["test"], "the app model must be the round-1 model the report describes"


def test_round2_numbers_are_quoted():
    for suite in ("E2", "E2b", "E3"):
        assert en(ROUND2["test"][suite]["ndcg"]) in REPORT


@pytest.mark.parametrize("suite", ["E2", "E2b", "E3"])
def test_topic_table_row_matches_artifacts(suite):
    t, d = TOPIC["test"][suite], DECISION["test"][suite]
    row = f"{suite} & {en(t['fusion_ndcg'])} & {en(t['bm25_only_ndcg'])} & {en(d['ndcg'])} & \\textbf{{{en(t['ndcg'])}}} & {en(t['hit'])}\\\\"
    assert row in REPORT, f"table tab:topic should contain {row!r}"
    assert "+" + en(t["minus_fusion"]["mean"]) + " " + interval_en(t["minus_fusion"]) in REPORT
    assert "+" + vn(t["minus_fusion"]["mean"]) + " " + interval(t["minus_fusion"]) in SLIDES
    assert f"{sum(DECISION['tasks_per_cutoff'].values()):,} tasks" in REPORT


def hit_at(task_list, method, k):
    return sum(bool(set(t["methods"][method]["top10"][:k]) & set(t["targets"])) for t in task_list) / len(task_list)


def test_hit_table_matches_artifacts():
    fusion = {"E2": (hit_at(TEMPORAL["observations"], "fusion", 10), hit_at(TEMPORAL["observations"], "fusion", 5)),
              "E2b": (hit_at(AUTHOR["observations"], "fusion", 10), hit_at(AUTHOR["observations"], "fusion", 5)),
              "E3": (HOLDOUT["hits"]["fusion"]["hit10"], HOLDOUT["hits"]["fusion"]["hit5"])}
    for suite, (f10, f5) in fusion.items():
        d, t = DECISION["test"][suite], TOPIC["test"][suite]
        cells = [f10, d["hit"], t["hit"], f5, d["hit5"], t["hit5"]]
        row = f"{suite} & " + " & ".join(map(en, cells)) + "\\\\"
        assert row in REPORT, f"table tab:hit should contain {row!r}"
        assert f"{round(100 * t['hit'])}\\%" in REPORT  # the abstract quotes the topic model in percent


SLM = {r: load(f) for r, f in (("1", "explainer_round1.json"), ("2", "explainer_round2.json"), ("3", "explainer.json"))}
SLM_AUDIT, SLM_AUDIT2 = load("explainer_audit.json"), load("explainer_round2_audit.json")
LATENCY = load("latency_recommend.json")["summary"]
SIZES = ("0.5b", "1.5b", "3b")


def s2(x):
    return f"{x:.2f}".replace(".", ",")


def s2en(x):
    return f"{x:.2f}"


@pytest.mark.parametrize("round_", ["1", "2", "3"])
def test_slm_table_row_matches_artifacts(round_):
    models = SLM[round_]["models"]
    names = [next(n for n in models if f"-{size}-" in n) for size in SIZES if any(f"-{size}-" in n for n in models)]
    times = [s2en(models[n]["generation_seconds"]["p50"]) for n in names]
    shown = [f"{round(100 * models[n]['shown'])}\\%" for n in names]
    if round_ == "3":
        times[0] = f"\\textbf{{{times[0]}}}"
    assert " / ".join(times) in REPORT and " / ".join(shown) in REPORT, (round_, times, shown)


def test_slm_choice_and_manual_audit_match_artifacts():
    assert SLM["3"]["chosen"] == SLM_AUDIT["model"] and SLM_AUDIT["passed"]
    assert f"\\textbf{{{SLM_AUDIT['correct']}/{SLM_AUDIT['read']}}}" in REPORT
    assert f"{SLM_AUDIT['distinct_papers_correct']}/{SLM_AUDIT['distinct_papers']} distinct papers" in REPORT
    small, mid = (SLM_AUDIT2["models"][n] for n in sorted(SLM_AUDIT2["models"]))
    assert f"{small['correct']}/{small['read']}; {mid['correct']}/{mid['read']}" in REPORT
    chosen = SLM["3"]["models"][SLM["3"]["chosen"]]
    assert f"{s2en(chosen['first_token_seconds']['p50'])}\\,s to read the prompt" in REPORT
    assert f"(p95 {s2en(chosen['generation_seconds']['p95'])}\\,s)" in REPORT
    assert f"Retrieval takes {round(1000 * SLM['3']['retrieval_seconds']['p50'])}\\,ms" in REPORT


@pytest.mark.parametrize("name, label", [("cocitation", "Cocitation"), ("hybrid", "Hybrid"), ("fusion", "Fusion"),
                                         ("decision", "Decision model"), ("decision_topic", "Decision model + topic")])
def test_recommend_latency_row_matches_artifact(name, label):
    row = LATENCY[name]
    assert f"{label} & {row['p50_ms']:.1f} & {row['p95_ms']:.1f}\\\\" in REPORT


def test_slm_latency_row_and_http_numbers_match_artifacts():
    chosen = SLM["3"]["models"][SLM["3"]["chosen"]]["generation_seconds"]
    assert f"SLM 0.5B explanation (one item) & {round(1000 * chosen['p50'])} & {round(1000 * chosen['p95'])}\\\\" in REPORT
    http = load("latency_http.json")
    assert f"the decision model takes {round(http['recommend_ms']['p50'])}\\,ms at the median" in REPORT
    assert f"the median is {s2en(http['explain_seconds']['p50'])}\\,s" in REPORT
    assert f"One recommendation request takes {round(LATENCY['decision']['p50_ms'])}\\,ms at the median" in REPORT


SLM_RANKER, LLM_RANKER = load("slm_ranker.json"), load("llm_ranker.json")
GRAPH_RANKER = load("llm_ranker_graph.json")


def test_language_model_ranker_table_matches_artifacts():
    """tab:llmrank and its slide: same 20 candidates, decision model and fusion rerun on the same tasks; the
    decision model both on those 20 candidates (the comparison) and on its own candidates as the app runs it."""
    base = SLM_RANKER["baselines"]
    assert base == LLM_RANKER["baselines"]
    for suite in ("E2", "E2b", "E3"):  # the rerun reproduces the decision model's test numbers
        assert abs(base[suite]["decision"]["ndcg"] - DECISION["test"][suite]["ndcg"]) < 1e-9
    suites = ("E2", "E2b", "E3")
    cells = lambda key: " & ".join(en(base[s][key]["ndcg"]) for s in suites) + f" & {en(base['E2'][key]['hit10'])}"
    assert f"Fusion & {cells('fusion')} & " in REPORT and f"Decision model (app) & {cells('decision')} & " in REPORT
    bold = lambda key, hit, f=en: " & ".join("\\textbf{%s}" % f(x) for x in [base[s][key]["ndcg"] for s in suites]
                                       + ([base["E2"][key]["hit10"]] if hit else []))
    assert f"\\textbf{{Decision model}} & {bold('decision_pool', True)} & " in REPORT
    assert f"Decision (app) & {' & '.join(vn(base[s]['decision']['ndcg']) for s in suites)} & " in SLIDES
    assert f"\\textbf{{Decision (cùng 20)}} & {bold('decision_pool', False, vn)} & " in SLIDES
    for name, model in SLM_RANKER["models"].items():
        size = "0.5B" if "-0.5b-" in name else "1.5B"
        cells = [model["suites"][s]["ndcg"] for s in suites]
        seconds = model["seconds_per_request"]["p50"]
        assert f"SLM {size} & " + " & ".join(map(en, cells)) + f" & {en(model['suites']['E2']['hit10'])} & {s2en(seconds)}\\,s\\\\" in REPORT
        assert f"SLM {size} & " + " & ".join(map(vn, cells)) + f" & {s2(seconds)}\\,s\\\\" in SLIDES
        assert not any(model["beats_decision"].values())
    llm = [LLM_RANKER["suites"][s]["ndcg"] for s in suites]
    seconds = LLM_RANKER["seconds_per_request"]["p50"]
    assert "LLM, text only & " + " & ".join(map(en, llm)) + f" & {en(LLM_RANKER['suites']['E2']['hit10'])} & {s2en(seconds)}\\,s\\\\" in REPORT
    assert "LLM GLM-5.3 & " + " & ".join(map(vn, llm)) + f" & {s2(seconds)}\\,s\\\\" in SLIDES
    assert not any(LLM_RANKER["beats_decision"].values())
    assert ", ".join(en(base[s]["pool_recall"]) for s in suites[:2]) + f" and {en(base['E3']['pool_recall'])} of the right papers" in REPORT
    mid = next(m for n, m in SLM_RANKER["models"].items() if "-1.5b-" in n)["suites"]
    for s in suites:
        d = mid[s]["minus_decision_pool"]
        assert span_en(d) in REPORT
    assert ", ".join(signed_en(LLM_RANKER["suites"][s]["minus_fusion"]["mean"]) for s in suites[:2]) in REPORT
    assert ", ".join(signed_en(LLM_RANKER["suites"][s]["minus_decision_pool"]["mean"]) for s in suites[:2]) in REPORT
    d = LLM_RANKER["suites"]["E3"]["minus_decision_pool"]
    assert d["ci95"][1] < 0 and span_en(d) in REPORT
    assert LLM_RANKER["answers_naming_all"] == 1 and f"All {len(LLM_RANKER['rows'])} answers" in REPORT


def test_graph_rag_ranker_matches_artifacts():
    """LLM + graph RAG row, its deltas vs decision, fusion and text-only LLM, and how many answers ran out."""
    suites = ("E2", "E2b", "E3")
    assert GRAPH_RANKER["graph_evidence"] and GRAPH_RANKER["baselines"] == LLM_RANKER["baselines"]
    cells = lambda f: " & ".join(f(GRAPH_RANKER["suites"][s]["ndcg"]) for s in suites)
    seconds = GRAPH_RANKER["seconds_per_request"]["p50"]
    assert f"LLM + graph RAG & {cells(en)} & {en(GRAPH_RANKER['suites']['E2']['hit10'])} & {s2en(seconds)}\\,s\\\\" in REPORT
    assert f"LLM + graph RAG & {cells(vn)} & {s2(seconds)}\\,s\\\\" in SLIDES
    assert not any(GRAPH_RANKER["beats_decision"].values())
    span = span_en
    for s in suites:
        assert span(GRAPH_RANKER["suites"][s]["minus_decision_pool"]) in REPORT
    assert span(GRAPH_RANKER["suites"]["E3"]["minus_fusion"]) in REPORT
    gain = {s: [g["llm"] - t["llm"] for g, t in zip(GRAPH_RANKER["rows"], LLM_RANKER["rows"]) if g["suite"] == s]
            for s in suites}
    assert span(dict(mean=float(np.mean(gain["E3"])), ci95=bootstrap_mean(gain["E3"]))) in REPORT
    assert f"the differences are {signed_en(np.mean(gain['E2']))} and {signed_en(np.mean(gain['E2b']))}, with intervals that include zero" in REPORT
    empty = sum(r["named"] == 0 for r in GRAPH_RANKER["rows"])
    assert f"In {empty}/{len(GRAPH_RANKER['rows'])} answers the model still used all 16,384 tokens" in REPORT


SLM_100, GRAPH_100 = load("slm_ranker_pool100.json"), load("llm_ranker_graph_pool100.json")


def test_hundred_candidate_rankers_match_artifacts():
    """The 100-candidate block of tab:llmrank, its slide rows, and the deltas quoted in the text."""
    suites = ("E2", "E2b", "E3")
    base = GRAPH_100["baselines"]
    assert SLM_100["pool"] == GRAPH_100["pool"] == 100 and SLM_100["baselines"] == base
    ndcgs = lambda per_suite, key=None: [(per_suite[s][key] if key else per_suite[s])["ndcg"] for s in suites]
    decision = ndcgs(base, "decision_pool")
    assert f"Decision model (100) & {' & '.join(map(en, decision))} & {en(base['E2']['decision_pool']['hit10'])} & " in REPORT
    assert f"Decision (cùng 100) & {' & '.join(map(vn, decision))} & " in SLIDES
    for name, model in SLM_100["models"].items():
        size = "0.5B" if "-0.5b-" in name else "1.5B"
        row = f"SLM {size} (100) & {' & '.join(map(en, ndcgs(model['suites'])))}"
        assert f"{row} & {en(model['suites']['E2']['hit10'])} & {s2en(model['seconds_per_request']['p50'])}\\,s\\\\" in REPORT
        assert not any(model["beats_decision"].values())
    assert f"SLM 1.5B (100) & " in SLIDES
    llm = GRAPH_100["suites"]
    bold = lambda f: " & ".join("\\textbf{%s}" % f(x) for x in ndcgs(llm))
    seconds = GRAPH_100["seconds_per_request"]["p50"]
    assert f"\\textbf{{LLM + graph RAG (100)}} & {bold(en)} & {en(llm['E2']['hit10'])} & {s2en(seconds)}\\,s\\\\" in REPORT
    assert f"\\textbf{{LLM + graph RAG (100)}} & {bold(vn)} & {s2(seconds)}\\,s\\\\" in SLIDES
    assert not any(GRAPH_100["beats_decision"].values())
    span = span_en
    assert ", ".join(span(llm[s]["minus_decision_pool"]) for s in suites[:2]) in REPORT
    assert ", ".join(span(llm[s]["minus_fusion"]) for s in suites[:2]) in REPORT
    assert ", ".join(en(base[s]["pool_recall"]) for s in suites[:2]) + f" and {en(base['E3']['pool_recall'])};" in REPORT
    empty = sum(r["named"] == 0 for r in GRAPH_100["rows"])
    assert f"In {empty}/{len(GRAPH_100['rows'])} answers the model used all 16,384 tokens" in REPORT
