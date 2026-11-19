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
             "cocitation": "Đồng trích dẫn", "fusion": "Fusion (mặc định)"}


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
    v = DECISION["validation_ndcg"]
    for variant in v:
        assert vn(v[variant]) in REPORT
    assert f"{sum(DECISION['tasks_per_cutoff'].values()):,}".replace(",", ".") + " tác vụ" in REPORT
