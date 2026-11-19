"""The SLM + RAG explanation: what evidence it retrieves, which answers it refuses to show, and the API."""
import pytest

from researchlens.explain import Explainer, check, passages, prompt


def test_evidence_names_the_paper_the_saved_paper_and_the_paper_citing_both(engine):
    evidence, fact = passages(engine, ["P1"], "P3")
    assert evidence[0].startswith("Recommended: Neural information retrieval (2017). BM25 and BERT")
    assert evidence[1].startswith("Saved by the reader: Graph neural networks (2018).")
    assert evidence[2:] == ["Graph neural network benchmarks cites both [1] and [2].", "[2] cites [1]."]
    # The app states the graph facts itself, each citing the evidence line it comes from.
    assert fact == ("Closest saved paper: Graph neural networks [2]; Graph neural network benchmarks cites it "
                    "together with your saved paper [3]; your saved paper cites it [4].")


def test_a_paper_with_no_graph_link_still_names_the_closest_saved_paper(engine):
    assert passages(engine, ["P5"], "P1")[1] == "Closest saved paper: Opinion mining in reviews [2]."


def test_closest_saved_paper_is_the_one_linked_to_the_recommendation(engine):
    # P4 cites P3; P5 has no link to it, so P4 is the saved paper the explanation compares against.
    assert passages(engine, ["P5", "P4"], "P3")[0][1].startswith("Saved by the reader: Graph based information retrieval")


def test_the_model_sees_only_the_recommended_paper_after_one_worked_example():
    messages = prompt(["Recommended: A", "Saved by the reader: B", "[2] cites [1]."])
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"] == "[1] Recommended: A"


EVIDENCE = ["Recommended: Neural information retrieval (2017). BM25 and BERT support document search.",
            "Saved by the reader: Graph neural networks (2018)."]
TITLE = "Neural information retrieval"


@pytest.mark.parametrize("answer, shown", [
    ("It supports document search with BM25 and BERT [1].", True),
    ("It supports document search with BM25 and BERT [1][2].", False),  # the model was given only [1]
    ("It supports document search with BM25 and BERT.", False),  # cites nothing
    ("Neural information retrieval (2017) [1].", False),  # only repeats the title
    ("It supports document search with BM25 on 35 benchmarks [1].", False),  # invented number
    ("Published in 2017, it supports document search with BERT [1].", True),
])
def test_only_summaries_that_cite_the_paper_say_more_than_its_title_and_use_real_numbers_are_shown(answer, shown):
    assert check(answer, EVIDENCE[0], TITLE)["shown"] is shown


def test_novel_words_measures_content_absent_from_the_evidence():
    assert check("It supports document search [1].", EVIDENCE[0], TITLE)["novel_words"] == 0
    assert check("It achieves remarkable translation quality [1].", EVIDENCE[0], TITLE)["novel_words"] == 1


def test_the_shown_text_is_the_first_sentence_then_the_graph_facts():
    good = Explainer(lambda m: "It supports document search with BERT [1]. It also does more.", "fake").explain(
        EVIDENCE, "Fact [3].", TITLE)
    assert good["text"] == "It supports document search with BERT [1]. Fact [3]." and good["model"] == "fake"


def test_a_rejected_sentence_is_kept_as_draft_but_not_shown():
    bad = Explainer(lambda m: "It cites 99 papers [7].", "fake").explain(EVIDENCE, "Fact [3].", TITLE)
    assert bad["summary"] is None and bad["draft"] == "It cites 99 papers [7]." and not bad["checks"]["shown"]
    assert bad["text"] == "Fact [3]."  # the graph facts the app wrote are still shown


def test_api_explains_one_recommendation_from_retrieved_evidence(client):
    seen = []
    client.app.state.explainer = Explainer(lambda m: seen.append(m) or "It supports document search with BERT [1].", "fake")
    result = client.post("/api/recommendations/explain", json=dict(seeds=["P1"], paper_id="P3")).json()
    assert result["draft"] == "It supports document search with BERT [1]." and result["checks"]["shown"]
    assert result["evidence"][0].startswith("Recommended: Neural information retrieval")
    assert seen[0][-1]["content"].startswith("[1] Recommended: Neural information retrieval")
    assert "Graph neural networks" not in seen[0][-1]["content"]  # the saved paper is not shown to the model
    assert result["text"].endswith("your saved paper cites it [4].")
    assert result["seconds"] >= 0 and result["retrieval_seconds"] >= 0
    assert client.get("/api/health").json()["slm"] is True


def test_api_reports_a_missing_language_model_and_bad_ids(client):
    client.app.state.explainer, client.app.state.slm_path = None, None
    assert client.get("/api/health").json()["slm"] is False
    response = client.post("/api/recommendations/explain", json=dict(seeds=["P1"], paper_id="P3"))
    assert response.status_code == 503 and "uv sync --extra slm" in response.json()["detail"]
    assert client.post("/api/recommendations/explain", json=dict(seeds=["P1"], paper_id="nope")).status_code == 404
    assert client.post("/api/recommendations/explain", json=dict(seeds=["nope"], paper_id="P3")).status_code == 422


def test_a_recommendation_that_cites_the_saved_paper_says_so(engine):
    # P1 cites P3; with P3 saved, P1 is the recommendation that cites it.
    evidence, fact = passages(engine, ["P3"], "P1")
    assert "[1] cites [2]." in evidence and "it cites your saved paper" in fact


def slm_installed():
    import importlib.util
    from researchlens.explain import default_model_path
    path = default_model_path()
    return bool(path and path.exists() and importlib.util.find_spec("llama_cpp"))


@pytest.mark.skipif(not slm_installed(), reason="llama.cpp or the GGUF model is not installed (make slm)")
def test_the_installed_small_language_model_summarizes_a_real_abstract(engine):
    from researchlens.explain import default_model_path
    evidence, fact = passages(engine, ["P1"], "P3")
    result = Explainer.load(default_model_path()).explain(evidence, fact, "Neural information retrieval")
    assert result["text"].endswith(fact) and result["draft"] and result["seconds"] < 10
