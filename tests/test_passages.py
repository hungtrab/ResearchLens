"""Passage retrieval with hand-calculated vectors, exact quotes and isolated API."""
import hashlib
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from researchlens.api import create_app
from researchlens.passages import PassageIndex, split_sentences


@pytest.mark.parametrize("text", [None, "", " \n\t "])
def test_empty_abstract_has_no_sentence_choices(text):
    assert split_sentences(text) == []


def test_sentence_splitting_preserves_verbatim_unicode_and_internal_whitespace():
    first = "The café dataset contains  graph experiments."
    second = "Retrieval handles naïve baselines!"
    third = "Does this work?"
    text = f"  {first}  {second}\n{third}\n\nA final paragraph without punctuation  "
    result = split_sentences(text)
    assert result == [first, second, third, "A final paragraph without punctuation"]
    assert all(sentence in text for sentence in result)


def test_scientific_abbreviations_do_not_create_spurious_sentence_positions():
    first = "Dr. Smith and Lee et al. (2020) use Fig. 2 and Eq. 3 for the model."
    second = "We compare a baseline, e.g. GCN, against graph attention."
    assert split_sentences(first + " " + second) == [first, second]


@pytest.fixture
def passage_fixture(app_config, papers):
    ids = ["S", "S_COPY", "A_COPY", "A", "C", "B"]
    titles = ["Seed methods", "SEED: Methods!", "GRAPH: Experiments!",
              "Graph experiments", "Alternate experiments", "Retrieval experiments"]
    choices = [
        ["Graph methods reveal connected patterns.", "Retrieval methods rank documents."],
        ["Another edition describes the seed method."],
        ["The duplicate edition has a different abstract."],
        ["The café dataset contains graph experiments.", "A second experiment retrieves relevant documents."],
        ["An alternative method combines graphs with retrieval."],
        ["This source passage concerns graph evidence."],
    ]
    rows = [dict(papers[0], id=pid, title=title, abstract=" ".join(sentences), year=2020+i, references=[])
            for i, (pid, title, sentences) in enumerate(zip(ids, titles, choices))]
    rows.append(dict(papers[0], id="UNINDEXED", title="Metadata without abstract", abstract="", references=[]))
    corpus = app_config["data_dir"] / "papers.jsonl"
    corpus.write_text("".join(json.dumps(p, ensure_ascii=False)+"\n" for p in rows), encoding="utf8")
    (app_config["data_dir"] / "reviews.jsonl").unlink()
    texts = [sentence for group in choices for sentence in group]
    bundle = dict(
        version=1, corpus_sha=hashlib.sha256(corpus.read_bytes()).hexdigest(),
        ids=np.array(ids), titles=np.array(titles),
        texts_utf8=np.frombuffer(json.dumps(texts, ensure_ascii=False).encode("utf8"), dtype=np.uint8),
        offsets=np.array([0, 2, 3, 4, 6, 7, 8], dtype=np.int64),
        documents=np.array([[1, 0], [1, 0], [-1, 0], [1, 0], [.6, .8], [0, 1]], dtype=np.float32),
        sentences=np.array([[1, 0], [0, 1], [1, 0], [-1, 0], [1, 0], [0, 1], [.6, .8], [1, 0]], dtype=np.float32),
    )
    path = app_config["data_dir"] / "passages.npz"
    np.savez_compressed(path, **bundle)
    return dict(path=path, corpus=corpus, bundle=bundle, rows=rows, choices=dict(zip(ids, choices)))


def test_index_load_choices_and_sentence_offsets_are_exact(passage_fixture):
    fixture = passage_fixture
    index = PassageIndex.load(fixture["path"], fixture["corpus"])
    for pid, expected in fixture["choices"].items():
        assert index.choices(pid) == expected
    returned = index.choices("S")
    returned.append("Caller mutation")
    assert index.choices("S") == fixture["choices"]["S"]
    with pytest.raises(KeyError):
        index.choices("UNINDEXED")


def test_rank_combines_global_similarity_and_best_passage_with_title_dedup(passage_fixture):
    fixture = passage_fixture
    index = PassageIndex.load(fixture["path"], fixture["corpus"])
    result = index.search("S", 0, limit=100)
    assert [item["id"] for item in result] == ["A", "C", "B"]
    # A: (1 + 1)/2; C: (.6 + .6)/2; B: (0 + 1)/2.
    assert [item["score"] for item in result] == pytest.approx([1, .6, .5])
    assert result[0]["quote"] == fixture["choices"]["A"][0]
    assert result[1]["quote"] == fixture["choices"]["C"][0]
    by_id = {paper["id"]: paper for paper in fixture["rows"]}
    assert all(item["quote"] in by_id[item["id"]]["abstract"] for item in result)
    assert index.search("S", 0, limit=1) == result[:1]
    assert index.search("S", 0) == result


def test_selected_sentence_changes_ranking_and_evidence_without_offset_confusion(passage_fixture):
    fixture = passage_fixture
    index = PassageIndex.load(fixture["path"], fixture["corpus"])
    result = index.search("S", 1)
    assert result[0]["id"] == "C" and result[0]["score"] == pytest.approx(.8)
    assert {item["id"] for item in result} == {"A", "B", "C"}
    candidate = next(item for item in result if item["id"] == "A")
    assert candidate["quote"] == fixture["choices"]["A"][1]
    assert candidate["score"] == pytest.approx(.5)
    assert index.search("S", 1) == result


@pytest.mark.parametrize("sentence_index", [-1, 2, 999])
def test_search_rejects_sentence_outside_this_paper(passage_fixture, sentence_index):
    index = PassageIndex.load(passage_fixture["path"], passage_fixture["corpus"])
    with pytest.raises(ValueError, match="sentence"):
        index.search("S", sentence_index)


@pytest.mark.parametrize("limit", [0, -1])
def test_search_requires_positive_result_limit(passage_fixture, limit):
    index = PassageIndex.load(passage_fixture["path"], passage_fixture["corpus"])
    with pytest.raises(ValueError, match="positive"):
        index.search("S", 0, limit=limit)


def test_index_with_only_seed_title_variants_has_no_candidates(passage_fixture):
    fixture = passage_fixture
    bundle = dict(fixture["bundle"], ids=fixture["bundle"]["ids"][:2], titles=fixture["bundle"]["titles"][:2],
                  texts_utf8=np.frombuffer(json.dumps(fixture["choices"]["S"]+fixture["choices"]["S_COPY"]).encode(), dtype=np.uint8),
                  offsets=np.array([0, 2, 3]), documents=fixture["bundle"]["documents"][:2], sentences=fixture["bundle"]["sentences"][:3])
    np.savez_compressed(fixture["path"], **bundle)
    assert PassageIndex.load(fixture["path"], fixture["corpus"]).search("S", 0) == []


@pytest.mark.parametrize("change", [
    {"version": 99}, {"corpus_sha": "wrong-digest"},
    {"ids": np.array(["S", "S", "A_COPY", "A", "C", "B"])},
    {"offsets": np.array([0, 2, 3, 4, 6, 7, 9])},
    {"offsets": np.array([0, 2, 3, 3, 6, 7, 8])},
    {"offsets": np.array([0., 2., 3., 4., 6., 7., 8.])},
    {"documents": np.zeros((5, 2))}, {"sentences": np.zeros((7, 2))},
])
def test_index_rejects_wrong_provenance_or_inconsistent_shapes(passage_fixture, change):
    fixture = passage_fixture
    np.savez_compressed(fixture["path"], **dict(fixture["bundle"], **change))
    with pytest.raises(ValueError):
        PassageIndex.load(fixture["path"], fixture["corpus"])


def test_index_corpus_digest_detects_even_unindexed_metadata_changes(passage_fixture):
    fixture = passage_fixture
    fixture["corpus"].write_text(fixture["corpus"].read_text()+"\n")
    with pytest.raises(ValueError, match="corpus"):
        PassageIndex.load(fixture["path"], fixture["corpus"])


def test_passage_api_uses_index_choices_and_hydrates_current_paper_metadata(app_config, passage_fixture):
    fixture = passage_fixture
    with TestClient(create_app(**app_config)) as client:
        response = client.get("/api/papers/S/passages")
        assert response.status_code == 200
        assert response.json() == {"available": True, "sentences": fixture["choices"]["S"]}
        result = client.get("/api/papers/S/passage-matches", params={"sentence": 0})
        assert result.status_code == 200
        items = result.json()["items"]
        assert [item["id"] for item in items] == ["A", "C", "B"]
        by_id = {paper["id"]: paper for paper in fixture["rows"]}
        for item in items:
            assert item["title"] == by_id[item["id"]]["title"]
            assert item["year"] == by_id[item["id"]]["year"]
            assert item["quote"] in by_id[item["id"]]["abstract"]
        assert client.get("/api/papers/UNINDEXED/passages").json() == {"available": False, "sentences": []}
        assert client.get("/api/papers/UNINDEXED/passage-matches?sentence=0").status_code == 409
        assert client.get("/api/papers/missing/passages").status_code == 404
        assert client.get("/api/papers/missing/passage-matches?sentence=0").status_code == 404


@pytest.mark.parametrize("query", ["", "?sentence=-1", "?sentence=2", "?sentence=word", "?sentence=0.5"])
def test_passage_api_validates_required_sentence_position(app_config, passage_fixture, query):
    with TestClient(create_app(**app_config)) as client:
        assert client.get("/api/papers/S/passage-matches"+query).status_code == 422


def test_absent_passage_index_leaves_other_features_available(client):
    assert client.get("/api/papers/P1/passages").json() == {"available": False, "sentences": []}
    assert client.get("/api/papers/P1/passage-matches?sentence=0").status_code == 409
    assert client.get("/api/papers/P1").status_code == 200
    assert client.get("/api/search?q=graph").status_code == 200


@pytest.mark.parametrize("damage", ["stale_corpus", "not_npz", "missing_field"])
def test_unusable_index_degrades_to_unavailable_without_breaking_startup(app_config, passage_fixture, damage):
    fixture = passage_fixture
    if damage == "stale_corpus":
        fixture["corpus"].write_text(fixture["corpus"].read_text()+"\n")
    elif damage == "not_npz":
        fixture["path"].write_bytes(b"not an index")
    else:
        np.savez_compressed(fixture["path"], version=1)
    with TestClient(create_app(**app_config)) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/papers/S/passages").json() == {"available": False, "sentences": []}
        assert client.get("/api/papers/S/passage-matches?sentence=0").status_code == 409


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
@pytest.mark.parametrize("field", ["documents", "sentences"])
def test_nonfinite_vectors_are_rejected_before_serving_scores(passage_fixture, bad_value, field):
    fixture = passage_fixture
    vectors = fixture["bundle"][field].copy()
    vectors[3, 0] = bad_value
    np.savez_compressed(fixture["path"], **dict(fixture["bundle"], **{field: vectors}))
    with pytest.raises(ValueError):
        PassageIndex.load(fixture["path"], fixture["corpus"])


@pytest.fixture
def policy_passage_fixture(passage_fixture):
    fixture = passage_fixture
    shortlist = np.full((8, 3), -1, dtype=np.int32)
    relevance = np.zeros((8, 3), dtype=np.float32)
    shortlist[0] = [5, 4, 3]  # B, C, A: policy order deliberately differs from blend.
    relevance[0] = [.93, .81, .44]
    shortlist[1] = [3, 5, -1]
    relevance[1] = [.88, .55, 0]
    bundle = dict(fixture["bundle"], version=2, shortlist=shortlist, relevance=relevance)
    np.savez_compressed(fixture["path"], **bundle)
    return dict(fixture, bundle=bundle)


def test_v2_uses_precomputed_policy_rank_and_score_but_selects_nearest_quote(policy_passage_fixture):
    fixture = policy_passage_fixture
    index = PassageIndex.load(fixture["path"], fixture["corpus"])
    result = index.search("S", 0)
    assert [item["id"] for item in result] == ["B", "C", "A"]
    assert [item["score"] for item in result] == pytest.approx([.93, .81, .44])
    assert result[2]["quote"] == fixture["choices"]["A"][0]
    second_sentence = index.search("S", 1)
    assert [item["id"] for item in second_sentence] == ["A", "B"]
    assert [item["score"] for item in second_sentence] == pytest.approx([.88, .55])
    assert second_sentence[0]["quote"] == fixture["choices"]["A"][1]
    assert index.search("S", 0, limit=1) == result[:1]
    with pytest.raises(ValueError, match="three"):
        index.search("S", 1, limit=50)
    assert index.search("S_COPY", 0) == []


@pytest.mark.parametrize("bad_row", [[5, 5, 3], [0, 4, 3], [6, 4, 3], [-2, 4, 3], [1, 4, 3], [2, 3, 5]])
def test_v2_rejects_duplicate_self_unknown_or_equivalent_title_candidates(policy_passage_fixture, bad_row):
    fixture = policy_passage_fixture
    shortlist = fixture["bundle"]["shortlist"].copy()
    shortlist[0] = bad_row
    np.savez_compressed(fixture["path"], **dict(fixture["bundle"], shortlist=shortlist))
    with pytest.raises(ValueError):
        PassageIndex.load(fixture["path"], fixture["corpus"])


@pytest.mark.parametrize("damage", ["float_ids", "short_rows", "two_candidates", "score_shape", "nan_score", "inf_score"])
def test_v2_rejects_inconsistent_policy_arrays(policy_passage_fixture, damage):
    fixture = policy_passage_fixture
    bundle = dict(fixture["bundle"])
    if damage == "float_ids":
        bundle["shortlist"] = bundle["shortlist"].astype(np.float32)
    elif damage == "short_rows":
        bundle["shortlist"] = bundle["shortlist"][:-1]
    elif damage == "two_candidates":
        bundle["shortlist"] = bundle["shortlist"][:, :2]
    elif damage == "score_shape":
        bundle["relevance"] = bundle["relevance"][:, :2]
    else:
        bundle["relevance"] = bundle["relevance"].copy()
        bundle["relevance"][0, 0] = np.nan if damage == "nan_score" else np.inf
    np.savez_compressed(fixture["path"], **bundle)
    with pytest.raises(ValueError):
        PassageIndex.load(fixture["path"], fixture["corpus"])


def test_passage_api_serves_policy_rankings_without_reordering_by_excerpt_similarity(app_config, policy_passage_fixture):
    fixture = policy_passage_fixture
    with TestClient(create_app(**app_config)) as client:
        assert client.get("/api/papers/S/passages").json() == {"available": True, "sentences": fixture["choices"]["S"]}
        response = client.get("/api/papers/S/passage-matches?sentence=0")
        assert response.status_code == 200
        items = response.json()["items"]
        assert [item["id"] for item in items] == ["B", "C", "A"]
        assert [item["score"] for item in items] == pytest.approx([.93, .81, .44])
        assert items[2]["quote"] == fixture["choices"]["A"][0]
        assert client.get("/api/papers/S_COPY/passage-matches?sentence=0").json() == {"items": []}


@pytest.fixture
def feedback_passage_fixture(policy_passage_fixture):
    fixture = policy_passage_fixture
    extra_ids = ["D", "E", "F", "G", "D_COPY"]
    extra_titles = ["Support D", "Support E", "Support F", "Support G", "SUPPORT: D!"]
    extra_choices = [
        ["Direct graph support is provided.", "A method spanning graph and document retrieval."],
        ["Additional evidence supports the retrieval method."],
        ["A further independent experiment is reported."],
        ["This alternative relates the two research methods."],
        ["Another edition of the supporting work is available."],
    ]
    extra = [dict(fixture["rows"][0], id=pid, title=title, abstract=" ".join(sentences))
             for pid, title, sentences in zip(extra_ids, extra_titles, extra_choices)]
    rows = fixture["rows"] + extra
    fixture["corpus"].write_text("".join(json.dumps(p, ensure_ascii=False)+"\n" for p in rows), encoding="utf8")
    texts = json.loads(fixture["bundle"]["texts_utf8"].tobytes()) + [text for group in extra_choices for text in group]
    shortlist = np.full((14, 3), -1, dtype=np.int32)
    shortlist[:8] = fixture["bundle"]["shortlist"]
    relevance = np.zeros((14, 3), dtype=np.float32)
    relevance[:8] = fixture["bundle"]["relevance"]
    followups = np.full((14, 4, 3), -1, dtype=np.int32)
    followup_scores = np.zeros((14, 4, 3), dtype=np.float32)
    # First page is B,C,A. Slots 1,2,3 follow those positions, not alphabetical IDs.
    followups[0] = [[6, 7, 8], [9, 6, 7], [8, 9, 6], [7, 8, 9]]
    followup_scores[0] = [[.80, .70, .60], [.91, .82, .73], [.92, .83, .74], [.93, .84, .75]]
    followups[1, 0] = [6, -1, -1]
    followup_scores[1, 0] = [.5, 0, 0]
    followups[1, 1] = [7, 8, -1]
    followup_scores[1, 1] = [.6, .4, 0]
    followups[1, 2] = [9, -1, -1]
    followup_scores[1, 2] = [.7, 0, 0]
    bundle = dict(fixture["bundle"],
                  corpus_sha=hashlib.sha256(fixture["corpus"].read_bytes()).hexdigest(),
                  ids=np.concatenate([fixture["bundle"]["ids"], extra_ids]),
                  titles=np.concatenate([fixture["bundle"]["titles"], extra_titles]),
                  texts_utf8=np.frombuffer(json.dumps(texts, ensure_ascii=False).encode("utf8"), dtype=np.uint8),
                  offsets=np.array([0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14]),
                  documents=np.vstack([fixture["bundle"]["documents"], [[.8, .6], [0, 1], [-1, 0], [.6, .8], [.8, .6]]]),
                  sentences=np.vstack([fixture["bundle"]["sentences"], [[1, 0], [.6, .8], [0, 1], [-1, 0], [.6, .8], [1, 0]]]),
                  shortlist=shortlist, relevance=relevance, followups=followups, followup_scores=followup_scores)
    np.savez_compressed(fixture["path"], **bundle)
    return dict(fixture, bundle=bundle, rows=rows, choices={**fixture["choices"], **dict(zip(extra_ids, extra_choices))})


@pytest.mark.parametrize("after,expected,scores", [
    ("next", ["D", "E", "F"], [.8, .7, .6]),
    ("B", ["G", "D", "E"], [.91, .82, .73]),
    ("C", ["F", "G", "D"], [.92, .83, .74]),
    ("A", ["E", "F", "G"], [.93, .84, .75]),
])
def test_feedback_index_selects_continue_or_clicked_first_page_slot(feedback_passage_fixture, after, expected, scores):
    fixture = feedback_passage_fixture
    index = PassageIndex.load(fixture["path"], fixture["corpus"])
    first = index.search("S", 0)
    result = index.search("S", 0, after=after)
    assert [p["id"] for p in first] == ["B", "C", "A"]
    assert [p["id"] for p in result] == expected
    assert [p["score"] for p in result] == pytest.approx(scores)
    assert {p["id"] for p in result}.isdisjoint({"S", "S_COPY", "A_COPY", *[p["id"] for p in first]})
    rows = {p["id"]: p for p in fixture["rows"]}
    assert all(p["quote"] in rows[p["id"]]["abstract"] for p in result)
    assert index.search("S", 0, limit=1, after=after) == result[:1]
    assert index.search("S", 0) == first


def test_feedback_quotes_use_click_context_and_padding_never_wraps_to_last_paper(feedback_passage_fixture):
    fixture = feedback_passage_fixture
    index = PassageIndex.load(fixture["path"], fixture["corpus"])
    continuation = next(p for p in index.search("S", 0, after="next") if p["id"] == "D")
    clicked = next(p for p in index.search("S", 0, after="B") if p["id"] == "D")
    assert continuation["quote"] == fixture["choices"]["D"][0]
    assert clicked["quote"] == fixture["choices"]["D"][1]
    assert [p["id"] for p in index.search("S", 1, after="next")] == ["D"]
    assert [p["id"] for p in index.search("S", 1, after="A")] == ["E", "F"]
    assert [p["id"] for p in index.search("S", 1, after="B")] == ["G"]
    assert index.search("S_COPY", 0, after="next") == []


@pytest.mark.parametrize("bad_row", [
    [0, 7, 8], [1, 7, 8], [2, 7, 8], [3, 7, 8], [4, 7, 8], [5, 7, 8],
    [6, 6, 8], [6, 10, 8], [11, 7, 8], [-2, 7, 8], [6, -1, 8],
])
def test_followup_validation_excludes_seed_all_first_page_titles_and_duplicates(feedback_passage_fixture, bad_row):
    fixture = feedback_passage_fixture
    followups = fixture["bundle"]["followups"].copy()
    followups[0, 1] = bad_row
    np.savez_compressed(fixture["path"], **dict(fixture["bundle"], followups=followups))
    with pytest.raises(ValueError):
        PassageIndex.load(fixture["path"], fixture["corpus"])


@pytest.mark.parametrize("damage", ["missing_scores", "missing_picks", "float_picks", "missing_slot", "wrong_scores_shape", "nan_score", "invalid_click_slot"])
def test_followup_validation_rejects_incomplete_arrays_or_impossible_clicks(feedback_passage_fixture, damage):
    fixture = feedback_passage_fixture
    bundle = dict(fixture["bundle"])
    if damage == "missing_scores":
        del bundle["followup_scores"]
    elif damage == "missing_picks":
        del bundle["followups"]
    elif damage == "float_picks":
        bundle["followups"] = bundle["followups"].astype(np.float32)
    elif damage == "missing_slot":
        bundle["followups"] = bundle["followups"][:, :3]
    elif damage == "wrong_scores_shape":
        bundle["followup_scores"] = bundle["followup_scores"][:-1]
    elif damage == "nan_score":
        bundle["followup_scores"] = bundle["followup_scores"].copy()
        bundle["followup_scores"][0, 0, 0] = np.nan
    else:
        bundle["followups"] = bundle["followups"].copy()
        # Sentence 1 has only A,B on page one: its third click cannot exist.
        bundle["followups"][1, 3] = [6, -1, -1]
    np.savez_compressed(fixture["path"], **bundle)
    with pytest.raises(ValueError):
        PassageIndex.load(fixture["path"], fixture["corpus"])


def test_feedback_api_advertises_refinement_and_serves_only_one_followup_page(app_config, feedback_passage_fixture):
    with TestClient(create_app(**app_config)) as client:
        first = client.get("/api/papers/S/passage-matches?sentence=0")
        assert first.status_code == 200 and first.json()["can_refine"] is True
        for after, expected in [("next", ["D", "E", "F"]), ("B", ["G", "D", "E"]),
                                ("C", ["F", "G", "D"]), ("A", ["E", "F", "G"])]:
            response = client.get("/api/papers/S/passage-matches", params={"sentence": 0, "after": after})
            assert response.status_code == 200
            assert [p["id"] for p in response.json()["items"]] == expected
            assert not response.json().get("can_refine", False)
        for invalid in ["S", "A_COPY", "D", "unknown"]:
            assert client.get("/api/papers/S/passage-matches", params={"sentence": 0, "after": invalid}).status_code == 422
        assert client.get("/api/papers/S/passage-matches?sentence=1&after=C").status_code == 422
        assert client.get("/api/papers/S/passage-matches?sentence=999&after=next").status_code == 422
        assert client.get("/api/papers/UNINDEXED/passage-matches?sentence=0&after=next").status_code == 409
        assert client.get("/api/papers/unknown/passage-matches?sentence=0&after=next").status_code == 404


def test_v2_without_optional_feedback_keeps_first_page_but_rejects_after(app_config, policy_passage_fixture):
    with TestClient(create_app(**app_config)) as client:
        first = client.get("/api/papers/S/passage-matches?sentence=0")
        assert first.status_code == 200 and not first.json().get("can_refine", False)
        assert [p["id"] for p in first.json()["items"]] == ["B", "C", "A"]
        assert client.get("/api/papers/S/passage-matches?sentence=0&after=next").status_code == 409
        assert client.get("/api/papers/S/passage-matches?sentence=0&after=B").status_code == 409
