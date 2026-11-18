"""Small deterministic metadata fixtures; no network or production database access."""
import json

import pytest
from fastapi.testclient import TestClient

from researchlens.api import create_app
from researchlens.mining import MiningEngine


@pytest.fixture(autouse=True)
def isolate_optional_services(monkeypatch):
    """Tests opt into a mocked Solr service explicitly, regardless of shell config."""
    monkeypatch.delenv("SOLR_URL", raising=False)


@pytest.fixture
def papers():
    rows = [
        ("P1", "Graph neural networks", "Graph neural networks learn node representations with GCN on Cora.", 2018, "ICLR", ["P3", "OUTSIDE", "P1"], True, "graphs"),
        ("P2", "Graph neural network benchmarks", "Graph neural networks learn node representations with GCN on Cora. Benchmark evaluation.", 2019, "NeurIPS", ["P3", "P1"], True, "graphs"),
        ("P3", "Neural information retrieval", "BM25 and BERT support document search on MS MARCO.", 2017, "SIGIR", [], False, "retrieval"),
        ("P4", "Graph based information retrieval", "Citation graph search improves document retrieval ranking.", 2020, "SIGIR", ["P3"], True, "retrieval"),
        ("P5", "Opinion mining in reviews", "Sentiment analysis extracts reviewer opinions and evaluation evidence.", 2021, "ACL", [], False, "sentiment"),
        ("P6", "Language modeling & reproducibility", "A paragraph describes experimental methods and code availability.", 2022, "ACL", ["P3"], True, "language"),
    ]
    return [dict(id=pid, title=title, abstract=abstract, year=year, venue=venue,
                 references=refs, open_access=oa, collection_topic=topic, topics=[topic],
                 authors=["An Nguyen", "Binh Tran"], citations=100, source="fixture",
                 url=f"https://example.org/{pid}", doi=f"https://doi.org/10.1234/{pid}")
            for pid, title, abstract, year, venue, refs, oa, topic in rows]


@pytest.fixture
def reviews():
    return [
        dict(id="r1", paper_id="P1", text="The evaluation is strong and the experiments are convincing.", rating=8, split="train"),
        dict(id="r2", paper_id="P2", text="The evaluation is strong and the experiments are thorough.", rating=7, split="train"),
        dict(id="r3", paper_id="P3", text="The evaluation is weak and the experiments are insufficient.", rating=3, split="train"),
        dict(id="r4", paper_id="P4", text="The evaluation is weak and the experiments are missing.", rating=4, split="train"),
        dict(id="r5", paper_id="P5", text="The evaluation describes experiments with balanced observations.", rating=5, split="test"),
    ]


@pytest.fixture
def engine(papers):
    return MiningEngine(papers)


@pytest.fixture
def app_config(tmp_path, papers, reviews):
    data = tmp_path / "corpus"
    data.mkdir()
    for name, rows in [("papers", papers), ("reviews", reviews)]:
        (data / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return dict(data_dir=data, db_path=tmp_path / "state" / "test.sqlite")


@pytest.fixture
def client(app_config):
    with TestClient(create_app(**app_config)) as client:
        yield client
