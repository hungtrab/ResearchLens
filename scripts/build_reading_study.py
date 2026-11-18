"""Freeze a blinded depth-three recommendation pool for four reading tasks."""
import hashlib
import json
import random
from pathlib import Path

from researchlens.catalog import title_key
from researchlens.mining import MiningEngine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/reading-study"
BRIEFS = [
    ("MF", "W2054141820", "Tôi muốn xây hệ gợi ý từ ma trận tương tác user–item thưa. Tôi cần cách học nhân tố ẩn hoặc một baseline có thể triển khai để so sánh dự đoán/xếp hạng."),
    ("KG", "W2283196293", "Tôi muốn dự đoán liên kết còn thiếu trong knowledge graph. Tôi cần mô hình embedding chấm điểm bộ ba (head, relation, tail), nhất là cách xử lý các loại quan hệ."),
    ("AB", "W2518806256", "Tôi muốn xác định thái độ của người viết review đối với từng khía cạnh sản phẩm. Tôi cần cách gắn từ hoặc câu đánh giá với khía cạnh và suy ra polarity cho khía cạnh đó."),
    ("LR", "W2129245267", "Tôi muốn đánh giá mô hình learning-to-rank cho tìm kiếm tài liệu theo query. Tôi cần dữ liệu có nhãn relevance hoặc phương pháp huấn luyện và đánh giá thứ tự kết quả."),
]
METHODS = ("popular", "content", "graph", "hybrid", "cocitation")
DEPTH = 3


def main():
    source = ROOT / "data/papers.jsonl"
    with source.open() as f:
        papers = [json.loads(line) for line in f]
    engine = MiningEngine(papers)
    cards, rankings = [], {}
    for topic, seed, brief in BRIEFS:
        pool = {}
        rankings[topic] = {}
        for method in METHODS:
            response = engine.recommend([seed], limit=50, method=method, diversity=0)
            ranking = []
            seen = {title_key(engine.by_id[seed]["title"])}
            for item in response["items"]:
                key = title_key(item["title"])
                if key in seen:
                    continue
                seen.add(key)
                code = topic + "-" + hashlib.sha256(key.encode()).hexdigest()[:8]
                pool.setdefault(code, dict(code=code, topic=topic, brief=brief, seed=engine.by_id[seed]["title"],
                                           title=item["title"], abstract=item["abstract"], id=item["id"], url=item["url"]))
                ranking.append(code)
                if len(ranking) == DEPTH:
                    break
            rankings[topic][method] = ranking
        cards.extend(pool.values())
    cards.sort(key=lambda c: c["code"])
    random.Random(20261118).shuffle(cards)
    fingerprint = hashlib.sha256(json.dumps(cards, sort_keys=True).encode()).hexdigest()
    packet = dict(study=fingerprint, cards=cards)
    for name, value in [("blind_packet", packet), ("frozen_runs", dict(study=fingerprint, corpus_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), depth=DEPTH, rankings=rankings))]:
        (OUT / (name + ".json")).write_text(json.dumps(value, ensure_ascii=False, indent=2))
    template = (ROOT / "scripts/reading_form.html").read_text()
    summaries = json.loads((ROOT / "scripts/reading_summaries_vi.json").read_text())
    assert all(c["code"].split("-")[1] in summaries for c in cards)
    template = template.replace("/*SUMMARIES*/null", json.dumps(summaries,ensure_ascii=False).replace("<","\\u003c"))
    (OUT / "annotate.html").write_text(template.replace("/*PACKET*/null", json.dumps(packet,ensure_ascii=False).replace("<","\\u003c")))
    print(f"{len(cards)} distinct task/paper pairs")

if __name__ == "__main__":
    main()
