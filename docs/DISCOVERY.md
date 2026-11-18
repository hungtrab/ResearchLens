# Conference library and online discovery

Snapshot ngày 14/10/2026: **129.596 records**, tám hội nghị, giai đoạn **2020–2026**. Đây là metadata thu thập được, không phải chứng nhận đã có mọi accepted paper. Không crawl hàng loạt PDF.

| Hội nghị | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ICML | 1144 | 1256 | 1233 | 1828 | 2610 | 3330 | 6552 |
| NeurIPS | 1909 | 2367 | 2928 | 3584 | 4538 | 5907 | 6567 |
| ICLR | 720 | 918 | 1095 | 1584 | 2296 | 3830 | 5468 |
| AAAI | 1864 | 2330 | 1619 | 2021 | 2866 | 3486 | 4913 |
| EMNLP | 1256 | 1322 | 1491 | 2249 | 2453 | 3498 | 0 |
| NAACL | 0 | 563 | 747 | 0 | 958 | 1378 | 0 |
| ACL | 873 | 1252 | 1110 | 2156 | 2013 | 3364 | 4823 |
| CVPR | 1475 | 1678 | 2095 | 2381 | 2748 | 2908 | 4042 |

Các ô 0 là adapter chưa trả record trong snapshot, không khẳng định hội nghị không diễn ra. Các track như Findings, short papers, datasets/benchmarks có thể được nguồn công bố cùng hội nghị. Counts có thể khác danh sách main-track accepted papers.

## Nguồn và pipeline

- ICML: [PMLR](https://proceedings.mlr.press/), volumes119/139/162/202/235/267/306.
- NeurIPS và ICLR: bulk JSON liên kết từ chương trình chính thức ở [NeurIPS](https://neurips.cc/) và [ICLR](https://iclr.cc/); loại workshop và session cha, hợp nhất event trùng paper.
- ACL/EMNLP/NAACL: [ACL Anthology XML](https://github.com/acl-org/acl-anthology/tree/master/data/xml), commit đầy đủ trong manifest; Findings chỉ lấy volume tương ứng hội nghị.
- CVPR: [CVF Open Access](https://openaccess.thecvf.com/); riêng2020 đọc ba trang theo ngày.
- OpenAlex: source IDs trong `researchlens/sources.py`, cursor phân trang đến hết kết quả. AAAI hiện chỉ dùng nguồn này; các hội nghị khác hợp với nguồn chính thức.

Cache gzip theo URL, timeout, ba lần thử và ba worker. Chạy lại dùng cache; không vượt qua challenge đăng nhập/anti-bot. Nguồn lỗi hoặc chưa có dữ liệu được ghi trạng thái riêng. Khử trùng exact normalized title trong conference/year; giữ OpenAlex ID và alias chính thức. 94.691 record có abstract (73,1%). Metadata còn thiếu không được sinh bù.

```bash
uv run python scripts/crawl_conferences.py --from-year 2020 --to-year 2026 --workers 3
# Restart ứng dụng để sync JSONL mới vào FTS; Docker cần build lại image.
```

`data/conferences/papers.jsonl` và `manifest.json` được bàn giao. Raw HTTP cache không nằm trong source ZIP; có thể tải lại bằng collector. Crawler dùng cache vô thời hạn theo URL để tái lập lần thu thập; muốn cập nhật từ nguồn mới, di chuyển thư mục `data/conferences/raw` sang một tên lưu trữ khác trước khi chạy lại. Việc thu thập lại có thể cho counts khác vì nguồn thay đổi. Mở rộng ngoài2020–2026 cần cập nhật/kiểm tra adapter và validation năm; không tự giả định volume của ICML.

## Ba phạm vi tìm kiếm

1. **Mining workspace:** snapshot3.595 paper và những paper đã import. Sparse BM25/TF-IDF/PageRank, review mining, recommendation và graph. Solr tùy chọn chỉ index snapshot gốc.
2. **Conference library:** toàn bộ129.596 record trên SQLite FTS5. BM25 field weights8/2/2/1 cho title/abstract/authors/conference; tối đa32 token, OR, lọc hội nghị/năm và phân trang. Chạy offline sau khi có snapshot.
3. **Search online:** [OpenAlex search](https://help.openalex.org/api/searching/), query/năm/pagination, cache5phút tối đa128entries. Đây là tìm kiếm metadata học thuật; không phải crawler toàn bộ web hoặc tìm kiếm nội dung PDF. Live results chưa tự vào graph.

`Import & read` lấy metadata theo ID từ provider, khử trùng DOI/tiêu đề, lưu SQLite rồi mở reader. Record chính thức được thử enrich bằng exact match OpenAlex; nếu thất bại thì giữ metadata nguồn và hiển thị thông báo. Alias được áp dụng vào references ở cả runtime và restart. Chỉ có cạnh khi nguồn cung cấp reference và target tồn tại trong workspace.

Import dùng chung trong instance; Project, notes và saved library riêng theo cookie profile. Engine rebuild/swap dưới lock trong **một process**, chưa hỗ trợ nhiều Uvicorn worker đồng thời. Chi phí import tăng theo số paper trong workspace; chỉ import paper cần nghiên cứu, không import hàng loạt toàn bộ catalog vào NetworkX.

API key tùy chọn đặt `OPENALEX_API_KEY` ở môi trường server hoặc `.env` cho Compose. Key đi trong Authorization header, không đưa ra browser hay ghi vào URL manifest. Hạn mức và khả năng anonymous access do [OpenAlex](https://help.openalex.org/api/authentication/) quyết định; UI báo lỗi rõ khi provider không sẵn sàng, local search tiếp tục dùng được.

## Đo lường và kiểm chứng

`uv run python scripts/evaluate_catalog.py` dựng index mới, warm20query rồi đo5vòng (100lần), xen kẽ có/không lọc conference. Kết quả đã đo: **p50 44,2ms; p95 153,2ms**, dựng index7,7s, database525,9MiB. Phạm vi gồm count/rank/top20 JSON decode, không gồm HTTP/browser/mạng. Môi trường và từng quan sát ở `artifacts/catalog_results.json`. Chưa có relevance labels cho catalog nên chưa báo cáo nDCG trên kho này.

Snapshot benchmark gốc và `artifacts/results.json` giữ nguyên hash. Không dùng số đo SciFact/reference recovery để tuyên bố chất lượng của kho mở rộng. UI Observatory phân biệt current workspace với fixed benchmark.

Unit/API tests không dùng mạng thật; browser acceptance ở `scripts/discovery_qa.py` dùng OpenAlex thật và DB tạm, kiểm tra search → import → reader → notes → Project → graph, pagination/coverage/mobile. Kết quả ở `artifacts/discovery_qa.json`. SHA/IDs/56 coveragecells được audit độc lập trong `docs/AUDIT.md`.
