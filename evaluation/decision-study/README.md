# Tìm bài có phương pháp phù hợp với đề tài

Người đọc có một bài mẫu và muốn tìm bài khác áp dụng kỹ thuật tương tự. Công cụ nhận mục tiêu, tên bài mẫu và ID bài đang cân nhắc. Nó trả lời bằng tiếng Việt: nên xem phần phương pháp trước, có thể để đọc sau, hoặc cần người đọc xem bản gốc vì thiếu abstract. Kết quả là gợi ý thử nghiệm chạy local. Trang tìm kiếm chính và phiếu chấm mù độc lập với công cụ này.

## Kết quả trên nhãn chuyên gia

CSFCube có nhãn chuyên gia về độ tương đồng theo từng khía cạnh của bài khoa học. Chúng tôi chọn khía cạnh **phương pháp** và dùng fold 1 do tác giả phát hành. Tám bài mẫu dùng để phát triển (956 cặp, còn 914 sau loại trùng ID với test). Chín bài mẫu khác dùng để kiểm tra (1.218 cặp). Chuyên gia chấm 63 cặp test đạt mức 2 hoặc 3, chiếm 5,2%. [CSFCube, Mysore, O’Gorman, McCallum và Zamani, NeurIPS Datasets and Benchmarks 2021](https://openreview.net/forum?id=8Y50dBbmGU)

| Phương án | nDCG@10 ↑ | AP gộp ↑ | Balanced accuracy ↑ | Pilot nội bộ khớp nhãn trợ lý |
|---|---:|---:|---:|---:|
| Hỏi Laya trực tiếp | 0,233 | 0,138 | 0,500 | 19/41 |
| Chỉ TF-IDF | 0,324 | 0,094 | 0,661 | 32/41 |
| Chỉ MiniLM | 0,360 | 0,129 | 0,724 | 37/41 |
| TF-IDF + MiniLM | **0,365** | **0,141** | **0,725** | **38/41** |
| TF-IDF + MiniLM + hai điểm Laya | 0,339 | 0,125 | 0,700 | 34/41 |
| SPECTER, thứ hạng do tác giả công bố | 0,262 | — | — | — |

nDCG@10 đo mức độ các bài liên quan cao xuất hiện sớm trong mười vị trí đầu. Mỗi bài mẫu đóng góp ngang nhau, gain bằng 2^grade−1. AP gộp các cặp test, lấy grade ≥ 2 làm dương. Balanced accuracy là trung bình của tỷ lệ phát hiện đúng ở hai lớp. Các hàng được so trên cùng pool ứng viên cố định. Điểm SPECTER tính lại từ khoảng cách tác giả cung cấp bằng đúng metric ở đây. [Thứ hạng tham chiếu](https://github.com/iesl/CSFCube/blob/7ffe012c1d92c470f185fc086086caf5b8e05ef3/eval_scripts/sample_ranked_pools/test-pid2pool-csfcube-specter-method-ranked.json)

Bản kết hợp được ấn định trước khi xem test. So với Laya trực tiếp, mức tăng trung bình nDCG@10 là +0,106, khoảng bootstrap ghép cặp 95% [−0,031, +0,233]. So với TF-IDF + MiniLM, mức thay đổi là −0,026 [−0,056, +0,007], tăng ở 2/9 bài mẫu. Cả hai khoảng đều đi qua 0. Hiệu quả trên đề tài mới còn bất định. Bản văn bản được chọn làm mặc định CLI sau khi xem kết quả và có thể bỏ tải checkpoint Laya. Bản thử nghiệm có Laya vẫn chạy được bằng cùng lệnh.

Ở ngưỡng phát triển của bản kết hợp, 43/63 cặp phù hợp được tìm thấy (recall 68,3%). Trong 370 gợi ý chỉ có 43 cặp phù hợp (precision 11,6%). Bản văn bản tìm thấy 49/63 trong 427 gợi ý (precision 11,5%). Luôn chọn “để sau” cũng đạt 1.155/1.218 quyết định đúng vì lớp âm chiếm đa số, nhưng bỏ sót mọi cặp phù hợp. Ngưỡng ở đây phục vụ việc sàng lọc ứng viên cho người xem tiếp.

## Mô hình học điều gì?

- Đầu vào phía truy vấn gồm tiêu đề bài mẫu và các câu được CSFCube gắn nhãn dự đoán là phương pháp. Nếu thiếu câu đó, dùng toàn bộ abstract. Phía ứng viên gồm tiêu đề và abstract.
- TF-IDF học IDF từ corpus ResearchLens. MiniLM tính cosine giữa hai vector câu. Laya trả về hai tín hiệu riêng về ý định lưu và mức khớp kỹ thuật. Mọi tín hiệu tính từ văn bản trước khi dùng nhãn. Trọng số của các checkpoint được giữ cố định.
- Hồi quy logistic học quyết định grade ≥ 2 trên các đặc trưng đã chuẩn hóa. Ngưỡng được chọn theo balanced accuracy từ dự đoán luân phiên giữ lại từng bài mẫu phát triển. Mỗi lượt loại bài trùng ID với phần đang giữ lại. Điểm đầu ra biểu thị mức phù hợp với tiêu chí chuyên gia CSFCube.
- Bảng pilot dùng nguyên trọng số và ngưỡng CSFCube cho bốn brief tiếng Anh, 41 cặp có nhãn do trợ lý chấm. Hai cặp khác có nhãn “chưa rõ”. Hai tựa bài của pilot cũng xuất hiện làm ứng viên phát triển trong CSFCube. Hành vi lưu thực tế của người dùng vẫn cần được đo qua [phiếu độc lập](../reading-study/annotate.html).
- Các số 34/41 và 38/41 chấm điểm mô hình trước quy tắc chuyển cho người đọc. CLI thực tế chuyển cả 12/43 cặp thiếu abstract cho người đọc. Trong 31 cặp có abstract và có nhãn pilot, bản kết hợp khớp 24, bản văn bản khớp 28.
- File so sánh còn lưu phép kiểm tra thăm dò nếu học lại **chỉ ngưỡng** từ ba brief nội bộ khác, loại paper trùng với brief đang chấm. Bản kết hợp khớp 32/41, bản văn bản khớp 33/41. Các ngưỡng nội bộ này không dùng trong CLI.

## Chạy với một bài trong kho local

Cần chạy lệnh trong thư mục gốc ResearchLens. Để tính lại bảng từ đặc trưng đi kèm và vẽ hình, chỉ cần các thư viện của project. Chạy các lệnh sau không khởi động server.

```bash
uv run python scripts/fit_reading_decision.py
uv run python scripts/plot_reading_decision.py
```

Để chấm bài mới, thiết lập môi trường ML riêng. Bản mặc định dùng mô hình tiếng Anh, ví dụ mục tiêu là tìm phương pháp dự đoán liên kết bị thiếu trong đồ thị tri thức. Lần đầu chương trình tải checkpoint từ Hugging Face. Sau khi có cache có thể đặt HF_HUB_OFFLINE=1. Chế độ text chạy được với torch, transformers và scikit-learn; laya chỉ được import ở chế độ cần nó.

```bash
uv venv --python 3.13 .venv-decision
uv pip install --python .venv-decision/bin/python 'torch==2.10.0' 'transformers==4.57.6' 'tokenizers==0.22.2' 'laya==0.3.0' 'scikit-learn==1.7.1'
.venv-decision/bin/python scripts/decide_reading.py --mode text \
  --goal "Predict missing knowledge graph links using embeddings that score triples and handle different relation types." \
  --seed-title "Knowledge Graph Embedding by Translating on Hyperplanes" \
  --paper-id W2184957013
```

Đổi thành `--mode laya` để dùng thêm hai tín hiệu của decision model. Dùng `--paper-id W4392553889` để kiểm tra trường hợp bài chỉ có tiêu đề. Để tái tạo toàn bộ đặc trưng từ nguồn công khai, chạy lệnh dưới đây trong môi trường ML, rồi chạy lại lệnh huấn luyện ở trên.

```bash
.venv-decision/bin/python scripts/decision_features.py --cache /tmp/researchlens-csfcube
```

## Dữ liệu và nguồn

[features.jsonl](features.jsonl) chứa khóa cặp, nhãn và bốn tín hiệu. [feature_provenance.json](feature_provenance.json) ghi revision mô hình, mã băm corpus và đặc trưng. [decision_comparison.json](decision_comparison.json) lưu dự đoán theo đúng thứ tự cặp, điểm từng truy vấn, confusion matrix và khoảng bootstrap (10.000 lượt lấy mẫu lại theo truy vấn). [reading_policy.json](reading_policy.json) và [text_policy.json](text_policy.json) chứa trọng số, cách chuẩn hóa và ngưỡng để dùng lại.

Nhãn, ID CSFCube và thứ hạng SPECTER được phân phối lại theo [CC BY-NC 4.0](CSFCube_LICENSE.md). Nhóm ResearchLens trích riêng phần phương pháp, tính thêm bốn đặc trưng và đánh giá lại trên fold 1. Bản trích không chứa toàn văn paper. [Nguồn dữ liệu cố định](https://github.com/iesl/CSFCube/tree/7ffe012c1d92c470f185fc086086caf5b8e05ef3)

Checkpoint nguồn mở là Laya Typed-Decisions và all-MiniLM-L6-v2. Jev chính hãng chưa được đo trong thí nghiệm này. [Laya](https://huggingface.co/convaiinnovations/laya-typed-decisions) · [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) · [Giao diện TypeSafe Choice](https://docs.typesafe.ai/primitives/choice)

## Tìm từ một câu trong trình đọc

Trong **Paper reading**, mở *Find papers from a sentence in this abstract*, chọn một câu và bấm *Find similar papers*. Ba tựa bài có thể mở ngay. Mỗi tựa có một câu được lấy nguyên văn từ abstract để xem hai bài giống nhau ở đâu. [Ảnh giao diện](../../artifacts/screenshots/passage_reader.png)

Mô hình logistic TF-IDF + MiniLM đã đóng băng xếp hạng theo câu được chọn kèm tiêu đề bài mẫu. Sau bước xếp hạng, MiniLM so riêng câu được chọn với các câu của từng bài đích và trả câu gần nhất. Các số đo dưới đây đánh giá thứ hạng cấp bài. Câu trích là đầu mối cho người đọc kiểm tra mức liên quan. Các câu cùng nói về một kỹ thuật vẫn có thể đưa ra kết luận khác nhau.

| Tác vụ CSFCube | Cặp đã chấm sau khử trùng | Bài hữu ích trong top 3, mô hình văn bản | Bài hữu ích trong top 3, phối hợp câu và abstract | Có ít nhất một bài hữu ích với mô hình văn bản |
|---|---:|---:|---:|---:|
| Tìm cùng phương pháp, 9 truy vấn | 1.214 | 8/27 | 6/27 | 6/9 truy vấn |
| Tìm kết quả tương tự, 6 truy vấn | 920 | 8/18 | 6/18 | 5/6 truy vấn |

Câu truy vấn là câu đầu tiên được bộ phân loại của CSFCube gắn nhãn đúng khía cạnh. Nếu câu gốc tách thành nhiều phần theo bộ tách câu của trình đọc thì lấy phần đầu. Dương tính là grade ≥ 2. Phương án phối hợp cho trọng số 1/2 cho cosine với toàn tài liệu và 1/2 cho cosine với câu gần nhất. File kết quả còn lưu đối chứng chỉ dùng toàn tài liệu, chỉ dùng câu gần nhất và mô hình văn bản cũ nhận toàn bộ mục tiêu trên tập phương pháp. Với toàn bộ mục tiêu cũ, mô hình văn bản tìm được 4/27 ở top 3. Cách chọn đầu vào vì vậy cũng ảnh hưởng so sánh.

Chín truy vấn phương pháp đã được xem trong thí nghiệm trước. Sáu truy vấn kết quả thuộc fold1_test, sau khi loại các bài mẫu đã xuất hiện trong 17 truy vấn phương pháp. Có 17 bài ứng viên của phần kết quả từng xuất hiện trong tập phát triển phương pháp. Hai tác vụ có tiêu chí hữu ích khác nhau. Thí nghiệm dùng các pool CSFCube đã chấm và giữ một ID cố định cho mỗi tiêu đề chuẩn hóa để các phương án xếp cùng ứng viên. Khi chạy trong kho local, hệ thống loại các phiên bản trùng tiêu đề theo thứ tự điểm. Chênh lệch precision@3 (phối hợp trừ mô hình văn bản) lần lượt là −0,074 và −0,111. Khoảng bootstrap 95% theo bài mẫu lần lượt là [−0,259, +0,074] và [−0,278, +0,111]. Lựa chọn giao diện dùng mô hình văn bản để xếp hạng được đưa ra sau khi xem so sánh này. [Pool, nhãn, điểm và top 3 từng truy vấn](passage_comparison.json) · [Nguồn và giấy phép CSFCube](https://github.com/iesl/CSFCube/tree/7ffe012c1d92c470f185fc086086caf5b8e05ef3)

Chỉ mục v2 trong data/passages.npz chứa 2.725 bài có abstract trong snapshot 3.595 bài. Các bài thiếu abstract, bài mới import và danh mục hội nghị ngoài snapshot chưa có trong chỉ mục này. Tổng cộng đã kiểm tra 55.686 lượt trả câu trích trên 18.562 câu đầu vào. Khi đổi corpus, digest giúp ứng dụng phát hiện bản chỉ mục cũ. Việc tạo lại chỉ mục dùng môi trường ML đã mô tả ở trên. Khi chạy web, điểm xếp hạng được đọc từ shortlist đã tính sẵn, hai câu so sánh nằm trong corpus local.

```bash
PYTHONPATH=. .venv-decision/bin/python scripts/build_passage_index.py
PYTHONPATH=. .venv-decision/bin/python scripts/evaluate_passages.py --cache /tmp/researchlens-csfcube
uv run python scripts/verify_passage_index.py
uv run --extra dev python scripts/passage_qa.py
```

Lệnh kiểm tra giao diện dùng Chromium của Playwright và FastAPI TestClient, không mở cổng HTTP. Dữ liệu mô hình MiniLM dùng theo [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

## ASPIRE có đưa đúng bài vào top 3?

Giữ nguyên câu truy vấn, ba vị trí đầu và nhãn hữu ích (grade ≥ 2). Thử thay tín hiệu văn bản bằng ASPIRE single-match cho khoa học máy tính, hoặc kết hợp hai danh sách bằng reciprocal-rank fusion. ASPIRE học từ các bài được đồng trích dẫn, mã hóa câu trong ngữ cảnh tiêu đề và abstract. Ở đây dùng khoảng cách L2 nhỏ nhất đến một câu của bài đích. Cài đặt chia abstract dài thành cửa sổ tối đa 512 token, lặp lại tiêu đề và cắt câu quá dài để mỗi câu trong Reader vẫn có vector. Nhóm tác giả đã đánh giá trên CSFCube và phát hành checkpoint tốt nhất trong ba lần chạy. [Model card của tác giả](https://huggingface.co/allenai/aspire-contextualsentence-singlem-compsci)

| Tập đánh giá | Reader, top 3 | ASPIRE, top 3 | nDCG@10 Reader → ASPIRE |
|---|---:|---:|---:|
| Phương pháp, 9 tác vụ đã xem | 8/27 | 7/27 | 0,392 → 0,455 |
| Kết quả, 6 tác vụ đã xem | 8/18 | 10/18 | 0,536 → 0,519 |
| Chuyển giao, 14 tác vụ | 19/42 | 24/42 | 0,477 → 0,567 |

Trên tám truy vấn phát triển, năm trọng số hòa nhau ở 8/24. ASPIRE thuần thắng theo nDCG@10. Tập chuyển giao gồm toàn bộ tác vụ bối cảnh hoặc kết quả với ID bài mẫu chưa xuất hiện trong 17 truy vấn phương pháp và 6 truy vấn kết quả đã dùng. Có 14 tác vụ, 11 bài mẫu, 1.662 cặp; trong đó bối cảnh đạt 17/27 so với 13/27 và kết quả đạt 7/15 so với 6/15. Có 24 ứng viên từng xuất hiện trong tập học policy văn bản. Một truy vấn bối cảnh thiếu câu được gắn nhãn tự động nên dùng câu đầu. Chênh lệch precision@3 là +11,9 điểm phần trăm, CI bootstrap 95% theo bài mẫu xấp xỉ [0,0; +25,0]. CI của chênh lệch nDCG@10 là [−0,008; +0,206]. ASPIRE thiếu một bài hữu ích so với Reader ở top 3 phương pháp nên cấu hình đang phục vụ giữ nguyên. [Quy tắc chọn](context_protocol.md) · [Điểm, từng ứng viên và kết quả từng truy vấn](context_comparison.json)

## Phản hồi sau khi đã xem ba bài

**Find more like this paper** lấy ba tựa tiếp theo dựa trên cả câu ban đầu và một bài người đọc chọn. **Show next three** tiếp tục thứ hạng ban đầu. Hệ thống loại cả ba tựa đã xem, các phiên bản trùng tên và bài mẫu. Cả hai luồng đọc kết quả tính sẵn từ chỉ mục local, không gửi thao tác ra dịch vụ ngoài.

| Cách lấy trang hai | Vị trí hữu ích trên toàn bộ 37 tác vụ | Trong 28 tác vụ có thể chọn bài |
|---|---:|---:|
| Xem tiếp thứ hạng cũ | 38/111 | 33/84 |
| Dùng câu ban đầu và bài được chọn, 1:1 | 42/111 | 37/84 |
| Chỉ dùng bài được chọn (đối chứng offline) | 43/111 | 38/84 |

Tín hiệu chọn bài được giả lập bằng bài đầu tiên đạt grade ≥ 2 trong top 3. Chín tác vụ thiếu bài như vậy tiếp tục danh sách cũ và đạt 5/27 cho cả ba phương án. Trong 37 tác vụ, mức chênh giữa bản kết hợp và tiếp tục là +3,6 điểm phần trăm, bootstrap 95% [−8,1; +14,3] theo 26 ID bài mẫu. Riêng phần phương pháp giảm từ 7/27 xuống 5/27. Dữ liệu ủng hộ việc cho người đọc thử một hướng khác sau khi thấy bài phù hợp, còn lợi ích trung bình trên người dùng thật cần đo trực tiếp. [Thiết kế mô phỏng](feedback_protocol.md) · [Các bài đã xem, lượt chọn và kết quả từng trang](feedback_comparison.json)

Chạy lại hai nghiên cứu với môi trường ML ở trên. Lần đầu ASPIRE cần tải checkpoint xấp xỉ 420 MiB. Web app chỉ cần chỉ mục đóng gói sẵn.

```bash
PYTHONPATH=. .venv-decision/bin/python scripts/evaluate_contextual.py --cache /tmp/researchlens-csfcube
PYTHONPATH=. .venv-decision/bin/python scripts/evaluate_reading_feedback.py --cache /tmp/researchlens-csfcube
```

## Khi tên phương pháp xuất hiện trong câu

Trên 172 truy vấn SciERC, Reader tìm được bài cùng tên phương pháp trong top 3 ở 69 truy vấn. Thêm BM25 cụm từ bằng hợp nhất thứ hạng đạt 83. Khi che tên, hai cách lần lượt đạt 20 và 19. Thí nghiệm riêng trên 700 cặp dùng Laya xếp lại cùng 20 ứng viên như cross-encoder. [Phân tích ba bộ dữ liệu, điểm từng mô hình và hướng dẫn chạy](method_findings.md)
