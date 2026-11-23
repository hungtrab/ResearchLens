# Nhật ký tiến độ ResearchLens

## 23/11/2026 — Viết lại report bằng tiếng Anh, LaTeX cổ điển

- `report/main.tex` và `report/discovery.tex` viết lại toàn bộ bằng tiếng Anh. Dùng `article` 11pt, có abstract, mục lục, danh sách hình và bảng, natbib `plainnat`, không dùng màu. Số liệu giữ nguyên, chỉ đổi dấu thập phân sang dấu chấm (0.207). Slide vẫn bằng tiếng Việt.
- Câu chữ: không có em dash, không có "--" trong văn bản, không ghép từ bằng gạch nối (chỉ còn tên riêng như TF-IDF, ACL-ARC, scikit-learn và các flag trong lệnh shell), không có từ hype. Đã kiểm bằng grep.
- Hình vẽ theo phong cách figures4papers: `scripts/report_figures.py` sinh 13 hình PDF và PNG vào `artifacts/figures/report/`. `make figures` đã gọi script này.
- Thêm 5 mục vào `references.bib` (Small 1973, SciERC, ACL-ARC, CSFCube, ASPIRE). Ghi chú trong bib đã dịch sang tiếng Anh.
- `tests/test_report_numbers.py`: phần report kiểm số dạng tiếng Anh, phần slide vẫn kiểm dạng tiếng Việt. Mean và CI95 giờ được kiểm trong cùng một chuỗi, nên số test giảm từ 579 xuống 574. Kết quả: 574/574 test đạt, coverage 1.391/1.405 statement.
- Report không còn nhắc đường dẫn file, tên script hay tên hàm. Bỏ phụ lục cách chạy lại và phụ lục danh mục artifact (cách chạy vẫn có trong README).
- Thêm progress report 3 trang theo yêu cầu của thầy: `report/progress.tex`, xuất ra `deliverables/Group05_ProgressReport.pdf`. Nhóm 5: Tran Quang Hung 20235502, Vu Thuong Tin 20230091, Chu Anh Duc 20230081, Dinh Quoc Bao 20235478. Không có link repo, vì thầy không yêu cầu.
- Bỏ phụ lục demo năm phút. Report còn 40 trang.
- Bìa report: logo ĐHBKHN lớn (bản 1900 px từ hust.edu.vn, đổi chữ trắng sang đỏ) và logo SoICT bên dưới (bản gốc từ soict.hust.edu.vn), lưu ở `report/images/`.
- Kiểm lại 27 tài liệu tham khảo. Tên tác giả, tên bài, tập và trang khớp với Crossref hoặc OpenAlex. Thêm DOI cho SciERC, ACL-ARC, ASPIRE, PeerRead và SciFact. CSFCube trỏ tới OpenReview. Sách của Manning dùng DOI, vì trang Stanford không truy cập được. DOI trong PDF giờ là link bấm được (gói `doi`). Mọi link đều tới đúng trang; ACM, Wiley, IOP, MIT Press và Emerald chặn curl với mã 403 nhưng doi.org chuyển hướng đúng bài. Riêng Robertson và Zaragoza (2009) giữ tập 3(4), trang 333–389 như bản gốc; Crossref hiện ghi 4(1-2), trang 1–174 sau khi Emerald đăng lại.
- Report 40 trang, không có overfull box hay ký tự thiếu. Gói zip đã build lại và giải nén thử: 573 test đạt, 1 test bỏ qua vì không có model GGUF. Không có `llm_api/` và không có chuỗi `nvapi-` trong zip.

## 20/11/2026 — SLM và LLM làm bộ xếp hạng: không vượt decision model

- **Câu hỏi:** để mô hình ngôn ngữ xếp hạng thay decision model có tốt hơn không? Cùng 20 ứng viên của fusion, 589 tác vụ (E2, E2b, E3), decision và fusion chạy lại trên đúng tác vụ đó (khớp `decision_model.json`). Luật đặt trước: thắng decision ở ≥ 2/3 phép thử với CI95 > 0.
- SLM (Qwen2.5 0.5B/1.5B, llama.cpp CPU): Yes/No cho từng ứng viên. LLM (GLM-5.3-flash, API NVIDIA): xếp cả 20 bài một lần, kiểu RankGPT.

| nDCG@10 | E2 | E2b | E3 | p50 mỗi lượt |
|---|---:|---:|---:|---:|
| Decision (app, ứng viên riêng) | 0,224 | 0,132 | 0,297 | 85 ms |
| Fusion | 0,207 | 0,113 | 0,259 | 21 ms |
| **Decision (cùng 20 ứng viên)** | **0,219** | **0,131** | **0,293** | 85 ms |
| SLM 0.5B | 0,148 | 0,078 | 0,155 | 3,48 s |
| SLM 1.5B | 0,186 | 0,114 | 0,193 | 6,17 s |
| LLM GLM-5.3-flash | 0,204 | 0,122 | 0,250 | 110,61 s |
| LLM + graph RAG | 0,202 | 0,127 | 0,303 | 221,95 s |

- Không mô hình nào đạt luật. LLM chỉ đọc văn bản ngang fusion; thấp hơn decision ở cả ba phép thử nhưng CI chứa 0.
- Graph RAG: mỗi ứng viên thêm một dòng bằng chứng trích dẫn (đồng trích dẫn với bài đã lưu, liên kết trích dẫn với bài đã lưu, số lần được trích). Ở E3 cao hơn LLM chỉ văn bản +0,053 [0,017; 0,090] và vượt fusion +0,043 [0,006; 0,079]; so với decision trên cùng 20 ứng viên vẫn −0,017, −0,004, +0,009, CI chứa 0.
- Chạy sạch: giới hạn 16.384 token (8.192 làm 31% prompt graph bị cắt khi đang suy luận). Còn 15/589 câu graph và 0/589 câu văn bản bị cắt. Gọi API bằng streaming, vì gateway trả HTTP 504 cho các câu suy luận lâu.
- 100 ứng viên (21/11): 20 ứng viên chỉ chứa 36%/17%/47% bài đúng; 100 chứa 66%/37%/75%. Đưa hết 2.520–3.595 bài thì không còn là RAG (SLM ~10 phút/tác vụ, prompt LLM ~200k token). LLM được hỏi chọn và xếp top 10 trong 100.

| nDCG@10, cùng 100 ứng viên | E2 | E2b | E3 | p50 |
|---|---:|---:|---:|---:|
| Decision | 0,229 | 0,130 | 0,296 | 85 ms |
| SLM 0.5B | 0,067 | 0,040 | 0,075 | 14,00 s |
| SLM 1.5B | 0,127 | 0,068 | 0,088 | 24,71 s |
| **LLM + graph RAG** | **0,242** | **0,135** | **0,310** | 202,13 s |

- LLM + graph RAG cao hơn decision ở cả ba (+0,013, +0,006, +0,013) nhưng CI chứa 0, không đạt luật; vượt fusion có ý nghĩa ở cả ba. 28/589 câu hết token. ~9,7 triệu token. SLM tệ đi với nhiều ứng viên. Chưa chạy LLM chỉ văn bản với 100 ứng viên.
- Rà lại implementation (21/11): bảng cũ ghi "cùng 20 ứng viên" nhưng decision model chạy trên tập ứng viên riêng (~300 bài); 33% top 10 của nó nằm ngoài 20 bài của fusion. Thêm baseline decision xếp lại đúng 20 bài (0,219/0,131/0,293), luật so với baseline này; kết luận không đổi. Với baseline này SLM 1.5B thua có ý nghĩa ở cả ba phép thử, LLM văn bản thua có ý nghĩa ở E3. SLM tính lại từ điểm từng tác vụ đã lưu (`--rescore`), LLM từ cache, không gọi lại model.
- Cũng sửa: chỉ đọc số dạng `[k]` trong câu trả lời (48 câu có thêm đoạn giải thích sau thứ tự; kết quả không đổi); câu rỗng đã hỏi ở 16.384 token không hỏi lại; GLM không tất định ở temperature 0, nên prompt trùng dùng một câu trả lời trong cache (graph E2b 0,128 → 0,127).
- Lỗi tìm ra khi viết SLM ranker: tái dùng KV cache theo prefix dài nhất làm điểm phụ thuộc thứ tự chấm (−0,27 thành −0,18). Sửa: đọc phần thư viện một lần, mỗi ứng viên luôn nối ngay sau.
- Key API nằm ở `llm_api/.env`, không vào git (`.gitignore`) và không vào gói zip.

## 19/11/2026 (đêm) — SLM + RAG cho lời giải thích, đo và tối ưu tốc độ

- **Decision model vẫn là phương pháp chính.** SLM chỉ viết lời giải thích khi bấm *Explain with AI*, không đổi thứ tự.
- **Tốc độ gợi ý** (warm, in-process, 150 lần): đồng trích dẫn 0,7 ms, fusion 20,8 ms, decision 84,7 ms, decision + đề tài 89,0 ms; HTTP thật 53 ms. Trước tối ưu decision mất 222 ms: bỏ tính lặp PageRank/content (143 ms), giới hạn OpenMP 1 luồng khi predict (85 ms). 120 lượt gợi ý giống hệt trước/sau.
- **SLM + RAG, ba vòng** (Qwen2.5 Q4_K_M, llama.cpp, 8 nhân; luật chọn đặt trước: qua kiểm tra cao nhất với p50 ≤ 3 s):
  - Vòng 1, SLM viết cả dữ kiện đồ thị: gán sai số đếm, đảo chiều trích dẫn.
  - Vòng 2, SLM đọc hai abstract: nhầm bài đã lưu thành bài gợi ý; đọc tay 0.5B đúng 3/20, 1.5B 4/10.
  - Vòng 3, SLM chỉ tóm tắt abstract bài gợi ý, app viết quan hệ đồ thị; đo trên 30 thư viện mới: 0.5B p50 0,36 s, đọc tay **17/20** đúng (tiêu chí ≥ 16/20 đạt). Qua kiểm tra 40% vì 31/90 bài không có abstract.
- **Lỗi tốc độ tìm ra:** llama.cpp mặc định 16 luồng đọc prompt trên 8 nhân, token đầu 4,21 s; đặt 8 luồng còn 0,42 s.
- Cài: `make slm` (llama-cpp-python + model 0,4 GB). Không cài thì nút ẩn, API trả 503.

## 19/11/2026 (khuya, tiếp) — Decision model làm mặc định, thêm retrieval theo đề tài

- **Decision model là mặc định** ở My library theo quyết định của chủ project. Fusion là phương án dự phòng khi thiếu file mô hình.
- **Vòng cải tiến 2 (kết quả âm).** Thêm tín hiệu fusion và mô hình lớn hơn: validation 0,303 so với 0,258, nhưng E3 giảm từ 0,297 còn 0,260. Giữ bản cũ. Lưu ở `artifacts/decision_model_round2.json`.
- **Decision + retrieval theo đề tài.** Ô Topic tùy chọn: BM25 lấy thêm 100 ứng viên, thêm 2 tín hiệu (BM25, TF-IDF với đề tài), một mô hình riêng xếp hạng lại.

| nDCG@10 | Fusion | Chỉ BM25 | Decision | Decision + đề tài | Hit@10 |
|---|---:|---:|---:|---:|---:|
| E2 | 0,207 | 0,107 | 0,224 | **0,237** | 0,644 |
| E2b | 0,113 | 0,135 | 0,132 | **0,182** | 0,584 |
| E3 | 0,259 | 0,087 | 0,297 | **0,304** | 0,783 |

  So với fusion: +0,030 [0,006; 0,055], +0,069 [0,051; 0,087], +0,045 [0,005; 0,087]. Cả ba khoảng tin cậy không chứa 0.
- **Chỉ số chính là Hit@10 / Hit@5** (có ít nhất một bài đúng trong 10 / 5 gợi ý đầu):

| | Fusion | Decision | Decision + đề tài |
|---|---:|---:|---:|
| E2 Hit@10 / Hit@5 | 58% / 47% | 65% / 48% | 64% / 49% |
| E2b Hit@10 / Hit@5 | 42% / 28% | 44% / 33% | **58% / 45%** |
| E3 Hit@10 / Hit@5 | 71% / 57% | 75% / 62% | **78% / 63%** |
- Đề tài trong phép thử là thứ người dùng có thật: bài đang viết (E2, E3) hoặc tiêu đề bài mới của tác giả (E2b).
- Lệnh: `uv run python -m scripts.decision_model --query`. Artifact: `artifacts/decision_query_model.json`, `.pkl`.

## 19/11/2026 (khuya) — AI decision model

Decision model của Astra là Laya zero-shot ở phần Reader, chưa từng được dạy bài toán đọc tiếp. Bản mới học từ chính dữ liệu của bài toán.

- **Cách học.** Cắt đồ thị ở mỗi năm 2010–2018, tự tạo tác vụ có đáp án theo hai protocol (che reference, tác giả đọc gì tiếp), không cần người gán nhãn. 1.862 tác vụ huấn luyện (2010–2017), 219 tác vụ validation (2018). Cạnh bị che của E3 bị loại khỏi mọi dữ liệu huấn luyện.
- **Mô hình.** Gradient boosting trên 10 tín hiệu (đồng trích dẫn, content, PPR, quan hệ trích dẫn với bài đã lưu, reference chung, số trích, tuổi bài, hai cosine SPECTER) cộng hạng phần trăm của chúng trong từng tác vụ. Chọn trên validation trong 4 phương án.
- **Kết quả test (một lần).**

| nDCG@10 | Fusion | Decision model | Chênh lệch [95%] | Hit@10 |
|---|---:|---:|---|---:|
| E2 | 0,207 | **0,224** | +0,017 [−0,004; 0,038] | 0,653 |
| E2b | 0,113 | **0,132** | +0,019 [0,004; 0,034] | 0,440 |
| E3 | 0,259 | **0,297** | +0,038 [0,001; 0,074] | 0,750 |

- **Quyết định.** Tiêu chí đặt trước đòi CI không chứa 0 ở cả ba; E2 chưa đạt. Fusion giữ mặc định. App có lựa chọn "AI decision model (experimental)" với điểm và tín hiệu cho từng gợi ý, khoảng 0,2 s mỗi lượt. Ảnh: `artifacts/screenshots/decision_model.png`.
- **Trung thực.** Lần chạy đầu chưa có validation và đã nhìn test một lần; report ghi lại điều này. Điểm của model không phải xác suất đã hiệu chỉnh.
- Code: `researchlens/decision.py` (dùng chung cho app và script), `scripts/decision_model.py`, `tests/test_decision.py`. 541/541 test, 99% coverage.

## 19/11/2026 (tối) — Kiểm số liệu bằng test, thử SPECTER và hồi quy logistic trên E3

Làm theo `~/shared/laura_methodology.md`: chốt tiêu chí trong `.audit/tighten.tsv` trước khi đo, mỗi bước có kiểm chứng riêng.

- **U1, kiểm số liệu bằng test.** `tests/test_report_numbers.py` đọc artifact và kiểm mọi con số gợi ý trong report và slide, kể cả từng dòng bảng. Cài lại lỗi cũ 0,192/0,223 làm test đỏ. Một lỗi chỉ sai một ô bảng lúc đầu lọt qua, nên đã thêm kiểm tra theo dòng. Test cũng bắt được một lỗi làm tròn mới khi viết mục SPECTER.
- **U2, trích dẫn.** McNee 2002, Strohman 2007, Beel 2016 đã tra trên web, đúng tác giả, tiêu đề, trang. Đã thêm DOI. Thêm Cohan 2020 (SPECTER).
- **U3, SPECTER1 thay TF-IDF.** Tiêu chí: content tăng ở cả E2 và E2b, khoảng tin cậy không chứa 0. Kết quả: +0,020 [−0,005; 0,046] và +0,002 [−0,014; 0,019]. Fusion dùng SPECTER còn kém hơn −0,020 [−0,038; −0,003] ở E2. **Không đạt, app giữ TF-IDF.** Dùng SPECTER1 thay SPECTER2 để tránh model đã thấy trích dẫn sau 2020.
- **U4, hồi quy logistic.** Tiêu chí: một mô hình thắng fusion trên E2, E2b và E3, khoảng tin cậy không chứa 0. Mô hình học trên che reference: E2 +0,023, E3 +0,037 [0,009; 0,066], nhưng E2b chỉ +0,010 [−0,002; 0,023]. **Không đạt, app giữ fusion.**
- 528/528 test đạt, 99% coverage.

### Còn mở
- Một phép thử riêng cho bài mới chưa ai trích, nơi content thực sự quan trọng.
- Học trọng số hồi quy logistic trên nhiều phép thử cùng lúc, rồi đo trên một mốc năm chưa dùng.
- 43 cặp brief–paper vẫn cần người chấm.

## 19/11/2026 (chiều) — Phép thử gần thực tế hơn và learning-to-rank

Phép thử che reference có lợi sẵn cho tín hiệu trích dẫn, vì đáp án là reference. Đáp án cũng là bài tác giả đã biết. Vì vậy có thêm **E2b, tác giả đọc gì tiếp**. Thư viện là các bài một tác giả viết đến 2020 cộng với reference của chúng. Đáp án là bài họ trích lần đầu sau 2020. Kết quả gồm 250 tác giả và 1.899 đáp án.

| | nDCG@10 E2 | Hit@10 E2 | nDCG@10 E2b | Hit@10 E2b |
|---|---:|---:|---:|---:|
| Global PageRank | 0,032 | 0,178 | 0,047 | 0,308 |
| Hybrid | 0,143 | 0,416 | 0,070 | 0,236 |
| Đồng trích dẫn | 0,173 | 0,525 | 0,099 | 0,396 |
| Fusion (mặc định) | 0,207 | 0,580 | 0,113 | 0,420 |
| **Hồi quy logistic** | **0,231** | **0,658** | **0,136** | **0,452** |

- E2b khó hơn E2 với mọi phương pháp. Fusion vẫn đứng đầu trong các phương pháp của app.
- Hồi quy logistic trên tám tín hiệu có tên được huấn luyện ở mốc 2018. Nó vượt fusion +0,023 [0,006; 0,040] ở E2 và +0,022 [0,007; 0,040] ở E2b.
- Ở E2b, độ mới và số lượt được trích có trọng số lớn nhất. Tác giả thật hay đọc bài mới và nổi tiếng.
- Hồi quy logistic chưa vào app. Lý do là mức tăng nhỏ, và nó cần kiểm tra lại trên E3 trước.
- Hạn chế: dữ liệu chỉ có tên tác giả, nên người trùng tên bị gộp làm một.
- Script: `scripts/evaluate_author_next.py`, `scripts/evaluate_ltr.py`. Report có mục 6.3 mới và trích dẫn McNee 2002, Strohman 2007, Beel 2016. Slide có trang "Phép thử gần thực tế hơn".
- Đã thử SPECTER ở mục trên: không cải thiện.

## 19/11/2026 — Siết lại bài toán, đổi gợi ý mặc định sang fusion

ResearchLens giải một bài toán: **từ các bài người dùng đã lưu, gợi ý bài nên đọc tiếp**, kèm lý do người dùng tự kiểm tra được. Tìm kiếm là cách lưu bài đầu tiên. Review là bằng chứng hiển thị cạnh gợi ý.

### Câu hỏi nghiên cứu và kết quả

Phép thử: che một phần ba reference của một bài, dùng phần còn lại làm tập đã lưu, đo xem hệ thống tìm lại được bao nhiêu. Thí nghiệm chính cắt corpus ở năm 2020, nên bài kiểm tra không có mặt trong đồ thị. Mọi phương pháp gọi cùng hàm `MiningEngine.recommend` của ứng dụng.

| | nDCG@10 trên 219 bài sau 2020 | nDCG@10 trên split ngẫu nhiên 120 bài |
|---|---:|---:|
| Content | 0,110 | 0,070 |
| Personalized PageRank | 0,121 | 0,174 |
| Hybrid (mặc định cũ) | 0,143 | 0,112 |
| Đồng trích dẫn | 0,173 | **0,305** |
| **Fusion (mặc định mới)** | **0,207** | 0,259 |

- **RQ1.** Trong các tín hiệu đơn, đồng trích dẫn tìm lại tốt nhất trên cả hai protocol.
- **RQ2.** Fusion vượt hybrid trên cả hai protocol: +0,064, khoảng tin cậy 95% [0,042; 0,087], và +0,147 [0,115; 0,178]. So với đồng trích dẫn đứng riêng, fusion thắng ở thí nghiệm chính (+0,034 [0,016; 0,054]) nhưng thua ở split ngẫu nhiên (−0,046 [−0,083; −0,007]). Vì vậy chưa kết luận được giữa hai cách này.
- **Vì sao mặc định là fusion.** Fusion thắng trên thí nghiệm chính. Với diversity mặc định 0,15 của app, fusion vẫn đạt 0,206 so với 0,152 của hybrid. Fusion cũng không trả danh sách rỗng khi bài đã lưu chưa có ai trích cùng, nhưng lý do này yếu: đồng trích dẫn chỉ rỗng ở 1/219 bài, và một luật dự phòng cũng giải quyết được. Hằng số k=60 lấy từ bài gốc của RRF, không chỉnh trên dữ liệu. Fusion được chọn sau khi đã xem kết quả temporal. Split ngẫu nhiên đã cố định từ trước, và được dùng làm phép kiểm tra lại.
- Đã thử một giả thuyết cho kết quả trái chiều: nhân chứng là bài xuất bản sau bài nguồn. Giả thuyết không đứng vững. Bỏ các nhân chứng đó thì đồng trích dẫn chỉ giảm từ 0,305 xuống 0,301. Chạy lại bằng `scripts/check_future_witnesses.py`.

[Report, mục 6.1–6.2](deliverables/ResearchLens_Report.pdf) · [Kết quả temporal từng bài](artifacts/temporal_recovery.json) · [Kiểm tra lại trên split ngẫu nhiên](artifacts/holdout_fusion.json)

### Code đã đổi

- `researchlens/mining.py`: `recommend` được tách thành các helper ngắn có tên rõ: `co_citation_witnesses`, `content_graph_scores`, `reciprocal_rank_fusion`, `mmr_select` và `fusion_reason`. Mỗi helper dài tối đa 15 dòng. Năm phương pháp cũ trả kết quả giống hệt bản trước trên 310 ca của corpus thật. Fusion là mặc định ở engine, API và giao diện My library.
- Mỗi gợi ý fusion ghi hạng đồng trích dẫn, hạng hybrid và tối đa ba bài nhân chứng.
- Đã xóa nhánh gọi API thương mại chưa từng chạy trong `scripts/reading_decisions.py`.
- **441/441 test đạt**, 99% statement coverage. `tests/test_fusion.py` có 10 ca kiểm tra hành vi. Bảy lỗi được cố tình cài vào code đều làm ít nhất một ca thất bại.

### Report và slide

- Mục bài toán giờ có một bài toán và hai RQ. Thí nghiệm cắt theo năm là E2, thí nghiệm chính.
- Đã thêm công thức fusion, kèm trích dẫn Cormack và cộng sự (2009).
- Các thí nghiệm cho chức năng Reader (CSFCube, SciERC, ACL-ARC, Laya, ASPIRE) đã chuyển xuống phụ lục B của report và phụ lục D của slide, vì chúng thuộc truy hồi và có ít truy vấn.
- [Report LaTeX 44 trang](deliverables/ResearchLens_Report.pdf), [slide Beamer 35 trang](deliverables/ResearchLens_Slides.pdf) và [gói mã nguồn](deliverables/ResearchLens_Source.zip) đã được build lại. Checksum nằm trong `deliverables/manifest.json`.
- Thư mục đã có git local. Commit `474dbf8` là trạng thái trước khi sửa, dùng `git diff 474dbf8` để xem toàn bộ thay đổi.

### Phần còn mở

- **Gợi ý theo brief của Project chưa có người chấm.** 43 cặp brief–paper mới có nhãn pilot do trợ lý chấm. Cần ít nhất hai người đọc chấm độc lập bằng [phiếu chấm](evaluation/reading-study/annotate.html) để có số liệu và mức đồng thuận.
- **Fusion và đồng trích dẫn chưa phân xử được.** Bước tiếp theo là lặp phép cắt theo năm trên các mốc khác. Chưa thử PPR thay hybrid làm thành phần thứ hai của fusion. Một hướng là cho trọng số của hai hạng phụ thuộc số nhân chứng, chọn trên một mốc rồi khóa trước khi đo mốc khác.
- `scripts/evaluate.py` (split ngẫu nhiên) vẫn tự tính điểm, không gọi `recommend` của app. Bảng E3 trong report vì vậy chưa có fusion. Phần kiểm tra fusion trên split này nằm ở `scripts/evaluate_holdout_fusion.py`.
