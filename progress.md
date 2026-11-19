# Nhật ký tiến độ ResearchLens

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
