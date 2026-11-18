# Nhật ký tiến độ ResearchLens

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
- **Vì sao mặc định là fusion.** Fusion thắng trên thí nghiệm chính, và không trả danh sách rỗng khi bài đã lưu chưa có ai trích cùng. Hằng số k=60 lấy từ bài gốc của RRF, không chỉnh trên dữ liệu. Fusion được chọn sau khi đã xem kết quả temporal. Split ngẫu nhiên đã cố định từ trước, và được dùng làm phép kiểm tra lại.
- Đã thử một giả thuyết cho kết quả trái chiều: nhân chứng là bài xuất bản sau bài nguồn. Giả thuyết không đứng vững. Bỏ các nhân chứng đó thì đồng trích dẫn chỉ giảm từ 0,305 xuống 0,301.

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
- **Fusion và đồng trích dẫn chưa phân xử được.** Bước tiếp theo là lặp phép cắt theo năm trên các mốc khác. Một hướng là cho trọng số của hai hạng phụ thuộc số nhân chứng, chọn trên một mốc rồi khóa trước khi đo mốc khác.
- `scripts/evaluate.py` (split ngẫu nhiên) vẫn tự tính điểm, không gọi `recommend` của app. Bảng E3 trong report vì vậy chưa có fusion. Phần kiểm tra fusion trên split này nằm ở `scripts/evaluate_holdout_fusion.py`.
