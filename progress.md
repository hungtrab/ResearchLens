# Nhật ký tiến độ ResearchLens

## 18/11/2026 — Bản bàn giao và kết quả nghiên cứu

ResearchLens giúp người làm đề tài quyết định **nên đọc bài nào tiếp theo**. Người dùng có thể bắt đầu bằng brief của project, một bài mẫu hoặc một câu đang đọc, sau đó đối chiếu gợi ý với abstract, trích dẫn và review.

### Hệ thống đã triển khai

- **Tìm bài.** Đã thu thập 129.596 bản ghi từ ICML, NeurIPS, ICLR, AAAI, EMNLP, NAACL, ACL và CVPR trong giai đoạn 2020–2026. Kho hội nghị dùng SQLite FTS5/BM25, có lọc theo năm và hội nghị. Người dùng có thể tìm thêm qua OpenAlex rồi nhập bài vào project. Độ phủ được ghi riêng theo từng hội nghị và năm. [Dữ liệu và quy trình thu thập](docs/DISCOVERY.md)
- **Làm việc theo đề tài.** Tab Projects lưu brief, gợi ý bài liên quan, quản lý tiến độ đọc và hiển thị đồ thị trích dẫn riêng. Paper reading mở PDF trong trình duyệt, lưu ghi chú, bookmark và cho phép chọn một câu để tìm ba bài đọc tiếp. Chỉ mục Reader bao phủ 18.562 câu trong 2.725 bài có abstract. [Hướng dẫn sử dụng](README.md)
- **Khám phá qua đồ thị.** Đã làm đồ thị phong cách Obsidian với lực lò xo để khi kéo một nút, các nút liên kết cũng chuyển động. Người đọc có thể phóng to, tìm nút, tô sáng các bài lân cận và mở bài trực tiếp. Đã áp dụng hướng dẫn giao diện từ evondevKit, thêm chuyển động nhẹ và hỗ trợ tùy chọn giảm chuyển động. [Quyết định thiết kế và kiểm tra giao diện](docs/WORKSPACE_DESIGN.md)
- **Khai phá dữ liệu.** Snapshot dùng cho thí nghiệm gồm 3.595 bài, 13.481 cạnh trích dẫn và 1.320 review có điểm. Đã triển khai tìm kiếm BM25/TF-IDF, tích hợp Apache Solr cho snapshot, trích từ khóa và tên phương pháp, phân tích ý kiến trong review kèm câu gốc, và gợi ý qua nội dung, Personalized PageRank, đồng trích dẫn. [Audit chức năng](docs/CAPSTONE_AUDIT.md)

### Thí nghiệm đã chạy

| Câu hỏi | Kết quả đã đo | Ý nghĩa đối với hệ thống |
|---|---|---|
| Có tìm lại được tài liệu tham khảo bằng đồ thị? | Trên 219 bài sau mốc 2020, gợi ý bằng đồng trích dẫn đạt nDCG@10 **0,173**, Personalized PageRank đạt **0,121**. | Trong phép thử này, các bài thường được trích cùng nhau giúp tìm lại tài liệu bị che. |
| Có tìm được bài nhắc cùng tên kỹ thuật? | Trên 172 câu SciERC, Reader tìm đúng trong top 3 ở **69** câu. Kết hợp với BM25 dùng cả cặp từ liền nhau đạt **83** câu. Mức tăng là **8,1 điểm phần trăm**, khoảng tin cậy 95% từ **2,6 đến 14,0**. | Khi che tên kỹ thuật trong cùng các câu đó, hai cách đạt **20** và **19**. Lợi ích quan sát được phụ thuộc vào tên kỹ thuật còn xuất hiện. |
| Mô hình quyết định có chọn được bài tốt hơn từ cùng 20 ứng viên? | Laya đã chấm **700 cặp**. Trên 26 câu ACL-ARC, thứ tự kết hợp Reader + BM25 tìm đúng nguồn trong top 3 ở **8** câu, Laya ở **6**, cross-encoder ở **4**. Trên 9 truy vấn CSFCube, số câu có bài hữu ích lần lượt là **6, 5, 4**. | Đã giữ TF-IDF + MiniLM làm mặc định cho Reader. **12/26** nguồn đúng ở ACL-ARC đã vắng trong top 20 trước bước xếp lại. Jev của TypeSafe chưa được đo. |

Cũng đã thử ASPIRE và mô phỏng lượt đọc thứ hai sau khi người đọc chọn một bài phù hợp. ASPIRE tăng số vị trí hữu ích từ 19/42 lên 24/42 ở nhóm bối cảnh và kết quả trên bài mẫu mới, nhưng giảm từ 8/27 xuống 7/27 ở nhóm phương pháp đã khảo sát. Số vị trí hữu ích ở lượt đọc thứ hai tăng từ 38/111 lên 42/111 trong mô phỏng, với khoảng tin cậy cho chênh lệch còn chứa 0. [Thí nghiệm Reader](evaluation/decision-study/README.md) · [Phân tích SciERC, ACL-ARC và CSFCube](evaluation/decision-study/method_findings.md) · [Kết quả đồ thị theo mốc thời gian](artifacts/temporal_recovery.json)

### Kiểm chứng và bàn giao

- Agent audit phát hiện tiêu đề bài nguồn bị lặp trong cả 1.941 bản ghi ACL-ARC và một số bài đích có nhiều tên. Đã bỏ trường tiêu đề lỗi, gộp bản trùng theo ID và tên, rồi tính lại kết quả. Agent dựng lại độc lập **35/35 danh sách top 20** dùng để so sánh Laya với cross-encoder. [Thiết kế phép thử](evaluation/decision-study/method_retrieval_protocol.md)
- Agent kiểm thử bổ sung 33 ca cho phép đo truy hồi, gồm thứ tự cụm từ, xử lý đồng hạng, che tên đúng ranh giới từ và tránh rò rỉ tiêu đề. Toàn bộ **430/430 test đạt**, với khoảng **99% statement coverage** cho package ứng dụng. [Báo cáo test](docs/TEST_REPORT.md)
- Đã biên dịch [report LaTeX 41 trang](deliverables/ResearchLens_Report.pdf) và [slide Beamer HUST 32 trang](deliverables/ResearchLens_Slides.pdf), dùng biểu đồ từ số liệu thực nghiệm và ảnh chụp giao diện. [Gói mã nguồn bàn giao](deliverables/ResearchLens_Source.zip) đã được kiểm tra checksum. Server đã tắt theo yêu cầu.

### Phần còn mở

- Đã tạo phiếu chấm mù cho 43 cặp brief–paper. Các nhãn pilot do trợ lý chấm. Một phiếu người dùng chỉ dùng để thử cách hỏi. Cần lượt chấm độc lập của người đọc để đánh giá gợi ý theo brief và đo mức đồng thuận. [Phiếu và hướng dẫn chấm](evaluation/reading-study/README.md)
- Hướng thử tiếp theo là mở rộng tập ứng viên bằng đồ thị đồng trích dẫn rồi đo xem tìm lại được bao nhiêu trong 12 nguồn ACL-ARC còn thiếu. Phép đo này chưa chạy.
