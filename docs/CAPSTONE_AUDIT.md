# Audit học thuật ResearchLens · 18/11/2026

Trang 7 của syllabus IT4868E liệt kê link analysis, opinion mining và recommendation cho capstone, đồng thời loại Information Retrieval khỏi danh sách đề tài. Trong code phục vụ request, search trả về danh sách paper và opinion classifier dự đoán polarity. Không có mô hình ngôn ngữ sinh câu trả lời từ văn bản truy hồi. Đóng góp dễ bảo vệ nhất là gợi ý tài liệu bằng cấu trúc mạng trích dẫn, có bằng chứng mở được cho từng gợi ý.

| Mức ảnh hưởng | Phát hiện khi audit | Xử lý hoặc giới hạn còn lại |
|---|---|---|
| Cao cho kết luận | Random edge holdout che 442 cạnh nhưng giữ cả các bài xuất bản sau bài đang đánh giá. | Đã thêm phép phục hồi reference với graph và vocabulary giới hạn đến 2020. Hai protocol trả lời hai câu hỏi khác nhau. |
| Cao cho demo | Lời giải thích cũ gán đóng góp PPR cho cả content và global popularity. Slider MMR mặc định khác cài đặt của baseline. | Giải thích riêng cho từng phương pháp, ghi weight khi bật MMR. Giao diện khởi đầu ở 0. Co-citation hiển thị bài làm bằng chứng. |
| Cao cho phạm vi | Kho tìm kiếm 129.596 record và graph thực nghiệm 3.595 paper khác nhau về phạm vi. | Dùng đúng mẫu số khi bảo vệ. Search và Reader giúp sử dụng kết quả mining. Chất lượng recommendation đo trên graph của snapshot nhỏ. |
| Vừa | Review classifier dùng nhãn suy từ rating. Aspect rules chưa có nhãn do người đọc gán. | Diễn giải macro-F1 0,469 cho polarity theo rating. Bước tiếp theo đáng làm là nhờ hai người gán aspect độc lập trên một mẫu câu, đo agreement, rồi tính precision/recall từng aspect trên các nhãn đã thống nhất. |
| Vừa | Mọi phép đo gợi ý hiện dùng reference trong paper thay cho đánh giá của người đọc. | Kết quả cho biết mức phục hồi thư mục tham khảo trong corpus. Hiệu quả đọc thực tế cần người dùng chấm. |

Audit cũng phát hiện toán tử AND xử lý sai khi ngăn bằng tab hoặc xuống dòng. Đã sửa parser và thêm regression test cho hai kiểu khoảng trắng. Huấn luyện lại review classifier cho macro-F1 0,46895 trên 115 review test, khớp số đã công bố.

## Phép thử phục hồi tham chiếu

Cắt theo năm 2020 trước khi dựng graph và TF-IDF. Chọn tất cả 219 bài xuất bản sau mốc có ít nhất 6 reference đến 2.520 bài cũ. Với mỗi bài, che một phần ba số reference cũ, làm tròn xuống. 826 reference bị che trên tổng số 219 tác vụ. Năm của các bài đủ điều kiện nằm trong 2021–2024. Toàn bộ graph 7.881 cạnh chỉ có các bài đến 2020, vì vậy chính bài test không thể trở thành nhân chứng đồng trích dẫn. Cả năm phương pháp gọi cùng hàm recommend dùng trong API, tắt MMR, so top 10. Co-citation có thể trả danh sách ngắn hoặc rỗng, vẫn tính đủ mọi tác vụ.

| Phương pháp | nDCG@10 | Recall@10 |
|---|---:|---:|
| Global PageRank | 0,032 | 0,058 |
| Content | 0,110 | 0,147 |
| Personalized PageRank | 0,121 | 0,141 |
| Hybrid | 0,143 | 0,183 |
| Co-citation | 0,173 | 0,219 |
| Fusion (co-citation + hybrid) | **0,207** | **0,257** |

Mỗi bài trích cả ứng viên và ít nhất một seed được tính một lần. Phương pháp mở rộng phép đếm đồng trích dẫn cổ điển sang tập bài đã lưu. [Small (1973)](https://doi.org/10.1002/asi.4630240406)

Co-citation trả ít nhất một kết quả cho 218/219 tác vụ. Chênh lệch nDCG@10 so với Personalized PageRank là +0,052, khoảng bootstrap ghép cặp 95% theo bài nguồn [0,029; 0,075], 2.000 lần lấy mẫu. Cùng cutoff, cùng candidate pool và cùng tập bài đã lưu. Khoảng này chỉ phản ánh biến thiên giữa các bài test trên một graph. Snapshot được thu thập hồi cứu theo chủ đề, metadata và việc một paper được chọn có thể chịu ảnh hưởng của thông tin sau 2020. Chỉ bài có đủ liên kết nội bộ mới đủ điều kiện. Cần thử nhiều mốc hoặc một corpus khác để kiểm chứng độ bền của kết quả.

## Demo và tái lập

Trong My library, gợi ý mặc định là AI decision model; chọn Fusion hoặc Co-citation để chỉ xem đồng trích dẫn. Mở tên bài làm bằng chứng dưới gợi ý để kiểm tra hai reference. Ví dụ thật trong snapshot, bài *Knowledge Transfer for Out-of-Knowledge-Base Entities* (W2963406064) trích cả *Learning Entity and Relation Embeddings for Knowledge Graph Completion* (W2184957013) và *Knowledge Graph Embedding by Translating on Hyperplanes* (W2283196293).

Chạy `uv run python scripts/evaluate_temporal.py`. Tệp `artifacts/temporal_recovery.json` lưu hash của snapshot, tập seed/target, top 10 và hai metric của từng phương pháp trên từng paper. Phần thử nghiệm theo thời gian không dùng OpenAlex trực tuyến.

Phép thử tác giả đọc gì tiếp (E2b) chạy bằng `uv run python scripts/evaluate_author_next.py` và ghi `artifacts/author_next.json`. Hồi quy logistic chạy bằng `uv run python -m scripts.evaluate_ltr` sau hai script trên, và ghi `artifacts/ltr.json`. Hai script đọc tác vụ đã cố định từ artifact, nên số liệu không đổi giữa các lần chạy. Phép thử SPECTER cần thêm torch và sentence-transformers, không cài cùng app; lệnh chạy nằm ở đầu `scripts/embed_specter.py`, sau đó chạy `uv run python -m scripts.evaluate_specter`. Decision model huấn luyện bằng `uv run python -m scripts.decision_model` (khoảng 2 phút, cần `artifacts/specter.npz`), ghi `artifacts/decision_model.json` và `artifacts/decision_model.pkl`; app nạp hai file này khi khởi động, thiếu thì phương pháp `decision` trả 422. Thêm `--query` để huấn luyện mô hình đọc Topic (`artifacts/decision_query_model.*`); `--variant gbdt_rank` tạo lại đúng mô hình đang chạy. Tốc độ gợi ý: `uv run python -m scripts.benchmark_recommend` (`artifacts/latency_recommend.json`). Lời giải thích SLM + RAG: `make slm` rồi `uv run python -m scripts.evaluate_explainer` (`artifacts/explainer.json`; vòng 1–2 trong `explainer_round1.json`, `explainer_round2.json`; kết quả đọc tay trong `explainer_audit.json`, `explainer_round2_audit.json`). `tests/test_report_numbers.py` báo lỗi nếu một con số trong report hoặc slide lệch khỏi artifact.
