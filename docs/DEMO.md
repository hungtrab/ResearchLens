# Kịch bản trình bày ResearchLens

Thời lượng gợi ý: 15–18 phút trình bày + 5 phút demo. Slide 1–23 là nội dung chính, slide 24–27 là phụ lục Q&A. Dừng ở slide kết luận (23) khi trả lời câu hỏi.

| Slide | Trọng tâm trình bày |
|---|---|
| 1–2 | Bài toán và ba câu hỏi nghiên cứu. |
| 3–5 | Snapshot 3.595 paper, kiến trúc CPU, giao diện tìm kiếm. |
| 6 | Kho 129.596 record: coverage từng năm, nguồn và latency riêng. Không gán nDCG cũ cho kho mới. |
| 7 | OpenAlex live search → import → reader/Project; references phải có bằng chứng. |
| 8–10 | Hướng citation, đồng trích dẫn, fusion/hybrid/MMR, review polarity và evidence. |
| 11–12 | Project brief, citation map, PDF thật và notes lưu bền. |
| 13–18 | Protocol, PPR thắng trên reference recovery, diversity, review, SciFact, latency. |
| 19 | Phục hồi reference với graph đến năm 2020. |
| 20–21 | 278 tests và audit; giới hạn proxy evaluation. |
| 21–23 | Nguồn và kết luận; chuyển sang demo. |
| 24–27 | Phụ lục Q&A. |

## Demo 5 phút

1. Mở Discover, nhập `graph neural networks`. Đổi BM25, TF-IDF, Hybrid, và Solr nếu đã bật. Dùng `"graph neural networks"`, `graph AND networks`, `graph -neural` để minh họa query processing.
2. Mở paper, chỉ nguồn, abstract, entities và local/global citations. Lưu 2–3 bài. Không hứa mọi paper đều có review vì join chỉ khớp đúng title chuẩn hóa.
3. Mở citation neighborhood từ paper. Zoom/pan, click nút để đọc paper khác; nhắc graph chỉ có các cạnh nội bộ quan sát được.
4. Mở My library. Gợi ý mặc định là AI decision model; mỗi bài ghi điểm, các tín hiệu và bài nhân chứng. Nhập một câu vào ô Topic, ví dụ "knowledge graph completion with embeddings", rồi bấm Enter để thêm retrieval theo đề tài; lý do có thêm "topic match". Chọn fusion, hybrid hoặc co-citation để đối chiếu, tăng diversity để xem MMR. Export BibTeX.
5. Mở Review insights. Phân tích một review, chỉ đúng câu evidence. Mở Observatory để đối chiếu các chart với JSON thí nghiệm.

## Câu hỏi thường gặp

**Tại sao không dùng Spark hoặc mô hình lớn?** Quy mô 3.595 paper chưa cần distributed compute. Baseline sparse/graph đủ nhanh, rẻ và tái lập; Solr cung cấp Lucene inverted index thực tế.

**Tại sao không dùng RMSE theo hình đặc tả?** Không có user ratings để làm regression. Với ranking, Recall/nDCG/MRR có ý nghĩa; với review classes dùng macro-F1.

**Vì sao mặc định là decision model?** Trên cả ba phép thử nó vượt fusion; khi có Topic, khoảng tin cậy của mức vượt không chứa 0 ở cả ba. Fusion vẫn có trong menu và là phương án dự phòng khi thiếu file mô hình.

**Vì sao trước đó mặc định là fusion mà không phải đồng trích dẫn?** Fusion thắng trên thí nghiệm cắt năm 2020 (0,207 so với 0,173) và không rỗng khi bài đã lưu chưa có ai trích cùng. Trên split ngẫu nhiên, đồng trích dẫn cao hơn (0,305 so với 0,259), nên so sánh này còn mở. Fusion hơn hybrid trên cả hai.

**Có leakage không?** Loại 442 held-out edges trước mọi graph score; lưu seed/target từng source; review train/dev/test paper-disjoint và exact-text-disjoint. Tuy nhiên graph không time-censor nên không tuyên bố dự báo tương lai.

**Các community có phải topic gold không?** Không. Louvain nhóm theo cấu trúc graph, 787 community gồm 770 isolates; chỉ 17 community có hơn một node trong snapshot này.

**Đã production-ready chưa?** Đây là hệ thống local cho môn học, có validation/session isolation và container. Dịch vụ nhiều người dùng cần authentication, quản trị quyền, load testing và vận hành riêng.


## Workspace mới (1.1)

1. Mở Projects → New project → Use an example brief. Tạo project, thêm hai paper khác tiêu đề từ suggestions.
2. Quan sát citation map; bật/tắt Neighbors, Groups; chọn node rồi Read paper. Membership không phải cạnh citation.
3. Tìm “Benchmarking Graph Neural Networks”, paper OpenAlex W3007332492. Mở Paper reading để tải PDF arXiv thật. Nhập bookmark4 rồi Go to bookmark.
4. Viết notes, đổi trạng thái Reading/Read, chờ Saved. Reload để chứng minh lưu bền. Focus mode và export notes hoạt động.
5. Giải thích PDF cần mạng; bài không có PDF được hiển thị abstract rõ nhãn. Nguồn paper trong snapshot có thể khác phiên bản PDF hiện tại.
6. Graph nền than: pan, zoom, kéo node, search, groups, labels; animation tắt theo reduced motion.

Thiết kế áp dụng evondevKit UI/UX; chi tiết, attribution và ý tưởng tiếp theo ở WORKSPACE_DESIGN.md.

## Discovery mở rộng (1.2)

1. Mở Conference library, tìm `graph neural networks`, lọc ICML 2025. Mở bảng coverage để giải thích ô 0 và giới hạn nguồn.
2. Mở Search online, tìm `Flow Matching for Generative Modeling`. Chỉ provider, metadata, nguồn gốc; import một paper chưa có.
3. Thêm paper vừa import vào Project, mở reader và citation neighborhood. Nếu thiếu PDF/references, giải thích trạng thái thật thay vì hứa có toàn văn/mọi cạnh.
4. Mở Observatory: thống kê workspace phản ánh import, các kết quả benchmark cố định vẫn giữ nguyên.

## Tìm bài từ một câu trong abstract

1. Mở Paper reading cho *Learning Entity and Relation Embeddings for Knowledge Graph Completion* (W2184957013).
2. Mở **Find papers from a sentence in this abstract**. Chọn câu bắt đầu bằng “Recently, models such as TransE and TransH” rồi bấm **Find similar papers**.
3. Đọc câu trích bên dưới *Text-enhanced representation learning for knowledge graph*. Mở tựa bài để đối chiếu với abstract gốc. Slide 21 minh họa đúng thao tác này.

4. Nhấn **Find more like this paper** ở một gợi ý để tìm ba bài mới theo cả câu và bài đó. **Show next three** cho xem phương án tiếp tục danh sách ban đầu.
