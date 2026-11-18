# Kiểm tra xếp hạng theo ngữ cảnh câu

Giữ một câu người đọc chọn, bài mẫu và pool CSFCube như lần đo Reader. Đáp án hữu ích là grade ≥ 2. Chỉ số chính là số bài hữu ích trong ba vị trí đầu. Chỉ số thứ hai là macro nDCG@10 với gain 2^grade−1. Giữ tất cả truy vấn đủ điều kiện, kể cả truy vấn không có kết quả hữu ích trong top 3.

**Phát triển.** Dùng tám bài mẫu method/fold1_dev để chọn trọng số ASPIRE trong tập {0, 0.25, 0.5, 0.75, 1}. Hai nguồn xếp hạng là policy TF-IDF + MiniLM đóng băng và khoảng cách L2 nhỏ nhất giữa câu ASPIRE đã chọn và các câu ứng viên. Kết hợp bằng reciprocal rank với hằng số 60. Chọn P@3 cao nhất, sau đó nDCG@10 cao nhất, sau cùng ưu tiên trọng số nhỏ hơn. Các điểm của tám truy vấn phục vụ lựa chọn.

**Kiểm tra.** So các phương án trên 9 truy vấn method và 6 truy vấn result từ lần đo Reader trước để đối chiếu trực tiếp. Riêng phép kiểm tra chuyển giao dùng tất cả tác vụ background/result mà ID bài mẫu chưa xuất hiện trong toàn bộ 17 bài mẫu method hoặc 6 bài mẫu result đã xem. Khóa trọng số bằng tập phát triển trước khi tính bảng kiểm tra. Báo cả hai khía cạnh và tổng. Bootstrap ghép cặp 10.000 lượt theo ID bài mẫu vì một bài có thể tạo hai tác vụ.

**Quyết định cho Reader.** Chỉ đưa cấu hình được chọn vào Reader khi cả P@3 và nDCG@10 cao hơn baseline trên tập chuyển giao và số bài hữu ích trên 9 truy vấn method ít nhất bằng baseline. Quy tắc này là tiêu chí triển khai trên snapshot. ASPIRE từng được tác giả đánh giá trên CSFCube. Tập chuyển giao mới đối với nhóm dự án và thuộc cùng nguồn CSFCube.

Nguồn mô hình: [ASPIRE single-match CS](https://huggingface.co/allenai/aspire-contextualsentence-singlem-compsci). Nguồn nhãn: [CSFCube](https://github.com/iesl/CSFCube).
