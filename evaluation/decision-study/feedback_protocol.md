# Thử phản hồi sau ba gợi ý đầu

Lượt đầu giữ nguyên Reader TF-IDF + MiniLM. Mô phỏng người đọc chọn bài đầu tiên trong ba gợi ý có grade chuyên gia ≥ 2. Khi cả ba đều chưa đạt ngưỡng, tiếp tục danh sách cũ. Loại cả ba bài đã xem trước mọi phép xếp hạng lượt sau. Đây là thí nghiệm offline có giả định về hành vi người đọc.

Ba phương án cho lượt thứ hai: tiếp tục danh sách cũ, chỉ dùng bài được chọn, hoặc dùng đều câu truy vấn ban đầu và bài được chọn. Với phương án cuối, lấy trung bình hai logit của policy văn bản đóng băng, tương đương dùng trọng tâm hai vector truy vấn trong từng không gian tuyến tính, không chuẩn hóa lại. Trọng số 1:1 được cố định trước khi đo. Không dùng grade của bất cứ bài nào ngoài top 3 để tạo danh sách tiếp theo.

Dùng toàn bộ tác vụ CSFCube có ID bài mẫu ngoài tám ID method/fold1_dev dùng hiệu chỉnh mô hình gốc. Đo số bài hữu ích trong ba vị trí tiếp theo và báo riêng nhóm có thể nhấn chọn lẫn toàn bộ tác vụ. Bootstrap ghép cặp theo ID bài mẫu. Dùng split này làm thí nghiệm thăm dò hành vi sau tương tác; nhiều bài đã xuất hiện trong thí nghiệm lần đầu. Chỉ thêm lựa chọn này vào giao diện nếu cách kết hợp vượt tiếp tục danh sách về số bài hữu ích trên toàn bộ tác vụ đánh giá.
