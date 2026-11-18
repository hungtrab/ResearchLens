# Bạn có lưu bài này để đọc tiếp không?

Mở **annotate.html** trong trình duyệt. Chọn một biệt danh. Mỗi lượt chỉ có một việc muốn làm và một bài nghiên cứu. Đọc phần giới thiệu ngắn bằng tiếng Việt rồi chọn hành động phù hợp. Có thể mở abstract gốc, chọn “chưa rõ”, dừng giữa chừng và xuất JSON. File dùng được khi server tắt.

Ví dụ, mục tiêu là hiểu câu “pin tốt nhưng camera kém”. Người đọc quyết định một bài về phân loại cảm xúc có giúp ứng dụng tìm được **cả lời khen pin và lời chê camera** hay không. Không cần biết tên thuật toán. Hai người có thể chọn khác nhau và ghi lý do nếu muốn.

## Thiết kế và tình trạng nhãn

Bốn ca được chọn chủ đích, gồm gợi ý từ tương tác thưa, hoàn thiện đồ thị tri thức, ý kiến theo khía cạnh và xếp hạng tài liệu. Mỗi ca có một seed. Với mỗi phương pháp trong năm phương pháp, lấy tối đa ba kết quả đầu sau loại trùng tiêu đề chuẩn hóa và seed, từ cửa sổ 50 ứng viên. Hợp các kết quả thành 43 cặp brief–paper. File blind_packet.json cố định fingerprint. Form gom theo đề tài, xáo thứ tự bài trong đề tài bằng seed cố định và ẩn nguồn xếp hạng. Các mô tả tiếng Việt là bản lược dịch do trợ lý viết từ metadata, có thể ảnh hưởng cách hiểu. Người chấm có thể đối chiếu bản gốc.

Có **43 nhãn pilot do trợ lý đánh giá** từng bài, với lý do trong assistant_pilot.json. Hai cặp có nhãn U vì thiếu bằng chứng. Một phiếu số 3 do người dùng cung cấp trong lúc thử cách hỏi được đánh dấu calibration_only vì sau đó người dùng phản hồi rằng câu hỏi khó hiểu. Chưa có bộ nhãn người thật hoàn chỉnh. Không dùng phiếu thử để đo hiệu quả hoặc độ đồng thuận. Để tạo nhãn tham chiếu, để từng người chấm độc lập trước khi mở kết quả mô hình hoặc nhãn pilot. Nếu dùng hai người, giữ lại hai phiếu gốc, xem các mã bất đồng và lưu mức đã thống nhất trong một phiếu mới.

Chạy script từ thư mục project:

```bash
uv run python scripts/build_reading_study.py
uv run python scripts/score_reading_study.py evaluation/reading-study/assistant_pilot.json
uv run python scripts/score_reading_study.py reading-TV01.json --compare-human reading-TV02.json
```

Điểm bằng nDCG@3 với gain 2^grade−1, IDCG trên toàn pool của cùng brief. Bài thiếu trong top 3 đóng góp 0. Topic có phiếu chưa trả lời được đánh dấu pending. Với nhãn U, thử tất cả mức 0–3 để báo cận dưới/trên (topic có hơn năm U tạm hoãn). Các khoảng này là khoảng do thiếu thông tin, không phải khoảng tin cậy. Trên bốn brief theo nhãn pilot, content và hybrid đều đạt 0,804–0,901, co-citation đạt 0,571–0,643. Trong nghiên cứu che cạnh trước đó, co-citation dẫn đầu. Hai phép thử có đáp án, cutoff và số seed khác nhau. Chưa thể quy chênh lệch riêng cho cách gán nhãn, hay tổng quát từ bốn brief sang người dùng.

## Thử mô hình chọn hành động

Thí nghiệm số 1 đưa mục tiêu, seed, tiêu đề và abstract cho mô hình chọn bỏ qua, đọc nền, hữu ích một phần, ưu tiên đọc hoặc thiếu thông tin. Thí nghiệm số 2 hỏi “có lưu không?” và gom nhãn 2/3 thành lưu, 0/1 thành bỏ qua. Thí nghiệm số 2 được thiết kế sau khi xem kết quả số 1; cả hai đều mang tính thăm dò. Các prompt mô hình dùng tiếng Anh để phù hợp với checkpoint tiếng Anh. Mô hình không thấy nhãn, thứ hạng hay tên phương pháp tạo ứng viên. Nếu xác suất cao nhất dưới 0,70 hoặc mô hình chọn thiếu thông tin, hệ thống chuyển cho người đọc. Đây là ngưỡng thử, chưa hiệu chỉnh bằng nhãn người thật.

Đã chạy **Laya Typed-Decisions** trên đủ 43 cặp. Năm lựa chọn khớp nhãn trợ lý ở 12/43, so với 15/43 nếu luôn chọn lớp đông nhất. Khi chỉ hỏi lưu/bỏ qua, mô hình khớp 19/41 cặp có nhãn 0–3, so với baseline 21/41. Quy tắc chuyển cho người đọc kích hoạt tương ứng 43/43 và 30/43 lần. Không dùng hai kết quả này để công bố độ chính xác trên người dùng hoặc để tự động ẩn paper.

Script chạy Laya hoàn toàn ở local. Các phiếu người chấm luôn ở local. [Laya model card](https://huggingface.co/convaiinnovations/laya-typed-decisions)

```bash
# Môi trường ML riêng: Laya 0.3.0, transformers 4.57.6, tokenizers 0.22.2, torch 2.10.0.
python scripts/reading_decisions.py --question graded --output /tmp/reading-laya.json
python scripts/reading_decisions.py --question save --output /tmp/reading-laya-save.json
```

Weights Laya dùng revision 1a793eb568e6718f15941d08f85432581df534e3. Hai file laya_results.json và laya_save_results.json chứa xác suất từng lựa chọn. Cột confidence do mô hình trả về, không phải độ chính xác đo thực tế. Bước tiếp theo là chấm lại bằng nhãn người thật.

Thử nghiệm tiếp theo đã hiệu chỉnh quyết định trên nhãn chuyên gia CSFCube và kiểm tra tác dụng của từng tín hiệu. [Kết quả, trọng số và cách chạy](../decision-study/README.md).
