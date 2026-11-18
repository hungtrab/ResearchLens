# Từ một câu đang đọc, tìm bài nào dùng cùng kỹ thuật?

**Nếu người đọc nêu tên kỹ thuật, việc giữ các cặp từ đứng cạnh nhau có ích.** Trên 172 câu SciERC, Reader tìm được ít nhất một abstract khác có cùng tên phương pháp trong ba kết quả đầu ở 69 câu. Kết hợp thứ hạng của Reader với BM25 dùng từ đơn và cặp từ liền nhau đạt 83 câu. Che tên kỹ thuật trong chính các câu ấy thì hai cách đạt 20 và 19 câu. Một ứng dụng có thể dùng kết quả đầu để thiết kế chức năng tìm theo tên kỹ thuật. Việc tìm ra hai phương pháp tương đương dù tác giả dùng cách diễn đạt khác vẫn là bài toán mở.

## Ba nghĩa của “đúng bài”

| Nguồn | Người gán nhãn xác nhận điều gì? | Bài toán xếp hạng của ResearchLens |
|---|---|---|
| SciERC | Vị trí và loại thực thể khoa học trong abstract, gồm tên phương pháp | Chọn câu chứa tên phương pháp đầu tiên cũng có trong abstract khác. Tìm abstract có nhãn cùng tên trong 499 bài còn lại. |
| ACL-ARC | Chức năng của một trích dẫn, trong đó *Uses* nghĩa là tác giả sử dụng công trình được trích | Bỏ dấu trích dẫn trong 26 câu test. Tìm lại đúng nguồn trong danh mục 1.294 bài, chỉ dùng tiêu đề ứng viên. Sau lọc theo năm và loại bài nguồn, mỗi câu còn 253–1.294 ứng viên. |
| CSFCube | Chuyên gia chấm mức tương đồng về phương pháp trên thang 0–3 | Dùng chín truy vấn đã khảo sát và 1.214 cặp. Một bài đạt nếu được chấm ít nhất 2. |

Nhãn thực thể SciERC cho phép kiểm tra trường hợp trùng tên. Các cặp truy vấn–đích ở đây được suy ra bằng luật, chưa được người đọc chấm riêng. Có 2.094 đoạn *Method* khớp nguyên văn trong 500 abstract. 172 abstract đạt điều kiện chọn truy vấn, bao phủ 81 tên sau chuẩn hóa. Có cả nhãn rộng như *parser* (11 truy vấn). Cùng một kỹ thuật có thể mang nhiều tên khác nhau. [Luan và cộng sự, 2018](https://aclanthology.org/D18-1360/)

Nhãn *Uses* cho biết tác giả trích dùng một công trình, gồm cả tài nguyên và bộ dữ liệu. Những công trình hữu ích khác có thể thiếu nhãn. Bản phân phối ACL-ARC trong SciCite có cùng giá trị tiêu đề bài nguồn ở cả 1.941 bản ghi của 186 bài. Thí nghiệm bỏ trường này và gộp các tên khác nhau của cùng bài qua ID hoặc tiêu đề chuẩn hóa. Trong tập test, không câu nào chứa nguyên tiêu đề đích sau khi bỏ dấu trích dẫn. [Dữ liệu gán nhãn của Jurgens và cộng sự](https://jurgens.people.si.umich.edu/citation-function/) · [Bản chia tập của AllenAI](https://github.com/allenai/scicite)

CSFCube chấm mức liên quan theo khía cạnh và cho phép một truy vấn có nhiều kết quả đúng. Chín truy vấn này đã được dùng để so sánh Reader và ASPIRE. Các kết quả mới trên chín truy vấn mang tính chẩn đoán. Hai bộ nhãn kia mới được đưa vào chuỗi thí nghiệm của dự án. Kho dữ liệu tiền huấn luyện của các encoder có thể chứa văn bản liên quan. [Mysore và cộng sự, 2021](https://arxiv.org/abs/2103.12906)

## Từng thành phần mang lại bao nhiêu?

Mỗi ô là số truy vấn có ít nhất một kết quả đạt trong ba vị trí đầu. Các phương án trên cùng một dòng nhận cùng văn bản và cùng ứng viên. Policy Reader đã khóa, gồm TF-IDF và MiniLM. BM25 dùng IDF trên catalog đánh giá, hoặc pool tương ứng đối với CSFCube. Fusion cộng nghịch đảo hạng của hai danh sách với hằng số 60. Cross-encoder chấm từng cặp truy vấn–ứng viên trong 20 kết quả đầu của fusion. [Thiết kế phép thử](method_retrieval_protocol.md) · [Model card của cross-encoder huấn luyện trên MS MARCO](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)

| Tác vụ | Reader | BM25 từ đơn | BM25 thêm cặp từ | Fusion | Cross-encoder |
|---|---:|---:|---:|---:|---:|
| SciERC, giữ tên (172) | 69 | 66 | 82 | 83 | 87 |
| SciERC, che tên (cùng 172) | 20 | 14 | 12 | 19 | 28 |
| ACL-ARC, tìm nguồn đã dùng (26) | 8 | 4 | 3 | 8 | 4 |
| CSFCube, cùng phương pháp (9) | 6 | 3 | 3 | 6 | 4 |

Trên chỉ tiêu chính, fusion giúp 23 câu từ trượt thành đạt và khiến 9 câu từ đạt thành trượt. Mức tăng ròng là **8,1 điểm phần trăm**, khoảng bootstrap ghép cặp 95% **[2,6; 14,0]** theo 81 tên kỹ thuật. Khi che tên, fusion giảm 37,2 điểm so với chính nó trên câu gốc, khoảng 95% [−45,1; −28,8]. Phép che giữ nguyên phần còn lại của câu, có thể để lại từ viết tắt hoặc mô tả và đôi khi khiến câu kém tự nhiên. Các so sánh ngoài chỉ tiêu chính mang tính thăm dò. [Kết quả từng câu và danh sách 20 ứng viên](method_retrieval.json)

**Lỗi nằm ở khâu nào?** Với tên kỹ thuật hiện rõ, 149/172 câu có ít nhất một bài đúng trong 20 ứng viên fusion. Trong đó 83 câu đã có bài đúng trong top 3 và 66 câu còn có thể cải thiện bằng cách đổi thứ tự. Có 23 câu thiếu bài đúng ở cả top 20. Khi che tên, chỉ 66/172 câu còn bài đúng trong top 20. Ở ACL-ARC, bước lấy ứng viên đặt trần ở 14/26 câu cho mọi mô hình chỉ xếp lại top 20. Sáu câu có thể cải thiện so với 8 câu fusion đang tìm trúng. Một mô hình quyết định hoàn hảo ở bước cuối cũng sẽ bỏ lỡ 12 nguồn trích trên bộ này.

**Ví dụ.** Câu trong abstract CVPR 2016 về mô tả ảnh GIF liệt kê *statistical machine translation* cùng hai phương pháp khác. Xét đúng nhãn *statistical machine translation*, bài phù hợp đầu tiên ở hạng 15 của Reader, hạng 3 của fusion và hạng 66 của fusion khi che tên. Trên CSFCube, câu về phân tích phụ thuộc bằng đồ thị vô hướng (ID 1198964) có bài được chuyên gia chấp nhận ở hạng 4 của Reader, hạng 1 của fusion và hạng 15 sau khi dùng cross-encoder. Tính trên cả chín câu CSFCube, fusion vẫn tìm trúng ở 6/9 câu. [Thứ hạng từng truy vấn](method_retrieval.json)

## Cho decision model chọn lại từ cùng 20 bài

Thí nghiệm bổ sung dùng Laya Typed-Decisions và cùng 20 bài của fusion trên mỗi truy vấn ACL-ARC và CSFCube. Mô hình nhận câu đang đọc và văn bản ứng viên, trả về một điểm cho câu hỏi liệu bài có mô tả kỹ thuật, dữ liệu hoặc thí nghiệm cụ thể phù hợp hay không. Dùng điểm để xếp hạng, giữ thứ tự fusion khi bằng điểm. Có 700 cặp được chấm. Câu hỏi được cố định trước lượt chạy Laya, sau khi đã xem kết quả năm phương án đầu. Đây là checkpoint nguồn mở Laya. Dịch vụ Jev của TypeSafe chưa được đo. [Protocol bổ sung](method_decision_protocol.md) · [Laya](https://huggingface.co/convaiinnovations/laya-typed-decisions) · [Giao diện câu hỏi của TypeSafe](https://docs.typesafe.ai/primitives/choice)

| Ba kết quả đầu | Fusion | Cross-encoder | Laya |
|---|---:|---:|---:|
| ACL-ARC, số câu tìm đúng nguồn trên 26 | 8 | 4 | 6 |
| CSFCube, số câu tìm được bài hữu ích trên 9 | 6 | 4 | 5 |
| CSFCube, số vị trí có bài hữu ích trên 27 | 8 | 8 | 8 |

Laya tìm đúng nguồn trong 6/26 câu ACL-ARC. Chênh lệch Hit@3 so với fusion là −7,7 điểm phần trăm, khoảng bootstrap theo bài nguồn 95% [−22,2; +7,7]. Trên CSFCube, cả ba cách cho 8 bài hữu ích trong 27 vị trí. Số truy vấn được phục vụ lại khác nhau do các kết quả hữu ích có thể dồn vào cùng một danh sách. Các kết quả này ủng hộ giữ Reader hiện tại làm mặc định và ưu tiên lấy đủ ứng viên trước khi đầu tư thêm vào việc chấm từng cặp. [Điểm Laya của từng ứng viên](method_decision.json)

## Tái lập

Dùng môi trường ML mô tả trong README cùng thư mục và cài package của dự án vào môi trường đó. Lần chạy đầu cần mạng để lấy dữ liệu và trọng số ở các revision ghi trong code. Tệp kết quả ghi SHA-256 của đầu vào. Chạy từ thư mục gốc dự án.

```sh
uv pip install --python .venv-decision/bin/python -e .
PYTHONPATH=. .venv-decision/bin/python scripts/evaluate_method_retrieval.py --cache /tmp/rl-method-study --csfcube-cache /tmp/rl-csfcube-source
PYTHONPATH=. .venv-decision/bin/python scripts/evaluate_method_decision.py --cache /tmp/rl-method-study --csfcube-cache /tmp/rl-csfcube-source
uv run python scripts/plot_method_retrieval.py
```

Lần chạy dùng torch 2.10.0, transformers 4.57.6, Laya 0.3.0, scikit-learn 1.7.1, NumPy 2.5.2, SciPy 1.16.1 và NetworkX 3.5. Việc xếp lại bằng Laya và cross-encoder trong nghiên cứu này diễn ra offline.
