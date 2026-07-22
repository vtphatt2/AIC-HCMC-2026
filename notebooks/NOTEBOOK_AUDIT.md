# Audit nhanh thư mục `notebooks`

## Kết luận

Đánh giá của tôi: **có một số ý tưởng kỹ thuật đáng thử, nhưng gần như chưa có bằng chứng thực nghiệm đủ để gọi là “calibrated”, “benchmark” hay “production-ready”.**

- Trong 13 notebook EDA (`01`–`12` và `advanced_*`): **11 notebook dùng toàn bộ dữ liệu mock/random**, `11_*` là hỗn hợp nhưng phần kết luận chính vẫn dựa trên số tự đặt, và chỉ `09_*` thực sự đọc ảnh/embedding thật ở quy mô nhỏ.
- Seed cố định chỉ làm số random **lặp lại được**, không làm chúng trở thành quan sát thật.
- Phần lớn biểu đồ đo lại quan hệ đã được viết sẵn trong hàm sinh dữ liệu; vì vậy insight thường là **kết quả tất yếu của giả định đầu vào**, không phải khám phá.
- Các file tổng hợp (`EDA_SUMMARY_REPORT.md`, `insight*.md`, `EDA_HUBNESS_*.md`, `production_signoff_metrics.json`) chủ yếu lặp lại kết quả mock và nâng cấp cách gọi thành “đã đo”, “đã calibrate”, “production-ready”. Không nên dùng chúng làm căn cứ thiết kế.

Chấm nhanh: **ý tưởng/sandbox 6/10; bằng chứng thực nghiệm 2/10; calibration production 0/10.**

## Cái gì thật và có giá trị

| Phần | Giá trị thật | Giới hạn |
|---|---|---|
| `09_real_data_keyframe_deepdive.ipynb` | Có đọc ảnh, transcript, scene và embedding thật. Trên 30 frame của `L01_V001`: 4/29 cặp bị đánh dấu trùng (13.8%). Trên 200 embedding đầu của cùng video: Gini hubness = **0.2749**. | Chỉ NMS trên 30 frame của một video và hubness trên 200 vector của một video; threshold `0.95/0.92` được đặt trước, không được calibrate. Dữ liệu nằm ở absolute path ngoài thư mục nên hiện không tái chạy độc lập được. Latency 2 ms/5 ms là số hard-code, không phải phép đo. |
| `vector_search_benchmark/` | Có cấu trúc benchmark tương đối đúng: 79 cảnh chọn từ 4,084 feature, 160 query tiếng Việt, ground truth nhất quán; không có query trùng hoàn toàn. Script benchmark có Recall@K/MRR và latency. | Không có file kết quả benchmark. Không có provenance/kiểm duyệt người gán nhãn cho query, và asset/feature gốc không nằm trong thư mục này. Đây là **bộ eval tiềm năng**, chưa phải bằng chứng hệ thống tốt. |
| `transcripts/`, `transcripts_sample/`, `metadata/`, các script scraper | Là transcript/metadata YouTube và code thu thập dữ liệu thật; hữu ích làm input. | Chưa được nối thành một thí nghiệm retrieval có ground truth. Có dữ liệu trùng giữa `transcripts` và `transcripts_sample`. |
| `TextEncoderExtract.ipynb` | Thực sự tải PE-Core, tách text encoder, export/quantize ONNX và thử một vector ảnh. | Là notebook tiện ích Colab, không phải EDA; chỉ test một mẫu, path hard-code, latency một lần chạy không phải benchmark. |

## Cái gì giả lập hoặc kết luận không được dữ liệu hỗ trợ

- `01_*`: similarity được định nghĩa trực tiếp bằng `0.85 - 0.4 * gap/max_gap + noise`, rồi notebook “khám phá” rằng gap nhỏ làm similarity cao. Đây là vòng tròn.
- `02_*`, `03_*`: chủ động tạo anisotropy, domain gap và phân phối L1 > L2 > L3, sau đó báo hubness/domain gap/query degradation như dữ kiện đo được.
- `04_*`: relevance, reranker logit và auxiliary score đều được sinh có điều kiện; K, temperature và fusion weight chỉ tối ưu trên chính thế giới giả đã tạo.
- `05_real_dataset_*`: tên có “real dataset” nhưng notebook tự ghi rõ dùng **mock distribution**. Genre density là số hard-code; typo degradation là công thức giả; λ=0.357 được fit để khớp target 0.5 do tác giả tự chọn.
- `06_*`, `07_*`, `08_*`, `10_*`, `advanced_*`: tiếp tục tái dùng các phân phối random/hard-code để chứng minh heuristic được thiết kế cho chính các phân phối đó.
- `11_*`: có đọc 559 khoảng cách timestamp transcript thật, nhưng điều này chỉ mô tả nhịp subtitle, không đo cross-shot retrieval. “Optimal chunk = 35 words” là do hàm Gaussian được viết sẵn với `peak = 35`; topic accuracy và weight cũng hard-code.
- `12_*`: “100 queries × 150 candidates” đều là mock; ground truth, modality score và reranker logit được sinh random/có điều kiện theo TP. Latency chỉ là thời gian chạy mảng NumPy proxy. Vì vậy `production_signoff_metrics.json` **không phải production sign-off**.
- Mâu thuẫn rõ nhất: `09_*` đo Gini thật là **0.2749**, nhưng chính phần diễn giải và các báo cáo sau đó viết thành “gần 0.998/0.99” và “anisotropy trầm trọng”. Đây là kết luận sai so với output của notebook.

## Thiết kế thực nghiệm đang thiếu

- Không có câu hỏi/hypothesis định lượng và tiêu chí pass/fail đặt trước.
- Không có tập train/calibration/test tách biệt; cùng dữ liệu được dùng để nghĩ luật, chỉnh tham số và tuyên bố thắng.
- Ground truth thường được sinh từ chính quy tắc đang được đánh giá; không có nhãn độc lập hoặc human review.
- Không có baseline end-to-end thật, ablation từng heuristic, nhiều seed/run, confidence interval, hay kiểm tra significance.
- Không đo retrieval trên corpus thật; nhiều con số latency/recall là proxy hoặc hard-code.

## Nên giữ gì và làm gì tiếp

**Giữ:** transcript/metadata + scraper, `vector_search_benchmark/`, `TextEncoderExtract.ipynb`, và phần load/đo thật của notebook 09. Xem các notebook còn lại như **design sketch/sandbox**, không phải kết quả nghiên cứu.

Thí nghiệm tối thiểu có ý nghĩa: khóa bộ 160 query hiện có, chạy baseline thật trên toàn corpus, sau đó bật từng heuristic một; báo `Recall@10/50/100`, MRR, p50/p95 latency và kết quả theo từng nhóm query. Dùng một tập khác để chỉnh threshold/weight. Chỉ những thay đổi thắng trên test cố định mới được đưa vào config.

**Phán quyết cuối:** không phải mọi thứ đều vô nghĩa, nhưng phần có giá trị nằm ở asset, tool và vài phép đo thật; gần như toàn bộ “insight định lượng” và “production config” hiện tại là giả thuyết được minh họa bằng dữ liệu tự dựng, chưa phải sự thật về hệ thống.
