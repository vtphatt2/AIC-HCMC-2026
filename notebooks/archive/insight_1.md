## Khám Phá Dữ Liệu Đa Phương Thức Cho Truy Xuất Video

### 1. Mối quan hệ giữa keyframes và transcript.
- Mô phỏng similarity score dựa trên khoảng cách thời gian từ frame đến tâm transcript, thì xác định được ngưỡng phân loại Quality Match tại 0.5 và cửa sổ dung sai thời gian tối ưu là ba giây. Kết quả cho thấy 68% số frame khớp được với transcript trong cửa sổ này.

### 2. Không gian embedding 1280-dim của PE-Core và 384 chiều của E5
- Sử dụng t-SNE và TruncatedSVD, thì phát hiện được hiện tượng anisotropy trầm trọng tức là các vector embedding không phân bố đều mà tập trung thành hình nón (AI viết thế chứ tao ko biết hình nón ra sao). Tỷ số Domain Gap giữa visual và text dao động từ 1.17 đến 1.35, nghĩa là khoảng cách cross-modal lớn hơn đáng kể so với intra-modal, đòi hỏi cơ chế alignment đặc biệt.

### 3. Mức độ nghiêm trọng của anisotropy (vector embedding phân bố ko đều)
- Thông qua hệ số Gini đạt 0.998 — gần mức tối đa tuyệt đối (Gini được define trong file EDA_HUBNESS_ANISOTROPY_NMS.md) lover chiếm 4%.

### 4. Bài toán Calibrate điểm số từ BGE-Reranker
- . Do cross-encoder trả về raw logit không giới hạn, cần ánh xạ về khoảng 0 đến 1 để fusion với các bounded scores khác. Ba phương pháp được thử nghiệm: Min-Max scaling, Z-score Gaussian CDF, và Soft-Sigmoid. Soft-Sigmoid với nhiệt độ T bằng 0.671 đạt chỉ số Separability Index cao nhất là 1.537, vượt trội so với raw logit ở mức 1.441 và Gaussian CDF ở mức 1.451. Kích thước cửa sổ reranker tối ưu là K bằng 100, cân bằng giữa Recall tại 10 và latency.

### 5. Calibrate các siêu tham số cho production. 
- Hệ số suy hao hàm mũ lambda được tối ưu ở mức 0.357, cho half-life 1.95 giây và ngưỡng cắt tuyệt đối 8.4 giây. Ma trận trọng số theo thể loại được thiết lập: các thể loại nặng về OCR như Giáo dục và Pháp luật nhận multiplier 1.4 cho OCR, trong khi các thể loại nặng về transcript như Du lịch và Thể thao nhận multiplier 1.3 cho transcript. Kiểm tra nhiễu truy vấn cho thấy ở mức 40% nhiễu, vector score giảm 13.6% trong khi BM25 score giảm 19.6%, khẳng định OCR cần cơ chế fuzzy matching.

### 6. Phân tích trong dữ liệu thô ?
- Phát hiện chính: TransNetV2 tạo ra tới 16% slot dư thừa cho các frame gần như giống hệt nhau trong cảnh tĩnh. Cơ chế Temporal NMS với AND gate sử dụng ngưỡng Histogram Intersection 0.95 và SSIM 0.92 giúp giải phóng 12% slot trong cửa sổ Top-150. Whisper và VAD cắt câu thoại tại các điểm im lặng khiến 15% truy vấn bị bỏ sót, nhưng sliding window hai chunk liên tiếp giúp khôi phục toàn bộ số truy vấn này. Khoảng 40% video là video câm như CCTV và B-roll, bị phạt điểm oan trong cơ chế fixed fusion — việc chuyển trọng số từ transcript sang OCR khi transcript rỗng giúp phục hồi trung bình 0.10 điểm cho mỗi video.

### 7. Thiết kế engine tối ưu keyframe
- Tối ưu keyframe kết hợp hai giai đoạn: offline NMS tại thời điểm ingestion và online anti-hubness tại thời điểm retrieval. Lớp KeyframeOptimizationEngine được triển khai với phương thức apply_ingestion_nms để lọc frame trùng và apply_online_hub_suppression để phạt frame hub. Benchmark trên ba pipeline cho thấy pipeline kết hợp NMS và Anti-Hubness đạt Recall tại 10 cao hơn 12 đến 15 phần trăm so với naive, với tỷ lệ giữ slot đạt khoảng 70%.

### 8. Xác thực toàn bộ giả thuyết trên dữ liệu thật từ AIC2026_sample.
- Nạp ảnh thật kích thước 910 nhân 512 pixel RGB từ ba video nhóm L01, tính toán SSIM thật qua thư viện skimage, và nạp embedding 1280 chiều thật từ file npy. Hệ số Gini trên dữ liệu thật đạt khoảng 0.99, khẳng định anisotropy là hiện tượng có thật trong production. Histogram similarity trên dữ liệu thật thấp hơn dữ liệu mô phỏng, chỉ đạt 0.6 đến 0.7, do ảnh hưởng của camera noise, thay đổi ánh sáng, và subtitle.

### 9. Thực hiện bốn thí nghiệm so sánh giữa baseline với luật cố định và next-gen với heuristic thích ứng. 
- Trong thí nghiệm CV, NMS kết hợp hubness đẩy các frame hub đỏ xuống mạnh trong khi bảo vệ các frame xanh có nội dung độc đáo, giải phóng 20% slot. Trong thí nghiệm NLP, adaptive cutoff hạ ngưỡng từ 0.50 xuống 0.35 cho truy vấn L3, giải cứu thêm các truy vấn phức tạp bị loại oan. Trong thí nghiệm Procrustes, phép biến đổi xoay trực giao tối ưu giúp nâng cross-modal cosine thêm 0.078. Trong thí nghiệm Soft Gating, hàm sigmoid mượt thay thế hard if-else, phục hồi trung bình 0.18 điểm cho video câm.

### 10. Tinh chỉnh ba heuristic nâng cao. 
- Hàm suy hao Gaussian với độ lệch chuẩn 1.656 giây được chứng minh ưu việt hơn hàm mũ truyền thống vì có đỉnh tại tâm transcript và giữ trọng số cao trong vùng dung sai ba giây. Ma trận trọng số động theo chủ đề được calibrate: các chủ đề trừu tượng như Kinh tế và Đời sống nhận trọng số visual 0.35, OCR 0.35, transcript 0.30; trong khi các chủ đề cụ thể như Thể thao và Du lịch giữ trọng số visual 0.50, OCR 0.25, transcript 0.25. Kích thước chunk transcript tối ưu nằm trong khoảng 20 đến 50 từ, với điểm tối ưu tuyệt đối tại 35 từ cho chất lượng thông tin cao nhất.

### 11. Tích hợp toàn bộ hệ thống và thực hiện A/B test cuối cùng.
- Ba proxy engine được lắp ráp: KeyframeOptimizationEngine cho lọc frame, BGERerankerService cho calibrate điểm, và StableFusionEngine cho fusion thích ứng. Một trăm truy vấn được chạy qua toàn bộ pipeline với 150 candidate mỗi truy vấn, đo các chỉ số Recall tại K, Mean Reciprocal Rank, tỷ lệ giữ slot, và latency. NextGen vượt Baseline ở mọi chỉ số, đặc biệt với các chủ đề trừu tượng. Toàn bộ tham số và kết quả được xuất ra file JSON ký duyệt triển khai.

#### Conlusion: 
- Hệ thống đã được nâng cấp đáng kể qua 30 tham số calibrate. Hệ số Gini 0.998 dẫn đến thiết kế toàn bộ anti-hubness engine. Cross-modal domain gap 0.088 được khắc phục qua Procrustes alignment. Hàm suy hao Gaussian với sigma 1.656 giây bảo vệ vùng dung sai thời gian. Soft gating mượt loại bỏ score cliff cho video câm. Có thể xem xét triển khai lên production (maybe tôi đoán thế bro)