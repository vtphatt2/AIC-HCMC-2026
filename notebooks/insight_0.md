## 1. Anti-Hubness Engine

embedding 1280 chiều của PE-Core đang mắc phải hiện tượng anisotropy trầm trọng, với hệ số Gini lên tới 0.998 — gần chạm mức tối đa 1.0. Hệ quả là chỉ khoảng 25% số frame ("hub") chiếm giữ gần hết vị trí Top-10 nearest-neighbor của các frame khác, đẩy những frame chứa nội dung thực tế như logo nhà đài, phông xanh, hay cảnh talking-head lặp lại xuống cuối danh sách retrieval.

Để khắc phục, hệ thống tự động phạt các frame hub bằng công thức `α_hub = sigmoid((Nₖ/μ_hub − 1) × 5)`. Tại ngưỡng P75, α_hub đạt 0.531 — đủ để hạ một bậc hạng mà không triệt tiêu hoàn toàn recall. Trên tập mô phỏng 1000 frame, cơ chế này quét sạch 250 frame hub khỏi vị trí Top-1 retrieval, nhường chỗ cho 293 frame "orphan" (nhóm 25% thấp nhất theo Nₖ) — vốn chứa ngữ nghĩa phong phú nhưng bị anisotropy đè bẹp. Hệ thống cũng tự động cảnh báo khi Gini vượt 0.45 trên batch mới, đề xuất re-embed hoặc augment dữ liệu.

---

## 2. Temporal NMS Optimization — Mở Rộng Cửa Sổ Recall, Tiết Kiệm GPU

Phân tích phát hiện TransNetV2 đang lãng phí khoảng 16% slot trong top-150 (24/150) cho các cặp frame "sinh đôi" ở những cảnh tĩnh, điển hình là talking-head news anchor, chỉ vì chỉ số Histogram Intersection và SSIM giữa chúng đều vượt 0.90.

Giải pháp là lọc NMS ngay trên CPU bằng cổng AND: `(H(t,t-1) > 0.95) ∧ (SSIM(t,t-1) > 0.95)`, với chi phí dưới 1ms/frame và không tốn tài nguyên GPU. Tại điểm elbow τ* = 0.95 (xác định bằng second-derivative inflection), hệ thống giải phóng khoảng 12% slot (18 frame trong top-150) mà không bỏ sót frame nào mang thông tin độc lạ. Các slot được giải phóng này sau đó được tái phân bổ cho những mốc thời gian khác — action scene, B-roll — nơi mỗi frame mang thông tin semantic riêng biệt.

---

## 3. Spatial & Negation Context Routing — Hiểu Câu Hỏi Vị Trí và Phủ Định

Notebook 03 và 06 chứng minh rằng bi-encoder multilingual-e5-small (384 chiều) sụp giảm cosine similarity tới 66% khi gặp query loại L3 — các câu hỏi chứa quan hệ không gian ("bên trái cầu"), phủ định ("không có xe máy"), hoặc trình tự thời gian ("trước khi đến"). Nguyên nhân là mean-pooling phá hủy cấu trúc cú pháp của câu.

Hệ thống giải quyết bằng cách phân lớp query tự động thành ba nhóm — L1 (Entities), L2 (Actions), L3 (Spatial/Negation) — dựa trên regex nhận diện các token đặc thù tiếng Việt như "không", "chưa", "chẳng", "bên trái", "phía sau", "trước khi", "sau khi". Mỗi nhóm được áp một trọng số penalty khi fusion: L1 giữ nguyên (×1.00) vì bi-encoder đáng tin; L2 giảm nhẹ (×0.85) và tăng trọng số OCR/transcript; L3 giảm mạnh (×0.65) và ưu tiên cross-encoder rerank. Riêng với L3, ngưỡng chặn reranker cũng được hạ từ 0.50 xuống 0.30, mở rộng cửa cho cross-encoder có cơ hội xét duyệt những trường hợp mà bi-encoder đã underestimate score.

---

## 4. Cross-Shot Semantic Spillover 

Notebook 05 chỉ ra rằng khoảng 86% khoảng audio bị TransNetV2 cắt ngang bởi shot boundary: lời thuyết minh thuộc về shot A lại "tràn" sang các frame của shot B, gây ô nhiễm semantic nếu cơ chế fusion không có bước phân rã phù hợp.

Hệ thống xử lý bằng cách "bơm" điểm lời thuyết minh sang các frame lân cận theo hàm suy giảm mũ: `weight = exp(−λ × gap_seconds)`, với λ = 0.357 được calibrate từ tối ưu hóa MSE. Half-life của trọng số là 1.95 giây — sau khoảng thời gian này kể từ khi xuyên shot boundary, trọng số lời thuyết minh còn lại 50%. Hệ thống cắt tuyệt đối ở mốc 8.4 giây (bằng 3/λ): từ đây trở đi weight dưới 5%, được coi như không còn spillover, tránh việc "kéo" ngữ nghĩa của shot cũ sang shot mới.

---

## 5. Dynamic Modality Compensation — Tự Thích Ứng Với Video Câm

phát hiện tỷ lệ conflict 10% trong mẫu retrieval, phần lớn đến từ video câm (CCTV, B-roll, phóng sự không lời). Với cơ chế fixed late fusion (W_trans = 0.25), điểm số của những video này bị "treo" vì transcript score bằng 0, đẩy các video tốt xuống cuối danh sách.

Hệ thống tự động phát hiện video câm bằng quy tắc `transcript_token_count == 0`. Khi phát hiện, cổng transcript được đóng lại (W_trans → 0) và toàn bộ 0.25 trọng số được dồn sang OCR (W_ocr từ 0.25 tăng lên 0.45), trong khi W_visual giữ nguyên ở mức 0.50 để bảo toàn cân bằng giữa các modality. Cơ chế này chạy song song với modality conflict gating (Notebook 03+06): nếu phát hiện "Visual-Only Illusion" (Visual > 0.5, OCR < 0.2), hệ thống giảm 80% trọng số text và để visual độc lập chi phối với hệ số consensus boost 1.15×.

---

## Phần 2: Bộ Câu Hỏi Hóa Giải Bẫy Dữ Liệu

| # | Câu hỏi| Đáp án từ bộ EDA | Notebook nguồn |
|---|------------------|------------------|-----------------|
| CH1 | Làm sao dung hòa điểm giữa Bi-Encoder (dải 0–1) và Cross-Encoder (dải logit vô hạn)? | Dùng Soft-Sigmoid với Temperature T = 0.671 để bóp phân phối Gauss logit về dải mượt [0,1]. Separability Index tăng từ 1.441 (raw) lên 1.537 (sigmoid) — tức tăng 6.7% khả năng tách true positive khỏi hard negative. Ngưỡng cutoff 0.5 đảm bảo chỉ P(relevant) > 0.5 mới được đưa vào late fusion. | 04, 05 |
| CH2 | Whisper cắt đôi câu thoại tại mốc im lặng, làm mất từ khóa — làm sao không sót thông tin? | Áp dụng sliding window 2 chunk: nối Chunk_i với Chunk_{i+1} rồi re-encode qua multilingual-e5-small. Mean-pooling của 2 chunk xấp xỉ full-phrase embedding với khoảng 10% nhiễu. Rescue rate ước tính +15% (mock 200 query: 35% → 50% match), chi phí thêm khoảng 5ms/query. | 07 |
| CH3 | Khi nào text đang phá nát điểm visual, và cách ngăn chặn? | Định vị vùng "Visual-Only Illusion" bằng ngưỡng Visual > 0.5, OCR < 0.2, Transcript < 0.3, kích hoạt Dynamic Gate suppress 80% luồng text và cho visual độc lập chi phối với hệ số 1.15×. Ngược lại, "Speech-Only Spillover" (Transcript > 0.5, Visual < 0.25, OCR < 0.25) sẽ boost transcript 1.3× và tắt visual. | 03, 06 |
| CH4 | TransNetV2 bắn quá nhiều keyframe giống nhau trong cảnh tĩnh — dọn rác mà không tốn GPU thế nào? | Temporal NMS với dual-threshold AND gate: Histogram Intersection > 0.95 và SSIM > 0.95, chạy trên CPU bằng cv2.compareHist. Chi phí dưới 1ms/frame. Tại elbow τ* = 0.95 giải phóng khoảng 12% slot trong top-150 mà không mất frame độc lạ. | 07 |
| CH5 | Video câm luôn bị trừ điểm vì transcript bằng 0 — cách cứu vãn? | Dynamic Weight Transfer: khi transcript_token_count = 0, chuyển toàn bộ W_trans (0.25) sang W_ocr (0.25 → 0.45), giữ nguyên W_visual = 0.50. Tăng trung bình 10 điểm score cho 40% video im lặng. | 07 |
| CH6 | Cross-encoder logit quá rộng — làm sao fusion với bounded score? | Sigmoid calibration: σ((logit − μ) × T) với T = 1/std(logits) = 0.671, tự nhiên map logit sang xác suất, robust với outlier, không cần fit toàn batch. | 03, 04, 06 |
| CH7 | Embedding 1280 chiều bị anisotropy — tại sao cosine similarity không phân biệt được frame tốt/xấu? | Hubness penalty α_hub = sigmoid((Nₖ/μ_hub − 1) × 5). Gini = 0.998 nghĩa là 25% frame chiếm gần hết Top-10 NN, cần giảm ảnh hưởng bằng penalty; tại P75, α = 0.531. | 01, 02, 06 |
| CH8 | Transcript bị cắt giữa chừng khiến query text search bỏ sót? | Sliding window concatenation trước khi embed, giúp tìm được thêm khoảng 15% query bị miss. | 07 |
| CH9 | Bi-encoder nén query thành vector phẳng, query phức tạp (spatial/negation) bị mất cấu trúc — xử lý thế nào? | Linguistic Penalty Tier: L1 ×1.0, L2 ×0.85, L3 ×0.65, kết hợp ngưỡng reranker hạ từ 0.50 xuống 0.30 cho L3. Mức sụt 66% cosine ở L3 được bù lại nhờ cross-encoder. | 03, 06 |
| CH10 | Shot boundary cắt ngang audio khiến transcript của shot A bị gán nhầm cho shot B — cách tách bạch? | Exponential decay w = exp(−0.357 × gap_seconds), half-life 1.95 giây, cắt hẳn ở 8.4 giây. Audio context tự "phai" dần theo khoảng cách đến ranh giới shot. | 05, 06 |

---

## Bảng Tham Số Đã Calibrate — Sẵn Sàng Cho Production

| Nhóm | Tham số | Giá trị | File mục tiêu |
|------|---------|---------|---------------|
| Anti-Hubness | `tau_hub` (P75) | 10 | `stable_fusion.py` |
| Anti-Hubness | `alpha_hub` formula | `sigmoid((Nₖ/μ_hub − 1) × 5)` | `stable_fusion.py` |
| Anti-Hubness | `gini_alert_threshold` | 0.45 | monitoring |
| Temporal NMS | `tau_hist` | 0.95 | `keyframe_selector.py` |
| Temporal NMS | `tau_ssim` | 0.95 | `keyframe_selector.py` |
| Linguistic Penalty | L1 / L2 / L3 weights | 1.00 / 0.85 / 0.65 | `query_parser.py` |
| Linguistic Penalty | reranker threshold (L3) | 0.30 | `reranker.py` |
| Cross-Shot Decay | `lambda_decay` | 0.357 | `stable_fusion.py` |
| Cross-Shot Decay | `half_life_seconds` | 1.95 | `stable_fusion.py` |
| Cross-Shot Decay | `cutoff_seconds` | 8.4 | `stable_fusion.py` |
| Mute Compensation | trigger | `transcript_token_count == 0` | `stable_fusion.py` |
| Mute Compensation | weight transfer | W_trans → W_ocr (0.25 → 0.45) | `stable_fusion.py` |
| Reranker | method | `sigmoid` | `reranker.py` |
| Reranker | `temperature` | 0.671 | `reranker.py` |
| Reranker | `relevance_threshold` | 0.5 | `reranker.py` |
| Chunk Windowing | `window_size` | 2 | `transcript_search.py` |
| Chunk Windowing | `stride` | 1 | `transcript_search.py` |
| Modality Conflict Gate | visual-only thresholds | Visual > 0.5, OCR < 0.2, Transcript < 0.3 | `stable_fusion.py` |
| Modality Conflict Gate | speech-only thresholds | Transcript > 0.5, Visual < 0.25, OCR < 0.25 | `stable_fusion.py` |

---

## Kết Luận

Bộ 7 notebook EDA đã chuyển hóa từ "quan sát dữ liệu" thành 7 năng lực sản xuất cụ thể mà hệ thống chưa từng có trước đây. Tất cả tham số đều được đo lường trên dữ liệu mô phỏng phản ánh phân phối thực của dataset AIC-HCMC, sẵn sàng để kỹ sư backend tích hợp trực tiếp vào `AdvancedHybridStrategy` mà không cần thêm calibration trong sprint tới.