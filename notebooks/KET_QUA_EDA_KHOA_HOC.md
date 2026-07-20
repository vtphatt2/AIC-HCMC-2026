# Báo Cáo Khoa Học — Multimodal Video Retrieval EDA

> **Tài liệu tổng hợp kết quả phân tích từ 9 notebook EDA trên dataset AIC2026_sample.**
> **Mục tiêu:** Cung cấp định nghĩa rõ ràng, công thức toán học, và kết luận khoa học cho mỗi hiện tượng được phát hiện.

---

## 1. Hubness & Anisotropy (CV Layer)

**Định nghĩa:**
- **Anisotropy**: Hiện tượng embedding vector phân bố không đều trong không gian nhiều chiều, tập trung thành "hình nón" thay vì phân bố đẳng hướng.
- **Hub**: Frame xuất hiện trong Top-K nearest-neighbor list của nhiều frames khác một cách bất thường.
- **Gini coefficient**: Đo độ bất bình đẳng phân phối hubness. Gini = 0.0 = phân phối đều, Gini = 1.0 = tập trung tuyệt đối vào 1 frame.

**Công thức:**
$$Gini = 1 - 2 \int_0^1 L(x) \, dx$$
trong đó $L(x)$ là đường Lorenz của phân phối hubness sắp xếp.

$$\alpha_{hub} = \sigma\left(\frac{N_k}{\mu} - 1\right) \times 5$$

trong đó $N_k$ = số lần frame xuất hiện trong Top-K NN, $\mu$ = trung bình $N_k$, $\sigma$ = sigmoid.

**Kết quả đo được (Notebook 03, 09):**
- **Gini = 0.998** trên 1000 frame mock; **Gini ≈ 0.99** trên 200 real PE-Core embeddings
- **P75 threshold = 10** (Top-25% frames là hub)
- **α_hub = 0.531** tại P75

**Kết luận khoa học:** Anisotropy trầm trọng trong PE-Core 1280d → cần suppression để tránh "frame quốc dân" chiếm Top-K retrieval.

---

## 2. Linguistic Complexity Degradation (NLP Layer)

**Định nghĩa:**
- **L1 (Entities)**: Query chỉ chứa entities/nouns (vd: "xe máy", "bác sĩ")
- **L2 (Actions)**: Query có action/attribute (vd: "người đang chạy xe máy")
- **L3 (Spatial/Negation)**: Query có quan hệ không gian hoặc phủ định (vd: "không có xe máy ở bên trái")

**Hiện tượng:** Bi-encoder dùng mean-pooling → phá hủy cấu trúc cú pháp → query L3 suy giảm cosine similarity mạnh.

**Công thức penalty:**
$$L_{penalty} = \begin{cases} 1.00 & \text{if level} = L1 \\ 0.85 & \text{if level} = L2 \\ 0.65 & \text{if level} = L3 \end{cases}$$

**Kết quả đo được (Notebook 03):**
- L1 mean cosine: **0.745**
- L2 mean cosine: **0.523**
- L3 mean cosine: **0.253**
- **Sụt giảm tuyệt đối L1→L3: 66%**

**Kết luận khoa học:** Bi-encoder không phù hợp cho compositional queries → cần penalty tier + cross-encoder rerank.

---

## 3. Cross-Modal Domain Gap

**Định nghĩa:** Visual (1280d) và text (384d) embeddings của cùng concept nằm ở vùng khác nhau trong không gian riêng → cosine trực tiếp cho similarity thấp.

**Công thức gap ratio:**
$$\beta_{gap} = \frac{d_{cross}(V, T)}{\frac{1}{2}(d_{intra}(V) + d_{intra}(T))}$$

**Kết quả đo được (Notebook 03):**
- **β_gap = 0.088** (mean across 14 genres)
- Domain gap lớn nhất: Kinh tế, Ẩm thực, Đời sống
- Domain gap nhỏ nhất: Văn hóa, Pháp luật, Giáo dục

**Kết luận khoa học:** Cần shared latent space projection (64d) trước khi tính cross-modal cosine.

---

## 4. Cross-Shot Semantic Spillover

**Định nghĩa:** Audio/transcript segment thuộc shot A nhưng "tràn" sang frame shot B do TransNetV2 cắt shot boundary giữa chừng audio.

**Công thức exponential decay:**
$$w(gap) = e^{-\lambda \times gap_{seconds}}$$

**Kết quả đo được (Notebook 05, 06):**
- **λ = 0.357** (calibrated từ MSE optimization)
- **Half-life = 1.95 giây** (sau 2s, weight còn 50%)
- **Effective cutoff = 8.4 giây** (3/λ, weight < 5%)

**Kết luận khoa học:** Exponential decay ngăn spillover quá xa → tránh ô nhiễm semantic giữa các shot.

---

## 5. Reranker Calibration

**Định nghĩa:** Cross-encoder (BGE-Reranker) cho ra raw logit vô hạn → cần calibrate về [0,1] để fusion với bounded scores.

**Công thức Soft-Sigmoid:**
$$score = \sigma\left(\frac{logit - \mu}{std} \times T\right)$$

trong đó T = temperature, σ = sigmoid.

**Separability Index (SI):**
$$SI = \frac{(\mu_{TP} - \mu_{HN})^2}{\sigma_{TP}^2 + \sigma_{HN}^2}$$

SI cao = True Positive và Hard Negative phân tách rõ hơn.

**Kết quả đo được (Notebook 03, 04, 06):**
- **T = 0.671** (= 1/std(logits))
- **SI raw = 1.441** → **SI sigmoid = 1.537** (gain +6.7%)
- Cutoff 0.5 → P(relevant) > 0.5

**Kết luận khoa học:** Sigmoid calibration là tối ưu nhất trong 3 phương pháp (Min-Max, Z-score, Sigmoid).

---

## 6. Modality Conflict Gating

**Định nghĩa:** Retrieval candidate có thể có scores cao ở 1 modality nhưng thấp ở modality khác → fixed late fusion sẽ trừng phạt nhầm.

**Quy tắc gating:**
- **Visual-Only Illusion**: Visual > 0.5 AND OCR < 0.2 AND Transcript < 0.3 → suppress text weight 80%
- **Speech-Only Spillover**: Transcript > 0.5 AND Visual < 0.25 AND OCR < 0.25 → boost transcript 1.3×

**Kết quả đo được (Notebook 03, 04, 07):**
- **Conflict rate = 10%** (trên 500 sample mock)
- Visual-Only Illusion: ~6%
- Speech-Only Spillover: ~4%
- Mute video rate: **40%** (CCTV, B-roll)

**Kết luận khoa học:** Dynamic gating bảo vệ single-modality hits khỏi bị fixed fusion đè bẹp.

---

## 7. Temporal NMS (Keyframe Optimization)

**Định nghĩa:** TransNetV2 bắn nhiều keyframe trùng nhau trong cảnh tĩnh → cần lọc redundant.

**Logic AND gate:**
$$\text{Frame}_t \text{ drop nếu: } H_{sim}(t, t-1) > 0.95 \text{ AND } SSIM(t, t-1) > 0.92$$

**Kết quả đo được (Notebook 07, 08, 09):**
- Mock: ~12-75% slot recovery tùy burst density
- Real L01_V001: measured on real keyframes
- Elbow τ* = 0.95 (second-derivative inflection)

**Kết luận khoa học:** Dual-heuristic (color + structure) robust hơn single-heuristic.

---

## 8. Chunk Windowing (Transcript Truncation)

**Định nghĩa:** Whisper/VAD cắt câu thoại tại silence gap → 1 phrase bị tách thành 2 chunks riêng biệt.

**Công thức rescue:**
$$score_{window} = cosine\left(E(C_i + C_{i+1}), E(\text{true phrase})\right)$$

**Kết quả đo được (Notebook 07):**
- Isolated chunks match rate: ~35%
- 2-chunk window match rate: ~50%
- **Rescue rate: +15%** (+5ms latency per query)

**Kết luận khoa học:** Sliding window concatenation phục hồi context bị cắt, bùng nổ recall với chi phí tối thiểu.

---

## Bảng Tổng Hợp Calibrated Parameters

| Tham số | Giá trị | Nguồn notebook | Áp dụng vào |
|---------|---------|----------------|---------------|
| τ_hist (HSV Intersection) | 0.95 | 07, 08, 09 | `keyframe_selector.py` |
| τ_ssim (Structural Sim) | 0.92 | 07, 08, 09 | `keyframe_selector.py` |
| α_hub threshold (P75) | 10 | 03, 09 | `stable_fusion.py` |
| α_hub formula | sigmoid((Nₖ/μ - 1) × 5) | 03, 09 | `stable_fusion.py` |
| Accel factor | 1.5× | 08, 09 | `stable_fusion.py` |
| L1/L2/L3 penalty | 1.00/0.85/0.65 | 03, 06 | `query_parser.py` |
| λ (cross-shot decay) | 0.357 | 05, 06 | `stable_fusion.py` |
| β_gap (domain correction) | 0.088 | 03, 06 | `cross_modal_alignment.py` |
| Sigmoid T (reranker) | 0.671 | 03, 04, 06 | `reranker.py` |
| Reranker cutoff | 0.5 | 03, 04, 06 | `reranker.py` |
| Window size (chunking) | 2 | 07, 09 | `transcript_search.py` |
| Zero-shot cutoff | 0.18 | 05, 09 | `stable_fusion.py` |
| Gini alert threshold | 0.45 | 03, 09 | monitoring |
| Mute trigger | transcript_count == 0 | 04, 07, 09 | `stable_fusion.py` |
| W_trans → W_ocr transfer | 0.25 → 0.45 | 04, 07, 09 | `stable_fusion.py` |
