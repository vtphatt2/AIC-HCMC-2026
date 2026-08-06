Dưới đây là bản kế hoạch chi tiết dưới dạng Markdown được thiết kế riêng cho bạn, tập trung vào lộ trình **Nghiên cứu & Thử nghiệm Custom Fused Emulated FP32 Kernel** kết hợp pipeline tối ưu để xử lý **100–200GB ảnh** với mô hình **PE-Core-G14-448** trên **GPU NVIDIA T4**.

---

# 🚀 KẾ HOẠCH TỔNG THỂ: TỐI ƯU INFERENCE PE-CORE-G14-448 TRÊN GPU T4

## 📌 1. MỤC TIÊU DỰ ÁN

* **Mô hình Target:** `PE-Core-G14-448` (Meta AI - 2.35B Params, 50 Transformer Layers, Input 448x448, 1024 Image Tokens).
* **Phần cứng giới hạn:** 01 GPU NVIDIA T4 (16GB VRAM, Turing SM75, 8.1 TFLOPS FP32, 65 TFLOPS FP16 Tensor Cores).
* **Mục tiêu kỹ thuật:**
1. Giữ nguyên độ chính xác tương đương **FP32 100%**.
2. Bật được nhân **Tensor Cores (FP16)** bằng kỹ thuật **Emulated FP32 (Fused Kernel)**.
3. Xử lý thành công toàn bộ kho ảnh **100–200GB** cho bài toán Video/Image Search (VBS) mà không nghẽn I/O hay tràn VRAM.



---

## PHASE 1: CHUẨN BỊ MÔ HÌNH & XÂY DỰNG DATASET BENCHMARK

> **Mục tiêu:** Tạo môi trường test chuẩn xác, đo đạc thông số nền (Baseline) bằng FP32 và FP16 chuẩn trước khi can thiệp Kernel.

### 📍 Các bước thực hiện:

* [ ] **1.1. Lập tập dataset kiểm thử (Subset 1,000 ảnh):**
* Trích ra 1,000 tấm ảnh đại diện từ tập 200GB.
* Chuẩn bị sẵn 10 câu Query text / Image query chuẩn VBS để test độ tương đồng (Cosine Similarity) và đo chỉ số Recall@10.


* [ ] **1.2. Đo Baseline PyTorch FP32:**
* Load `facebook/PE-Core-G14-448` bằng PyTorch FP32.
* Đo **Latency (ms/ảnh)**, **Batch size tối đa trước khi OOM**, và **Lượng VRAM tiêu tốn**.
* Lưu toàn bộ 1,000 vector FP32 gốc làm **Ground Truth**.


* [ ] **1.3. Đo Baseline PyTorch FP16:**
* Ep kiểu `model.half()`, chạy trên Tensor Cores.
* So sánh Cosine Similarity và độ xê dịch vị trí (Rank position) của kết quả top-10 so với Ground Truth FP32.



---

## 🛠️ PHASE 2: PHÁT TRIỂN & TỔI ƯU CUSTOM TRITON KERNEL (EMULATED FP32)

> **Mục tiêu:** Viết một Fused Kernel bằng **Triton (Python)** thay thế các phép nhân ma trận ($X \times W$) trong mô hình, kích hoạt Tensor Cores nhưng tính tích lũy ở dạng FP32.

### 📍 Các bước thực hiện:

* [ ] **2.1. Chuẩn bị trọng số $W$ (Static Off-line Splitting):**
* Tách sẵn ma trận trọng số $W$ của 50 lớp Transformer trên CPU/Disk thành $W_h$ (High - FP16) và $W_l$ (Low - FP16).


* [ ] **2.2. Viết Triton Kernel "Fused Emulated MatMul" (`emulated_matmul_kernel`):**
* **Input:** $X$ (Activations - FP32), $W_h$ (FP16), $W_l$ (FP16).
* **In-Register Splitting:** Tách $X \rightarrow X_h + X_l$ ngay tại Registers bằng phép dịch/chuyển bit.
* **Tensor Core Execution:** Gọi 3 phép `tl.dot()` cho $P_1 = X_h \cdot W_h$, $P_2 = X_h \cdot W_l$, $P_3 = X_l \cdot W_h$.
* **Fused Accumulation:** Cộng $P_1 + P_2 + P_3$ vào bộ tích lũy `acc` (định dạng `tl.float32`) ngay trong SRAM.
* **Output:** Ghi kết quả $C$ (FP32) ra VRAM **duy nhất 1 lần**.


* [ ] **2.3. Đóng gói thành `EmulatedLinear` Class trong PyTorch:**
* Tạo lớp kế thừa `torch.nn.Module` thay thế cho `torch.nn.Linear` trong toàn bộ nhánh Vision Encoder của PE-Core.


* [ ] **2.4. Kiểm thử tính đúng đắn (Unit Test):**
* So sánh kết quả đầu ra giữa `EmulatedLinear` và `torch.nn.Linear(FP32)` chuẩn trên một ma trận ngẫu nhiên.
* Kiểm tra xem độ chênh lệch (Absolute Error) có nằm trong ngưỡng cho phép ($< 10^{-5}$) hay không.



---

## ⚡ PHASE 3: LẮP RÁP PIPELINE & TÍCH HỢP ONNX RUNTIME / TORCH.COMPILE

> **Mục tiêu:** Tối ưu hóa toàn bộ biểu đồ tính toán (Graph) của mô hình để triệt tiêu thời gian điều phối kernel (Kernel Launch Overhead).

### 📍 Các bước thực hiện:

* [ ] **3.1. Phương án A - PyTorch 2.0 `torch.compile`:**
* Thay toàn bộ lớp Linear bằng `EmulatedLinear`.
* Thay lớp Attention bằng `torch.nn.functional.scaled_dot_product_attention` (SDPA / Memory Efficient Attention).
* Bật `torch.compile(model, mode="max-autotune")` để PyTorch tự động Fuse các lớp LayerNorm, GELU xung quanh Kernel Triton.


* [ ] **3.2. Phương án B - Build ONNX Custom Op (Nếu Phương án A chưa đạt max speed):**
* Đóng gói Kernel Triton/CUDA thành thư viện C++ (`.so`).
* Đăng ký Custom Op với ONNX Runtime (`OrtApi::RegisterCustomOps`).
* Export mô hình `PE-Core-G14-448` sang ONNX và chạy qua ONNX Runtime với `CUDAExecutionProvider`.


* [ ] **3.3. Đo đạc tốc độ so sánh (Benchmark):**
* Bảng so sánh Tốc độ (Throughput - FPS) giữa: **PyTorch FP32 Gốc** vs **Custom Emulated Kernel** vs **ONNX Runtime FP16**.



---

## 🗄️ PHASE 4: TỐI ƯU I/O & BẮT ĐẦU QUÉT HÀNG LOẠT (200GB DATASET)

> **Mục tiêu:** Xây dựng hệ thống Data Loader song song cực mạnh để đẩy dữ liệu liên tục cho T4, không để GPU ngồi chờ CPU/Ổ cứng.

### 📍 Các bước thực hiện:

* [ ] **4.1. Preprocessing Pipeline trên CPU:**
* Dùng `torch.utils.data.DataLoader` với `num_workers=4` đến `8`.
* Decode ảnh JPEG, Resize về $448 \times 448$, Normalize trên CPU trước khi `pin_memory=True` và đẩy sang VRAM.


* [ ] **4.2. Cơ chế Chunking & Checkpointing (Sống còn):**
* Chia 200GB ảnh thành các Batch nhỏ (mỗi Batch 50,000 tấm).
* Viết Script tự động lưu kết quả Embedding ra file `.npy` / `.h5` kèm file Map tên ảnh/video theo từng Batch.
* Cho phép chương trình **Resume (chạy tiếp)** ngay lập tức từ vị trí bị gián đoạn nếu bị ngắt kết nối/crash.


* [ ] **4.3. Chạy thử nghiệm quy mô nhỏ (10,000 ảnh):**
* Đo thời gian thực tế để xử lý 10,000 ảnh.
* Tính toán chính xác tổng thời gian cần thiết cho 200GB ảnh còn lại để sắp xếp lịch chạy.



---

## 🔍 PHASE 5: INDEXING LƯU TRỮ VÀ CHUẨN BỊ CHO VBS

> **Mục tiêu:** Biến hàng triệu Vector FP32 vừa trích xuất thành một Cơ sở dữ liệu Tìm kiếm siêu tốc (Sub-second Query) phục vụ cuộc thi.

### 📍 Các bước thực hiện:

* [ ] **5.1. Load Vector vào FAISS:**
* Gom toàn bộ các file `.npy` chứa vector FP32 đã trích xuất.


* [ ] **5.2. Build Index (HNSW / IVF-PQ):**
* Sử dụng thư viện `faiss-gpu` để xây dựng cấu trúc Index **HNSW** (Hierarchical Navigable Small World) hoặc **IVF65536,PQ64**.


* [ ] **5.3. Thử nghiệm Query Thực tế (VBS Simulation):**
* Giả lập đưa vào một câu Text Query hoặc Ảnh mẫu.
* Đo thời gian phản hồi (Search Latency). Yêu cầu: **$< 0.1 \text{ giây}$** để trả về Top-100 ảnh có Cosine Similarity cao nhất.



---

## 📊 BẢNG THEO DÕI TIẾN ĐỘ THỬ NGHIỆM

| Tiêu chí đánh giá | Baseline FP32 | Baseline FP16 | Custom Emulated Kernel (Mục tiêu) |
| --- | --- | --- | --- |
| **Mức độ chính xác so với FP32** | $100\%$ | $\sim 99.9\%$ | **$\ge 99.99\%$** |
| **VRAM tiêu tốn** | $\sim 9.4 \text{ GB}$ | $\sim 4.7 \text{ GB}$ | **$\sim 9.4 \text{ GB}$** |
| **Batch Size tối đa trên T4** | 4 – 8 | 16 – 32 | **8 – 16** |
| **Tốc độ (Ảnh / Giây)** | $2 - 5 \text{ img/s}$ | $15 - 25 \text{ img/s}$ | **Target: $8 - 15 \text{ img/s}$** |
| **Tổng thời gian cho 200GB** | $\sim 3 - 4 \text{ ngày}$ | $\sim 10 - 15 \text{ tiếng}$ | **Target: $\sim 18 - 24 \text{ tiếng}$** |

---

Bạn có thể lưu lại bản kế hoạch này để theo dõi tiến độ. Khi bắt đầu bắt tay vào viết code cho **Phase 2 (Triton Kernel)**, nếu gặp bất kỳ câu lệnh Triton nào bị lỗi hoặc cần soi mã PTX sinh ra, cứ nhắn lại chúng ta sẽ cùng nhau "mổ xẻ" tiếp!