# Phân chia công việc — 4 người

Chia dựa trên `paper_draft/01`–`05` và **chuyên môn thật đo từ git history**,
không phải phỏng đoán.

---

## 0. Căn cứ: ai đang thật sự làm gì

`git shortlog` + phân bố theo thư mục:

| Người | Tổng commit | Vùng chính |
|---|---:|---|
| Le-Anh-Duy | 49 | local-backend 18, remote-server 17, frontend 15 — **full-stack/kiến trúc** |
| Thanh Đạt | 30 | `preprocess/` 24 — **pipeline tiền xử lý, gần như độc quyền** |
| Ho Nam (+ Ho-Nam Nguyen) | 35 | `notebooks/` 16 — **data/EDA/phân tích** |
| RachalHys | 17 | remote-server 11, frontend 8, local-backend 6 — **full-stack** |
| Nguyen Nam | 11 | remote-server 6, frontend 4, local-backend 4 — full-stack |

> **Lưu ý danh tính:** `Ho Nam` và `Ho-Nam Nguyen` gần như chắc chắn là cùng
> một người (hai git config). `Nguyen Nam` thì không rõ có trùng không — vùng
> làm việc khác hẳn (backend vs notebook). Nếu team thực tế khác bảng này,
> **đổi người theo thực tế**, giữ nguyên cấu trúc 4 vai.

---

## 1. Ba ràng buộc cứng phải tôn trọng

1. **Gate 0 chặn mọi thứ.** Đo redundancy rate thật (§3.2 của `05`) tốn 0 đồng
   annotation, ~1 buổi, và quyết định SeqDiv có đáng làm không. **Không ai được
   bắt đầu gán nhãn trước khi Gate 0 xong.**
2. **Gate A0 bắt buộc 2 người gán nhãn độc lập.** Đây là định nghĩa của
   inter-annotator agreement — một người tự gán rồi tự đo kappa là vô nghĩa.
   Nên vai P2 và P4 phải cùng gán trên cùng một mẫu, **không bàn với nhau
   trong lúc gán**.
3. **Cuộc thi vẫn đang chạy.** Không được dồn cả 4 người vào việc paper. Vai P4
   giữ nhịp thi đấu; P3 giữ pipeline chạy được.

---

## 2. Bốn vai

### P1 — Retrieval & Experiment Lead → **Le-Anh-Duy**

*Lý do: đang sở hữu `app/strategies/`, `base_strategy.py`, `_similarity_filter.py`
ở cả 2 backend; commit nhiều nhất ở lõi retrieval.*

| Ưu tiên | Việc | Nguồn |
|---|---|---|
| **P0** | **Gate 0** — chạy 50–100 query thật, log toàn bộ top-K, đo tỉ lệ trùng thật (hiện dao động 4%–28%, quá rộng). Ra quyết định go/no-go cho SeqDiv. | `05` §3.2 |
| **P0** | Sửa lỗi `S_time`: tách `S_shape` (gap normalize) và `S_scale` (tổng thời lượng), ablate riêng. | `05` §3.1 |
| P1 | Dựng benchmark harness: cùng base retriever, log raw run + timing ra JSONL. | `03` §3 |
| P1 | Implement baseline: no-filter, one-per-video, frame-level threshold, **MMR trên concatenated aligned-step embedding** (baseline khó nhất), DPP nếu kịp. | `03` §4 |
| P2 | Implement SeqDiv + budget-aware refill. | `01` |

**Không làm:** đừng viết method mới trước khi Gate 0 trả lời. Nếu redundancy
ổn định dưới ~10% thì SeqDiv không có đất diễn → pivot, đỡ mất hàng trăm giờ.

---

### P2 — Data & Annotation Lead → **Ho Nam**

*Lý do: sở hữu `notebooks/`, và là người viết `NOTEBOOK_AUDIT.md` — đúng loại
tư duy phản biện dữ liệu mà việc này cần.*

| Ưu tiên | Việc | Nguồn |
|---|---|---|
| **P0** | Viết **guideline gán nhãn equivalence cluster** thật cụ thể. Phải chốt quy tắc cứng cho ca biên: "<5s cùng video = cùng cluster", cắt góc máy khác thì sao, hai đối tượng khác trong cùng bản tin thì sao. | `05` §3.3 |
| **P0** | **Gate A0**: pilot 50 query, gán độc lập với P4, đo Cohen's kappa. `>0.7` → đi tiếp. `<0.5` → **sửa định nghĩa bài toán**, không gán thêm. | `05` §3.3 |
| P1 | Dựng schema + tooling: `queries.jsonl`, `qrels_chains.jsonl`. | `03` §3 |
| P1 | Xây test set đóng băng ≥100 query, **split theo video**, tách calibration set riêng. Không tái dùng 160 query cũ làm test (lệch 95% về L01). | `03` §2–3 |
| P2 | Thống kê: bootstrap CI, paired test cho so sánh method. | `03` §7 |

**Cảnh báo:** đây là vai dễ đốt thời gian nhất. Tuyệt đối không gán full 100+
query trước khi kappa pilot pass.

---

### P3 — Pipeline & System-Paper Evidence → **Thanh Đạt**

*Lý do: sở hữu gần như độc quyền `preprocess/` (24/30 commit); hiểu sâu nhất
phần tiền xử lý và chi phí hạ tầng.*

| Ưu tiên | Việc | Nguồn |
|---|---|---|
| **P0** | Đóng gói bằng chứng cho **system paper**: latency breakdown end-to-end trên phần cứng thường (đã có: encode ~1.2s / search ~1.1s / filter ~0.1s / tổng ~2.4s), + số archive-native frame access (~200s → 8.7s cho 20 request đồng thời). | `05` §4.2 |
| P1 | Viết lại 2 phát hiện thành kết quả có thể trích dẫn: (a) oversample 3× **đắt hơn** là tiết kiệm — ngược trực giác "HNSW chỉ tốn theo `ef`"; (b) cơ chế 1 Range request/frame qua MP4 sample-table index. | `05` §4.2 |
| P1 | Ablation keyframe budget (hướng dự phòng A): sweep 20/40/60/80/100% vector, đo Recall/nDCG vs index size. Luật hiện tại `<=3s→1, <=10s→3, else 5` chưa nhìn semantic. | `04` §A |
| P2 | Reproducibility: manifest, checksum, lockfile, script sinh bảng từ raw output. | `03` §9 |

**Ghi chú:** đây là vai **ít rủi ro nhất** — mọi thứ đều đo được, không phụ
thuộc annotation. Nếu SeqDiv sập ở Gate 0 hoặc A0, phần này vẫn còn nguyên giá
trị và là xương sống của paper 1.

---

### P4 — Interaction, Competition Ops & Annotator #2 → **RachalHys**

*Lý do: full-stack nghiêng frontend (8 commit frontend + 11 remote-server);
gần người dùng nhất.*

| Ưu tiên | Việc | Nguồn |
|---|---|---|
| **P0** | **Annotator #2** cho Gate A0 — gán độc lập cùng mẫu 50 query với P2. Không trao đổi trong lúc gán. | `05` §3.3 |
| **P0** | Instrument frontend để log **time-to-first-correct**: timestamp query, số card đã xem, lúc submit, đúng/sai. Đây là dữ liệu cho RQ5 và **không cần cluster annotation**. | `05` §3.4 |
| P1 | Giữ nhịp thi đấu AIC 2026. Kết quả thi là điều kiện bảo chứng cho system paper — không có nó thì claim "nền tảng tốt" không chứng minh được. | `05` §4.1 |
| P2 | Viết phần UI của system paper, tập trung đúng một thứ nêu tên được: **slider duplicate-threshold + `config_overrides` tune ngay lúc query** = người dùng tự chỉnh retrieval policy tại thời điểm truy vấn. | `05` §4.3 |

**Không viết:** "UI của chúng tôi thân thiện và sáng tạo" — không phải
contribution nếu không có user study. Chỉ mô tả cơ chế cụ thể.

---

## 3. Thứ tự thực hiện

```text
TUẦN 1  ├─ P1: Gate 0 (redundancy thật)          ← CHẶN TẤT CẢ
        ├─ P2: viết guideline cluster
        ├─ P3: gom số cho system paper            ← chạy song song, không bị chặn
        └─ P4: instrument logging + thi đấu

        ▼ Gate 0 pass? ──── không ──→ pivot sang hướng A (keyframe budget), P3 dẫn

TUẦN 2  ├─ P2 + P4: pilot 50 query, đo kappa      ← Gate A0
        ├─ P1: sửa S_time, dựng harness + baseline
        └─ P3: ablation keyframe budget

        ▼ kappa > 0.7? ──── không ──→ sửa định nghĩa cluster, KHÔNG gán tiếp

TUẦN 3+ ├─ P2: gán full test set (đóng băng)
        ├─ P1: SeqDiv + budget-aware refill
        ├─ P3: viết paper 1 (system/resource)     ← nộp trước, dựa trên việc đã xong
        └─ P4: log thi đấu → dữ liệu RQ5
```

**Hai điểm thoát (quan trọng):**
- Gate 0 fail → SeqDiv dừng, cả nhóm dồn vào paper 1 + hướng keyframe budget.
  Mất đúng 1 buổi, không mất gì thêm.
- Kappa fail → dừng gán nhãn, sửa định nghĩa. Paper 1 vẫn đi tiếp bình thường.

---

## 4. Nguyên tắc chung

1. **Paper 1 (system/resource) không phụ thuộc bất kỳ gate nào.** P3 cứ chạy.
   Đây là cách de-risk: lấy được một paper từ công đã làm xong, trong khi
   SeqDiv còn đang chín.
2. **Không ai tune tham số trên test set.** Threshold, `lambda`, `tau`, event
   weight chỉ được chạm trên calibration split.
3. **Lưu raw run + timing JSONL, không chỉ lưu bảng tổng hợp.**
4. **Không dùng số từ notebook mock** (`EDA_SUMMARY_REPORT.md`,
   `production_signoff_metrics.json`, ...) làm bằng chứng — xem
   `notebooks/NOTEBOOK_AUDIT.md`.
5. Ghi lại seed, commit hash, index config, phần cứng cho mỗi lần chạy.
