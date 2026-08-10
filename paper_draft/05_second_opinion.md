# Second opinion — review draft SeqDiv + đề xuất bổ sung

Ngày: 2026-08-10. Review độc lập bản draft `01`–`04` sau khi kiểm chứng lại
bằng chứng trực tiếp trên repo và trên web.

**Kết luận ngắn:** draft hiện tại có chất lượng tốt và kỷ luật học thuật cao.
Không cần viết lại. Cần **sửa 1 lỗi công thức**, **thêm 1 gate đứng trước mọi
thứ**, và **cân nhắc tách thành 2 paper theo thứ tự thời gian** thay vì dồn
hết vào SeqDiv.

---

## 1. Đã kiểm chứng những gì

### 1.1 Citations — 5/5 spot-check đều là paper thật

Đây là rủi ro lớn nhất khi dùng LLM để draft paper (reference bịa). Đã fetch
trực tiếp từng cái, đối chiếu title + author:

| Citation | arXiv/DOI | Kết quả |
|---|---|---|
| DANTE | 2512.13169 | ✅ đúng title, đủ 6 tác giả |
| MERVIN | 2605.16120 | ✅ đúng; còn xác nhận được đã accepted SOICT 2025 |
| QSVideo | 2607.04559 | ✅ đúng title + 3 tác giả |
| CSES | 2608.00714 | ✅ đúng |
| KTV | 10.1609/aaai.v40i11.37862 | ✅ thật, AAAI-2026, pp. 9060–9068 |

Không phát hiện reference bịa. `references.bib` dùng được.

### 1.2 Claim về repo — chính xác từng con số

| Claim trong draft | Kiểm chứng |
|---|---|
| Keyframe rule `<=3s → 1`, `<=10s → 3`, else 5 | ✅ `keyframe_pipeline_global_v9_3/src/pipeline/phase1_transnet/decode.py:68` (`keyframe_count()`) |
| 192 competition query file, 9 TRAKE | ✅ đếm đúng 192 file; đúng 9 file `*-trake.txt` |
| 11/13 notebook dùng mock/random | ✅ trích từ `notebooks/NOTEBOOK_AUDIT.md` có thật (commit `fb5f96f`) |
| Các file strategy được tham chiếu | ✅ tồn tại đầy đủ |

### 1.3 Phần đáng khen nhất

Mục **"Không claim"** trong `01_seqdiv_proposal.md`. Chủ động từ chối claim
PE-Core/Milvus/TransNetV2/RRF/DP là mới, từ chối gọi oversampling 1.5× là
"adaptive retrieval", từ chối dùng số từ notebook mock. Đây là thứ hiếm và là
lý do chính khiến bản draft này đáng tin.

---

## 2. Đồng ý — giữ nguyên

1. **Novelty boundary** trong `02_related_work.md`: bảng "work nào chặn claim
   nào" là cách làm đúng. Đặc biệt việc tự chặn "MMR nhưng dùng video".
2. **Frozen base retriever** cho mọi diversification baseline — tránh trộn 2
   contribution vào 1 phép đo.
3. **Split theo video, không random theo frame.** Đúng; split theo frame sẽ rò
   rỉ thông tin giữa train/test vì frame cùng video cực giống nhau.
4. **Calibration set tách khỏi test set bị đóng băng.** Threshold, `lambda`,
   `tau`, event weights chỉ được tune trên calibration.
5. **Baseline list** (`03`, mục 4) đầy đủ và trung thực — có cả MMR trên
   concatenated aligned-step embedding, tức baseline khó nhất cho chính mình.
6. **Reproducibility checklist** và yêu cầu lưu raw run/timing JSONL.
7. **Phân biệt "paginated refill" vs "adaptive retrieval"** — mô tả đúng code
   hiện tại, không thổi phồng.
8. **Cảnh báo bộ 160 query lệch 95% về L01** — không dùng làm test set.

---

## 3. Cần sửa

### 3.1 LỖI: `S_time` mất thông tin thời lượng tuyệt đối

`01_seqdiv_proposal.md` định nghĩa:

```text
gap(C) = normalize([t2-t1, ..., tm-t(m-1)])
S_time = exp(-L1(gap(C), gap(D)) / tau)
```

Vì `gap()` đã normalize nên **thời lượng tuyệt đối bị vứt đi**. Hệ quả: một
chain trải 3 giây và một chain trải 3 phút, nếu tỉ lệ khoảng cách giữa các
event giống nhau, sẽ ra `S_time = 1.0` — bị coi là trùng khít.

Với bài toán này gần như chắc chắn sai: 3 giây và 3 phút là hai loại sự kiện
khác hẳn nhau.

**Sửa:** giữ cả *shape* lẫn *scale*. Ví dụ tách thành hai thành phần
(`S_shape` trên gap đã normalize, `S_scale` trên tổng thời lượng chain), rồi
ablate riêng để biết cái nào thật sự đóng góp.

### 3.2 THIẾU: Gate 0 — đo redundancy rate thật, trước tất cả

Draft có Gate A (qrels) / B (signal) / C (efficiency), nhưng thiếu câu hỏi
đứng trước mọi thứ:

> Top-K hiện tại có **thật sự** đủ trùng lặp để đáng diversify không?

Số đo được từ log hệ thống thật (threshold 0.98, strategy `raw_visual`):

| Candidates | Kept | Tỉ lệ bị lọc |
|---:|---:|---:|
| 100 | 93 | 7% |
| 100 | 94 | 6% |
| 150 | 139 | 7% |
| 200 | 192 | 4% |
| 200 | 165 | 17% |
| 300 | 216 | **28%** |

Dao động 4%–28% — quá rộng để kết luận gì. **Chưa ai đo redundancy rate thật
trên một bộ query thật.**

Đây là thí nghiệm **tốn 0 đồng annotation**, làm xong trong một buổi, và nó
quyết định paper này có đáng viết hay không. Đặt nó **sau** Gate A (vốn tốn
hàng trăm giờ gán nhãn) là ngược thứ tự rủi ro.

**Việc cần làm:** chạy ~50–100 query thật, log toàn bộ top-K, đo bằng mắt +
bằng cosine xem tỉ lệ cặp trùng thật là bao nhiêu. Nếu redundancy thấp
(<10% ổn định), SeqDiv không có đất diễn → pivot sớm, chưa mất gì.

### 3.3 RỦI RO: equivalence cluster là chỗ dễ sập nhất

`equivalence_cluster` (nhãn "mấy kết quả này tuy khác `frame_id` nhưng là cùng
một câu trả lời") là **bắt buộc** — không có nó thì không tính được
`alpha-nDCG`, tức không chứng minh được gì.

Nhưng nó cũng là nhãn chủ quan nhất. Ca rõ ràng thì dễ (3 frame cách nhau 1
giây = cùng cluster; khác video = khác cluster). Ca biên thì không:

- cùng đối tượng nhưng máy quay cắt sang góc khác sau 30 giây?
- cùng bản tin nhưng hai đối tượng khác nhau?
- cùng sự kiện nhưng một cảnh thấy bối cảnh, một cảnh không?

Nếu 2 annotator bất đồng ở tỉ lệ lớn các ca này → nhãn cluster là nhiễu →
`alpha-nDCG` tính trên đó cũng nhiễu → câu "SeqDiv tăng alpha-nDCG 3%" có thể
chỉ là sai số đo.

Trớ trêu: SeqDiv sinh ra để **tự động** phán đoán "hai chain có phải cùng một
đáp án không" — tức đang nhờ người làm thủ công, nhất quán, đúng cái phán đoán
mà ta lập luận là khó tới mức cần một contribution nghiên cứu.

**Sửa — thêm Gate A0 trước Gate A:**
1. Viết guideline gán nhãn thật cụ thể (chốt luôn quy tắc cứng, ví dụ
   "<5s trong cùng video = cùng cluster").
2. Pilot 2 annotator trên ~50 query.
3. Đo Cohen's kappa / Krippendorff's alpha.
4. `kappa > 0.7` → đầu tư tiếp. `kappa < 0.5` → **sửa định nghĩa bài toán**,
   đừng gán thêm 100 query nữa.

### 3.4 Đề nghị: đưa RQ5 lên trung tâm

`RQ5` (time-to-first-correct) đang xếp cuối, coi như user study nice-to-have.
Nên là **trục chính**, vì:

- Đó là metric cuộc thi thật sự tối ưu.
- Nó **không cần equivalence-cluster annotation** — ground truth là "user
  submit đúng chưa, mất bao lâu", cuộc thi cho miễn phí. Né sạch bottleneck 3.3.
- Gần như không ai trong literature diversification đo kiểu này, vì họ không có
  hệ thống interactive chạy thật với user chịu áp lực thời gian. Nhóm này có.

**Reframe câu hỏi trung tâm** từ "SeqDiv có tăng alpha-nDCG không" (đắt, dễ bị
cãi, incremental) sang:

> Dưới một deadline interactive cố định, hệ thống nên tiêu 500ms tiếp theo vào
> đâu — fetch thêm candidate, diversify cái đang có, hay decode thumbnail tốt
> hơn cho cái user đang nhìn?

Gần như toàn bộ literature diversification **giả định candidate là miễn phí**,
chỉ tính chi phí rerank. Hệ thống này đã đo được chi phí thật của từng nhánh
(xem §4.2). Đối tượng khoa học đáng giá ở đây là **chính sách phân bổ ngân sách
dưới deadline**, không phải similarity kernel — mà kernel thì như draft tự thừa
nhận, rất khó vượt MMR chuẩn bằng một tổ hợp tuyến tính 3 tham số.

---

## 4. Bổ sung: hướng paper thứ hai (system/resource)

`04_alternative_directions.md` hạ hướng systems xuống khá nhanh. Tôi nghĩ có
một phiên bản mạnh hơn mà draft chưa nắm: **không phải systems paper thuần, mà
là system/platform paper.**

### 4.1 Đây là category có thật

VBS (Video Browser Showdown) và MMM có truyền thống nhận **system paper**:
vitrivr, SOMHunter, diveXplore, VIRET. ACM MM còn có Open Source Software
track. Không phải thể loại tự nghĩ ra.

**Bằng chứng nhóm đã có sẵn trong git** (quan trọng nhất — không phải lời tự
khen):

- Commit `9c9d600` thêm 4 strategy DP temporal → chạm đúng **5 file, toàn bộ
  nằm trong `app/strategies/`**. Không sửa một dòng nào của `data_provider.py`,
  `main.py`, hay frontend.
- 3 contributor độc lập trong lịch sử strategy: `Le-Anh-Duy` (10 commit),
  `Nguyen Nam` (4), `RachalHys` (2).

Viết được thành claim cụ thể, kiểm chứng được: *"4 chiến lược retrieval mới
được ship bởi các contributor khác nhau, không sửa một dòng core nào."* Hầu hết
system paper chỉ nói "kiến trúc của chúng tôi mô-đun hoá" rồi thôi — không ai
đưa số.

**Điều kiện bắt buộc, không né được:** system paper dòng VBS đáng tin **vì có
thành tích thi đấu bảo chứng**. Nếu nhóm chưa thi hoặc kết quả kém, claim
"kiến trúc chúng tôi là nền tảng tốt" thành không thể chứng minh. Hướng này
gắn số phận vào kết quả AIC 2026.

### 4.2 Về tối ưu inference — nói thẳng

**Từng kỹ thuật riêng lẻ KHÔNG phải contribution.** Reviewer sẽ nói đúng câu
đó, và nếu viết thành "contribution" thì mất uy tín cả paper:

| Kỹ thuật | Thực chất |
|---|---|
| Warmup model lúc startup | Best practice, ai cũng làm |
| ONNX int8 quantization | Chuẩn |
| Piggyback vector vào kết quả search | "Denormalize để tránh N+1 query" — ý tưởng cũ của database |
| Oversample 1.5× | Hằng số tune tay |

**Nhưng bản thân phép đo thì có giá trị.** Literature interactive video
retrieval gần như không ai công bố breakdown latency end-to-end thành thật
trên phần cứng thường. Số đo được (i7, CPU, không GPU, warm):

| Giai đoạn | Thời gian |
|---|---:|
| Text encode (PE-Core ONNX) | ~1.15–1.24s |
| Vector search (Milvus Lite, top_k=150) | ~1.05–1.15s |
| Frame embeddings | ~0ms (sau khi piggyback; trước đó 0.2–0.4s) |
| Duplicate filter | ~108ms |
| **Tổng** | **~2.4s** |

Trước tối ưu: query đầu 6.8s, steady-state 3.2s (2 vòng retrieve+filter).

Hai phát hiện đáng báo cáo:

1. **Oversample 3× đắt hơn là tiết kiệm** — ngược trực giác "HNSW chỉ tốn theo
   `ef`". Chi phí payload + filter đều scale theo số candidate. Đây là empirical
   finding thật, phản bác một giả định phổ biến.
2. **Archive-native frame access** — mảnh mạnh nhất. 1 Range request/frame qua
   MP4 sample-table index: từ ~200s xuống **8.7s cho 20 request đồng thời**.
   Đây là *cơ chế*, không phải trick.

Về vị trí trong literature: Scanner và VStore đều giả định video nằm ở **local
storage đã decode/transcode**. Chế độ ở đây khác: *corpus nằm trong ZIP từ xa,
không tải hết được, mà vẫn cần frame-accurate random access dưới latency
interactive*. Ngày càng phổ biến (cloud archive, cold storage, data do BTC
host).

### 4.3 Về UI — chỗ khó claim nhất

"Friendly và sáng tạo" **không phải contribution** nếu không có user study.
Đây là chỗ dễ bị bác nhất.

Nhưng có đúng một thứ nêu tên được: **slider duplicate-threshold chỉnh trực
tiếp + `config_overrides` tune weight ngay lúc query** → *người dùng tự tune
retrieval policy tại thời điểm truy vấn*. Hầu hết hệ thống khác chôn dedup
thành policy cố định ở backend. Cái này cụ thể, mô tả được, và nối thẳng vào
SeqDiv (diversity thành tham số người dùng điều khiển, không phải hằng số hệ
thống).

---

## 5. Khuyến nghị: xếp thứ tự, đừng chọn một

| | Paper 1 (system/resource) | Paper 2 (SeqDiv) |
|---|---|---|
| Dựa trên | Thứ **đã làm xong** | Thứ cần annotate |
| Chi phí thêm | Gần như 0 | Hàng trăm giờ |
| Trần | Thấp hơn (demo/short/system paper — MMM/VBS, OSS track) | Cao hơn (full paper ICMR/MMM) |
| Rủi ro chính | Phụ thuộc kết quả thi đấu | Annotator không đồng thuận về cluster |
| Thời điểm | Bây giờ | Sau, để ủ |

Cách này de-risk: lấy được một paper từ công đã làm xong, trong khi cái đắt
tiền vẫn tiếp tục chín. Nếu dồn hết vào SeqDiv mà annotator không đồng thuận
nổi về equivalence cluster → mất cả hai.

---

## 6. Việc cần làm ngay (theo thứ tự)

1. **Gate 0** — đo redundancy rate thật trên 50–100 query. Tốn 0 đồng
   annotation, một buổi. Quyết định SeqDiv có đáng theo không. → §3.2
2. **Sửa `S_time`** trong `01_seqdiv_proposal.md` — tách shape/scale. → §3.1
3. **Gate A0** — viết guideline cluster + pilot 2 annotator + đo kappa trên 50
   query, **trước khi** cam kết gán nhãn full. → §3.3
4. Nếu Gate 0 + A0 pass → tiếp tục SeqDiv theo `03_experiment_protocol.md`,
   nhưng đưa RQ5 lên trung tâm. → §3.4
5. Song song: gom bằng chứng cho system paper (git stats, latency breakdown,
   competition log). → §4
