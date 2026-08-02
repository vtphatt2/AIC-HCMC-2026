# Input chuẩn để đánh giá các notebook

## Kết luận ngắn

Không nên tạo 13 bộ input riêng. Dùng **một evaluation dataset chung**, trong đó ảnh/frame, transcript, OCR, embedding, query và ground truth liên kết bằng ID. Mỗi notebook chỉ đọc phần nó cần.

Bộ hiện có mới là điểm khởi đầu:

- Có 9 metadata/transcript và 160 query cho 79 candidate.
- 152/160 query (95%) thuộc nhóm `L01`; `L02/L03` gần như chưa được đại diện.
- Thiếu image/keyframe corpus, embedding, shot boundary và OCR ngay trong thư mục.
- Chưa có nhãn relevance độc lập của con người, split calibration/test và kết quả chạy pipeline thật.

## Cấu trúc thư mục đề xuất

```text
eval_data/
├── manifest.json
├── videos.jsonl
├── keyframes.jsonl
├── images/{video_id}/{frame_number}.jpg
├── shots.jsonl
├── transcripts.jsonl
├── ocr.jsonl
├── frame_segment_labels.jsonl
├── queries.jsonl
├── qrels.jsonl
├── query_variants.jsonl              # tùy chọn: typo/bỏ dấu/viết tắt
├── embeddings/
│   ├── visual.npy
│   ├── visual_index.jsonl
│   ├── transcript.npy                # tùy chọn
│   └── transcript_index.jsonl        # tùy chọn
├── runs/{strategy}.jsonl
└── timings/{strategy}.jsonl
```

Dùng đường dẫn tương đối tính từ `eval_data/`; không dùng absolute path kiểu `/home/...`. JSONL phù hợp dữ liệu có schema rõ và dễ stream; vector lớn để trong một ma trận NumPy thay vì hàng nghìn file nhỏ.

## Schema tối thiểu

### `manifest.json`

Ghi nguồn và phiên bản để kết quả có thể tái lập.

```json
{
  "dataset_id": "aic2026_eval_v1",
  "created_at": "2026-07-22",
  "fps_unit": "frames_per_second",
  "time_unit": "milliseconds",
  "visual_embedding": {
    "model": "timm/PE-Core-bigG-14-448",
    "dimension": 1280,
    "dtype": "float32",
    "l2_normalized": true
  }
}
```

Nếu đổi model/checkpoint/preprocessing thì tạo dataset version mới; không trộn embedding khác model trong cùng ma trận.

### `videos.jsonl`

```json
{"video_id":"L01_V001","source_url":"https://...","fps":25.0,"duration_ms":1380000,"width":910,"height":512,"genre":"thoi_su"}
```

Bắt buộc: `video_id`, `fps`, `duration_ms`. `genre` chỉ dùng khi có nguồn gán nhãn rõ ràng.

### `keyframes.jsonl`

```json
{"frame_id":"L01_V001_031270","video_id":"L01_V001","frame_number":31270,"timestamp_ms":1250800,"image_path":"images/L01_V001/031270.jpg","shot_id":"L01_V001_s0123"}
```

Bắt buộc: ID duy nhất, video, frame number, timestamp và ảnh. Phải kiểm tra `timestamp_ms ≈ frame_number / fps × 1000`.

### `shots.jsonl`

```json
{"shot_id":"L01_V001_s0123","video_id":"L01_V001","start_frame":31120,"end_frame":31340,"start_ms":1244800,"end_ms":1253600}
```

Đây là output thật của shot detector, không phải duration lấy từ phân phối random.

### `transcripts.jsonl`

```json
{"segment_id":"L01_V001_t0042","video_id":"L01_V001","start_ms":22000,"end_ms":28000,"text":"Thị trường bán lẻ thành phố Hồ Chí Minh sôi động trở lại.","language":"vi","source":"youtube"}
```

Format hiện có trong `transcripts_sample/*.jsonl` gần đạt yêu cầu; cần thêm `segment_id`, `video_id`, `language`, `source`. Giữ raw text UTF-8, không ghi text lỗi encoding.

### `ocr.jsonl`

```json
{"frame_id":"L01_V001_031270","text":"THỊ TRƯỜNG BÁN LẺ","confidence":0.91,"engine":"paddleocr","engine_version":"x.y"}
```

Một frame có thể có nhiều dòng OCR; khi đó thêm `block_id` và bbox `[x1,y1,x2,y2]`. Không cần bbox nếu notebook chỉ đánh giá retrieval text.

### `frame_segment_labels.jsonl`

Nhãn độc lập để notebook temporal không tự coi “gần thời gian” là “đúng ngữ nghĩa”.

```json
{"frame_id":"L01_V001_031270","segment_id":"L01_V001_t0042","relevance":2,"annotator":"human_01"}
```

Quy ước: `0` không liên quan, `1` liên quan một phần, `2` liên quan rõ. Nên có hai người gán nhãn cho một tập nhỏ và giải quyết bất đồng trước khi chốt test set.

### `queries.jsonl`

```json
{"query_id":"q001","text":"cảnh slide về giao thông có hai người họp báo","language":"vi","complexity":"L2","topic":"giao_thong","split":"test"}
```

Bắt buộc: `query_id`, `text`, `language`, `split`. `complexity` dùng tập giá trị cố định `L1`, `L2`, `L3`; không suy ra mức này từ score sau khi chạy.

### `qrels.jsonl`

Một dòng cho mỗi cặp query–frame được gán nhãn. Đây là ground truth dùng chung cho Recall/MRR/NDCG.

```json
{"query_id":"q001","frame_id":"L01_V003_020127","relevance":2,"annotator":"human_02","source":"human"}
{"query_id":"q001","frame_id":"L01_V003_020164","relevance":1,"annotator":"human_02","source":"human"}
```

Không mở rộng ground truth chỉ bằng cosine similarity rồi dùng chính cosine đó để đánh giá. Neighbor expansion tự động có thể làm candidate cho người duyệt, nhưng `source` cuối cùng phải cho biết nhãn human hay heuristic.

### Embedding

`visual.npy` là ma trận `float32 [N, 1280]`; `visual_index.jsonl` ánh xạ row sang frame:

```json
{"row":0,"frame_id":"L01_V001_031270"}
```

Load hiệu quả bằng `np.load(path, mmap_mode="r")`. Transcript/query embedding khác model hoặc khác dimension phải để file riêng. Không tính cosine trực tiếp giữa vector 1280d và 384d; muốn shared space phải train projection trên calibration split rồi đánh giá trên test split.

### `query_variants.jsonl` (tùy chọn)

```json
{"variant_id":"q001_typo_01","query_id":"q001","text":"canh slide ve giao thong co hai nguoi hop bao","noise_type":"remove_accent","severity":0.4}
```

Variant dùng cùng qrels với query gốc. Đánh giá mức giảm metric thật, không tự giảm score bằng công thức.

### Output pipeline thật

`runs/{strategy}.jsonl` — một dòng cho mỗi result:

```json
{"query_id":"q001","frame_id":"L01_V003_020127","rank":1,"visual_score":0.41,"ocr_score":0.63,"transcript_score":0.52,"reranker_logit":2.17,"final_score":0.71}
```

`timings/{strategy}.jsonl` — một dòng cho mỗi query:

```json
{"query_id":"q001","latency_ms":83.4,"candidate_count":150,"device":"RTX_4090","warmup":false}
```

Latency phải đo quanh request/pipeline thật. Bỏ warm-up khỏi p50/p95 nhưng vẫn lưu để audit.

## Notebook nào dùng input nào

| Notebook | Input bắt buộc | Đánh giá hợp lệ |
|---|---|---|
| `01_*` | keyframes, transcripts, frame–segment labels, embedding tương thích | Accuracy/F1 của temporal match theo tolerance; similarity theo nhãn relevance, không theo score tự sinh |
| `02_*`, `03_*` | visual/text embeddings, queries, qrels, OCR | Hubness/Gini trên corpus thật; Recall@K theo L1/L2/L3; OCR–transcript overlap thực |
| `04_*` | top-K candidate run, qrels, reranker logits, timings | Recall@K/MRR/NDCG trước–sau rerank; calibration trên calibration split; latency thật theo K |
| `05_*` | videos, OCR, transcripts, shots, query variants, qrels | Density theo genre, cross-shot errors và robustness drop từ retrieval thật |
| `06_*` | output metric của các notebook trước | Chỉ là dashboard; không tự sinh lại data |
| `07_*`, `08_*`, `09_*` | ảnh liên tiếp, shots, visual embeddings, queries, qrels | Tỷ lệ frame bị NMS, Recall@K trước–sau NMS/hub suppression, số relevant frame bị xóa |
| `10_*`, `11_*`, `advanced_*` | cùng corpus, query topic/complexity, transcripts, qrels, run scores | Ablation từng heuristic; chunk size/temporal decay chọn trên calibration và xác nhận trên test |
| `12_*` | queries, qrels, real baseline/nextgen runs và timings | A/B end-to-end trên cùng corpus và cùng query; không dùng proxy/random candidates |
| `TextEncoderExtract`, scraper | Không thuộc evaluation | Utility tạo model/transcript; test riêng về correctness và runtime |

## Chia dữ liệu và quy tắc chạy

1. **Split theo video**, không random theo frame/query; frame gần nhau của cùng scene không được rơi vào hai split.
2. Dùng `calibration` để chọn threshold, weight, temperature, chunk size. Đóng băng chúng trước khi chạy `test`.
3. Giữ một baseline đơn giản cố định. Mỗi lần chỉ bật thêm một heuristic để biết gain đến từ đâu.
4. Chạy cùng query/corpus cho mọi strategy; lưu raw ranking và timing, không chỉ lưu số tổng hợp.
5. Báo tối thiểu `Recall@10/50/100`, MRR, NDCG@10, p50/p95 latency và số query lỗi. Với NMS, thêm `% relevant frames removed`.
6. Báo kết quả toàn tập và theo video/query complexity. Không kết luận theo genre nếu mỗi genre chỉ có vài query.
7. Chỉ gọi một thay đổi “tốt hơn” khi metric test tăng mà latency/memory vẫn trong budget đặt trước.

## Bộ tối thiểu để bắt đầu

Không cần chờ dataset hoàn hảo. Bản đầu đủ dùng gồm:

- Toàn bộ keyframe + PE-Core embedding của 9 video hiện có.
- Transcript JSONL đã có; bổ sung shot boundaries và OCR.
- Giữ 160 query hiện tại làm smoke test, nhưng bổ sung query cho `L02/L03` và chia lại theo video.
- Human-review qrels cho ít nhất 100 query test; mỗi query có thể có nhiều frame đúng.
- Chạy baseline text-vector search thật và lưu `runs/baseline.jsonl` + `timings/baseline.jsonl`.

Sau bước này mới lần lượt đánh giá NMS, anti-hubness, reranker, temporal decay và fusion. Nếu chưa có qrels và run thật thì notebook chỉ nên mang nhãn **sandbox**, không được sinh “production config”.
