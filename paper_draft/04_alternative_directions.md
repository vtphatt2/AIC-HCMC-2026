# Alternative paper directions

## Decision matrix

| Hướng | Novelty potential | Code readiness | Dữ liệu mới | Rủi ro chính |
|---|---:|---:|---:|---|
| SeqDiv ordered-chain diversification | Cao | Trung bình-cao | Cao | Cần chain qrels và objective vượt standard MMR |
| Retrieval-aware keyframe budgets | Trung bình-cao | Cao | Trung bình | Frame selection literature rất đông |
| Archive-native GPU pipeline | Trung bình | Rất cao | Benchmark phần cứng | Scanner/VStore đã cover nhiều systems ideas |
| Vietnamese temporal benchmark | Trung bình | Trung bình | Rất cao | License và annotation; dataset-only claim yếu nếu nhỏ |

## A. Retrieval-aware keyframe selection under storage budgets

### Working title

**Retrieval-Aware Keyframe Selection under Storage Budgets for Long-Form Video Search**

### Repo gap

Pipeline hiện chọn keyframe theo luật độ dài scene:

```text
duration <= 3s  -> 1 frame
duration <= 10s -> 3 frames
duration > 10s  -> 5 frames
```

Luật này không nhìn semantic variation, OCR/transcript changes hay retrieval
utility. Online duplicate filtering sau đó phải sửa redundancy quá muộn.

### Proposed method

Phân bổ một global keyframe budget giữa các scenes dựa trên:

- temporal coverage;
- PE-Core semantic novelty;
- OCR/transcript boundary evidence;
- scene duration;
- optional hubness/redundancy penalty.

Có thể formulate bằng knapsack/submodular budget allocation. Online SeqDiv là
một optional second stage, không nên để paper có quá nhiều contribution ngay từ
bản đầu.

### Baselines

- Uniform interval.
- TransNet fixed 1/3/5.
- Configurable linear rule trong `preprocess/`.
- K-means/k-medoids representatives.
- Scene-level farthest point sampling.
- Full/unpruned index.

### Experiments

- Budget sweep: 20%, 40%, 60%, 80%, 100% vectors.
- Recall/MRR/nDCG và relevant-scene coverage.
- Index size, embedding cost và query latency.
- Phần trăm relevant frames/scenes bị pruning.
- Pareto frontier quality vs storage/compute.

### Novelty boundary

Phải so sánh với empirical frame-selection studies, adaptive keyframe selection,
KTV/CSES và diversity-based selection. Điểm phân biệt nên là offline corpus
indexing và retrieval effectiveness dưới global storage budget, không phải
frame selection cho một query/VLM đơn lẻ.

## B. Archive-native streaming preprocessing systems

### Working title

**Archive-Native, Retrieval-Preserving Scheduling for Resource-Constrained Video Indexing**

### Repo strengths

- streaming TransNet windows;
- cross-video GPU batching;
- bounded decode/GPU queues;
- adaptive sequential vs seek decode;
- batches giữ accumulator qua video boundaries;
- memmap/atomic finalize và resumable pipeline;
- ZIP-native ingestion và remote range access.

### Vì sao current v9.3 chưa đủ

Sparse compressed-frame access, heterogeneous scheduling, dynamic batching và
resource optimization đều có prior systems work. Một systems paper mới cần
ít nhất một algorithmic systems contribution, ví dụ:

1. GOP-aware cost model chọn sequential decode hay seek;
2. phá barrier TransNet -> PE-Core để giảm time-to-first-index;
3. adaptive tuner cho worker count, batch size và flush timeout;
4. quality floor bảo đảm scene equivalence và retrieval Recall@K.

### Baselines

- Sequential v8.
- Global but non-streaming v9.
- Streaming global v9.2.
- Fixed sequential decode vs adaptive seek.
- Per-video flush vs cross-video embedding batches.
- Scanner-style/general pipeline nếu triển khai so sánh công bằng được.

### Metrics

- Makespan và time-to-first-indexed-video.
- Frames/keyframes per second.
- GPU idle time/utilization và batch fill ratio.
- Peak RAM/VRAM/disk, H2D bytes và cloud cost.
- Scene-boundary agreement và embedding cosine parity.
- Retrieval Recall@K parity.
- Recovery work sau fault injection.

Ít nhất ba hardware/resource profiles và raw performance summaries cần được
release. Nếu không, hướng này phù hợp applied systems workshop/demo hơn full
systems venue.

## C. Vietnamese sequential retrieval benchmark

### Working title

**ViSeqRet: A Human-Annotated Benchmark for Vietnamese Sequential Event Retrieval**

### Giá trị

- Vietnamese là low-resource trong multimedia retrieval.
- Sequential/TRAKE query yêu cầu nhiều event đúng thứ tự.
- Có thể benchmark visual, OCR, transcript và temporal reasoning đồng thời.

### Điều kiện để thành paper

- Quy mô đủ lớn và cân bằng query types/videos.
- Human-written hoặc human-reviewed queries/qrels.
- Multiple relevant chains và graded relevance.
- Public annotation, scripts và reproducible baselines.
- Rõ license khi không thể redistribute video.
- Baseline từ visual-only đến multimodal DP và SeqDiv.

Bộ hiện tại chưa đạt các điều kiện này: chỉ 9 TRAKE query, benchmark 160 query
lệch 95% về L01, và thiếu independent human qrels. Vì vậy benchmark nên là
artifact hỗ trợ SeqDiv trước khi cân nhắc một dataset paper riêng.

## Recommended decision

- Có nguồn lực annotation: chọn **SeqDiv**.
- Deadline ngắn hơn nhưng vẫn chạy được nhiều retrieval experiments: chọn
  **retrieval-aware keyframe budgets**.
- Không thể annotate nhưng có nhiều GPU/hardware để benchmark: cân nhắc
  **archive-native systems paper** sau khi thêm scheduler/tuner mới.
