# Experiment protocol

## 1. Evaluation principles

1. Chốt hypothesis và metric trước khi nhìn test results.
2. Split theo video, không random theo frame.
3. Calibration set dùng để chọn threshold, event weights, `lambda`, `tau` và
   refill policy; test set bị đóng băng.
4. Mọi method dùng cùng base retriever, candidate source và hardware.
5. Lưu raw rankings/timings, không chỉ lưu aggregate table.
6. Không dùng heuristic cosine expansion làm final human qrels.

## 2. Existing assets and limitations

### Có thể tái sử dụng

- 160 Vietnamese queries và 79 target scenes làm smoke/calibration seed.
- PE-Core visual embeddings và Milvus search pipeline.
- Transcript/metadata assets.
- Temporal DP, RRF và duplicate filter hiện tại làm baselines.

### Chưa đủ cho paper

- 152/160 query thuộc L01.
- Chưa có independent human qrels.
- Chỉ có 9 TRAKE query trong 192 competition query files.
- Generated queries không phải independent test queries.
- 11/13 notebook EDA dùng mock/random data.

## 3. Dataset construction

### Minimum viable evaluation

- Ít nhất 100 frozen test queries, mỗi query gồm 2-4 ordered events.
- Một calibration set riêng; không reuse test query để tune.
- Mỗi query có:
  - target video(s);
  - relevant interval cho từng event;
  - required order và optional gap tolerance;
  - graded relevance;
  - equivalence-cluster ID cho các chain trả cùng narrative/moment.
- Hai annotator trên ít nhất một subset đủ lớn; giải quyết disagreement trước
  khi freeze test set.

### Stronger evaluation

- Vietnamese AIC-style corpus cho in-domain evaluation.
- Một public corpus thứ hai. Có thể tạo ordered queries từ các video có nhiều
  temporally localized captions, nhưng phải publish deterministic construction
  scripts và human-review test labels.
- Nếu video không được redistribute, release annotation, manifest, checksum,
  data-preparation scripts và hướng dẫn xin dataset.

### Suggested schemas

`queries.jsonl`:

```json
{"query_id":"q001","split":"test","language":"vi","events":[{"event_id":"e1","text":"..."},{"event_id":"e2","text":"..."}],"constraints":{"ordered":true,"max_span_ms":30000}}
```

`qrels_chains.jsonl`:

```json
{"query_id":"q001","chain_id":"c001","video_id":"L01_V001","steps":[{"event_id":"e1","start_ms":10000,"end_ms":14000},{"event_id":"e2","start_ms":18000,"end_ms":23000}],"relevance":2,"equivalence_cluster":"q001_a","annotator":"human_01"}
```

`runs/<method>.jsonl`:

```json
{"query_id":"q001","rank":1,"chain_id":"pred_001","video_id":"L01_V001","frame_ids":["...","..."],"timestamps_ms":[12000,20500],"relevance_score":0.72,"diversity_score":0.18,"candidate_round":1}
```

`timings/<method>.jsonl`:

```json
{"query_id":"q001","latency_ms":84.3,"candidates_fetched":300,"refill_rounds":1,"top_k_filled":true,"warmup":false}
```

## 4. Baselines

### Retrieval baselines

1. Raw visual per-event retrieval + DP.
2. Fixed four-channel RRF + DP.

Giữ một trong hai làm frozen base retriever cho diversification experiment.

### Diversification baselines

1. No diversification.
2. One result per video.
3. Ingestion-time pHash/embedding dedup.
4. Frame-level cosine threshold trên final step.
5. Cosine threshold trên mean chain embedding.
6. MMR trên mean chain embedding.
7. MMR trên concatenated aligned-step embeddings.
8. DPP hoặc một submodular diversification baseline.
9. Current greedy all-step threshold.
10. Proposed SeqDiv.

### Refill baselines

1. Fetch exactly K.
2. Fixed oversampling: `1.5K`, `2K`, `4K`.
3. Refill fixed-size pages đến K.
4. Proposed budget-aware refill.

## 5. Metrics

### Relevance

- Recall@10/50/100.
- MRR.
- nDCG@10 và nDCG@K.
- Chain success rate: đúng video, đủ event, đúng thứ tự và trong tolerance.

### Diversity

- alpha-nDCG@K.
- Distinct equivalence-cluster coverage@K.
- Relevant video/scene coverage@K.
- Mean/max pairwise chain similarity.
- Redundant-pair rate trong top-K.

### Safety

- Tỷ lệ relevant chains bị diversification loại.
- Recall loss so với no-filter.
- Query count mà top-K không được fill.

### Efficiency

- Cold-start và warm p50/p95 latency.
- Candidate/vector count mỗi query.
- Refill rounds mỗi query.
- Embedding fetches và database round trips.
- Peak request memory nếu khác biệt đáng kể.

### User-centric metrics

- Time-to-first-correct.
- Số result/card đã inspect trước khi submit đúng.
- Task success trong fixed time budget.

## 6. Ablations

### Chain similarity

- Semantic only.
- Semantic + temporal gaps.
- Semantic + evidence modality.
- Full combination.
- Uniform vs learned/calibrated event weights.
- Mean vs minimum/product soft-AND aggregation.

### Selection objective

- Hard threshold.
- Standard MMR.
- Sequence-aware MMR.
- Coverage/submodular variant.

### Refill

- No refill.
- Current `1.5x` + paginate.
- Candidate-budget only.
- Latency-budget only.
- Combined stopping rule.

### Query characteristics

- 2, 3 và 4+ events.
- Short vs long temporal span.
- Visual-heavy, OCR-heavy và speech-heavy queries.
- High vs low initial duplicate rate.

## 7. Statistical reporting

- Report bootstrap 95% confidence intervals over queries.
- Dùng paired test hoặc paired bootstrap cho method comparisons.
- Report per-query deltas, không chỉ aggregate mean.
- Threshold/weight chỉ được chọn trên calibration split.
- Ghi seed, model/checkpoint, index config, hardware và commit hash.

## 8. Go/no-go gates

### Gate A — dataset validity

Không tiếp tục claim effectiveness nếu chưa có human-reviewed qrels, split theo
video và raw run files tái lập được.

### Gate B — method signal

SeqDiv phải tăng diversity/coverage rõ ràng so với frame-level và standard MMR,
trong khi Recall@K/nDCG không giảm quá trade-off đã đăng ký trước.

### Gate C — efficiency

SeqDiv phải hoạt động trong latency budget của interactive search và tốt hơn
fixed oversampling trên ít nhất một Pareto point candidate-cost vs quality.

Nếu fail Gate B, pivot sang retrieval-aware keyframe selection. Nếu fail Gate C
nhưng Gate B pass, định vị paper là effectiveness study và tách refill khỏi main
contribution.

## 9. Reproducibility checklist

- Dataset manifest, license và checksums.
- Frozen queries/qrels.
- Data preparation scripts.
- Raw run/timing JSONL.
- Config snapshot cho từng method.
- Environment lockfile/container.
- Script tạo tables/figures từ raw outputs.
- Không dùng số liệu copy từ notebook mock/archive.
