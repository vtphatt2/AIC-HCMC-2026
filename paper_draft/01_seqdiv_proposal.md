# SeqDiv proposal

## Working title

**SeqDiv: Budget-Aware Sequence-Level Diversification for Interactive
Multi-Event Video Retrieval**

Tên thay thế:

- *Diversifying Ordered Event Chains for Interactive Video Search*
- *Latency-Bounded Diversification of Multi-Event Video Retrieval Results*
- *Beyond Frame-Level Deduplication: Diverse Ordered Event Retrieval under a Candidate Budget*

## One-sentence pitch

SeqDiv đa dạng hóa ranked list ở cấp **ordered event chain**, thay vì ở cấp
frame/video độc lập, rồi lấy thêm candidate trong một ngân sách định trước để
duy trì top-K kết quả hữu ích cho interactive retrieval.

## Bài toán

Một temporal query gồm `m` event có thứ tự:

```text
q = (q1 -> q2 -> ... -> qm)
```

Retriever trả về các chain:

```text
C = (f1 -> f2 -> ... -> fm)
```

Mỗi `fi` là evidence cho event `qi`; chain còn chứa video ID, timestamp,
per-event score và modality provenance. Ranked list hiện thường chứa nhiều
chain gần như cùng một moment, làm lãng phí top-K và thời gian của người dùng.

## Repo đã có gì

- `remote-server/app/strategies/multi_source_temporal.py` fusion nhiều channel
  theo từng event rồi gọi temporal matching.
- `remote-server/app/strategies/_duy_temporal_core.py` dùng dynamic programming
  để tạo các chain hợp lệ theo video, thứ tự và temporal constraints.
- `remote-server/app/strategies/_similarity_filter.py` chỉ loại một result khi
  tất cả cặp step tương ứng đều vượt cosine threshold.
- `remote-server/app/strategies/base_strategy.py` oversample trang đầu `1.5x`,
  sau đó paginate/refill cho đến khi đủ top-K, hết dữ liệu hoặc chạm fetch cap.

Những thành phần trên là điểm xuất phát và baseline, chưa phải contribution
hoàn chỉnh của paper.

## Research gap

Related work gần nhất đã giải quyết:

- multimodal temporal retrieval;
- dynamic programming cho ordered events;
- ingestion-time keyframe deduplication;
- item/frame-level relevance-diversity reranking;
- near-duplicate detection ở cấp whole video.

Khoảng trống cần kiểm chứng là **diversification của ranked ordered event
chains**, trong đó redundancy phụ thuộc đồng thời vào:

1. semantic correspondence giữa các event cùng vị trí;
2. hình dạng temporal gaps của cả chain;
3. evidence/modalities được dùng để support từng event;
4. candidate và latency budget của interactive search.

## Research questions

### RQ1 — Effectiveness

Sequence-level diversification có tăng `alpha-nDCG@K` và distinct-chain
coverage so với no-filter, frame-level threshold và standard MMR không?

### RQ2 — Relevance preservation

Gain về diversity có đạt được mà không làm giảm đáng kể Recall@K, MRR và
nDCG@K hay không?

### RQ3 — Chain representation

Aligned event semantics, temporal-gap shape và modality provenance đóng góp bao
nhiêu vào việc nhận biết hai chain tương đương?

### RQ4 — Efficiency

Budget-aware paginated refill có duy trì top-K tốt hơn fixed oversampling dưới
cùng số candidate hoặc p95 latency hay không?

### RQ5 — User impact

Kết quả đa dạng hơn có giảm time-to-first-correct và số result người dùng phải
inspect trong một VBS-style task hay không?

## Proposed method

### 1. Base relevance

Giữ base retriever cố định để paper không trộn hai contribution:

```text
per-event retrieval -> optional multimodal RRF -> temporal DP -> candidate chains
```

Mỗi chain `C` có relevance score `R(C)` lấy từ temporal matcher. Base retriever
phải giống nhau cho mọi diversification baseline.

### 2. Chain similarity

Với hai chain có cùng số event:

```text
C = (f1, ..., fm), D = (g1, ..., gm)
```

Định nghĩa ba thành phần:

```text
S_sem(C,D)  = weighted aligned-event semantic similarity
S_time(C,D) = similarity between normalized temporal-gap vectors
S_evd(C,D)  = similarity/overlap of supporting modalities or evidence types
```

Một dạng khởi đầu:

```text
S_chain(C,D) = alpha * S_sem(C,D)
             + beta  * S_time(C,D)
             + gamma * S_evd(C,D)
```

Trong đó:

```text
S_sem(C,D) = sum_i w_i * cosine(e(fi), e(gi)) / sum_i w_i
gap(C)     = normalize([t2-t1, ..., tm-t(m-1)])
S_time     = exp(-L1(gap(C), gap(D)) / tau)
```

Cần ablate cả soft average, minimum/product kiểu soft-AND và current hard
all-step threshold. Hai chain khác số event không so sánh trực tiếp trong bản
đầu; extension có thể dùng sequence alignment nhưng không nên mở scope quá sớm.

### 3. Sequence-aware selection

Baseline objective theo MMR:

```text
score(C | selected) = lambda * R(C)
                    - (1-lambda) * max_D S_chain(C,D)
```

Paper nên thử thêm một coverage/submodular objective nếu annotation có distinct
chain clusters. Tuy nhiên standard MMR phải luôn là baseline mạnh; chỉ đổi từ
frame embedding sang concatenated chain embedding không đủ novelty.

### 4. Candidate-budgeted refill

Thay vì fetch một pool rất lớn ngay từ đầu:

1. fetch initial candidate page;
2. construct chains và diversify;
3. ước lượng duplicate rate / marginal coverage gain;
4. chỉ fetch trang tiếp theo khi expected gain còn đủ lớn và chưa vượt budget;
5. dừng khi đủ K, hết source, hết candidate budget hoặc hết latency budget.

Tên nên dùng trong paper là **budget-aware paginated refill**. Không gọi đây là
feedback-guided adaptive retrieval nếu policy chỉ lấy trang tiếp theo.

## Expected contributions

1. Một formulation cho relevance-diversity của ordered multi-event chains.
2. Một chain kernel/objective kết hợp aligned semantics và temporal-gap shape.
3. Một refill policy tối ưu top-K coverage dưới interactive latency budget.
4. Một benchmark/evaluation protocol có human qrels cho diversity của temporal
   video retrieval results.

## Không claim

- Không claim PE-Core, Milvus, TransNetV2, RRF hoặc DP là mới.
- Không claim current `1.5x` oversampling là adaptive retrieval.
- Không claim synthetic notebook metrics là kết quả thực nghiệm.
- Không claim absolute novelty; dùng cách viết “to the best of our review”.

## Paper outline

1. **Introduction** — top-K redundancy ở multi-event search và user cost.
2. **Related Work** — temporal retrieval, result diversification, near-duplicate
   detection và adaptive candidate expansion.
3. **Problem Formulation** — event chain, relevance, redundancy và budgets.
4. **SeqDiv** — chain similarity, selection objective và refill policy.
5. **Experimental Setup** — datasets, annotations, baselines và metrics.
6. **Results** — effectiveness, relevance preservation, efficiency, ablations
   và user study.
7. **Limitations** — annotation cost, fixed event correspondence, dataset/license.

## Venue fit

- Full/short paper: ACM ICMR, MMM main track, hoặc multimedia IR venue tương đương.
- Nếu chỉ có current heuristic + UI demo: MMM/VBS extended demo hoặc workshop.
- Muốn nhắm venue mạnh hơn cần public benchmark, principled objective, strong
  MMR/DPP baselines và kết quả trên ít nhất hai corpora.
