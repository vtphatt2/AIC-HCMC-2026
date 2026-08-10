# Paper draft workspace

Thư mục này ghi lại hướng nghiên cứu được đề xuất sau khi audit code, dữ liệu,
notebook và related work của repository vào ngày 2026-08-10.

## Khuyến nghị chính

**SeqDiv: Budget-Aware Sequence-Level Diversification for Interactive
Multi-Event Video Retrieval**

Câu hỏi trung tâm:

> Với cùng ngân sách latency và candidate, đa dạng hóa ở cấp cả chuỗi sự kiện
> có đưa được nhiều đáp án đúng và khác nhau lên top-K hơn các phương pháp lọc
> trùng từng frame, mà không làm giảm retrieval recall hay không?

Repo đã có phần nền cần thiết:

- multimodal retrieval và RRF;
- dynamic programming để tạo ordered event chains;
- greedy duplicate filtering ở cấp chuỗi;
- paginated refill khi filtering làm thiếu top-K.

Tuy nhiên, code hiện tại mới là heuristic. Full paper cần một chain-similarity
objective có cơ sở, baseline mạnh, human qrels và evaluation thật.

## Tài liệu

1. [01_seqdiv_proposal.md](01_seqdiv_proposal.md) — proposal và research questions.
2. [02_related_work.md](02_related_work.md) — bản đồ related work và novelty boundary.
3. [03_experiment_protocol.md](03_experiment_protocol.md) — dữ liệu, baseline, metric,
   ablation và go/no-go criteria.
4. [04_alternative_directions.md](04_alternative_directions.md) — hai hướng dự phòng.
5. [05_second_opinion.md](05_second_opinion.md) — review độc lập: đã kiểm chứng
   citations/claim, 1 lỗi công thức cần sửa, Gate 0 + Gate A0 cần thêm, và đề
   xuất tách thành 2 paper theo thứ tự thời gian.
6. [references.bib](references.bib) — bibliography khởi đầu.

## Trạng thái bằng chứng

Các con số trong phần lớn notebook archive **không phải kết quả paper**. Audit
nội bộ cho thấy 11/13 notebook EDA dùng toàn bộ dữ liệu mock/random. Bộ 160
query hiện tại có thể dùng làm smoke test, nhưng chưa phải test set vì lệch mạnh
về L01, thiếu human qrels và thiếu calibration/test split.

Trước khi viết Introduction, nhóm nên tạo một bảng thật gồm:

| Method | Recall@K | nDCG@K | alpha-nDCG@K | Chain coverage@K | p95 latency |
|---|---:|---:|---:|---:|---:|
| No diversification | | | | | |
| Frame-level MMR | | | | | |
| Current all-step greedy | | | | | |
| Proposed SeqDiv | | | | | |

Nếu SeqDiv không tăng diversity/coverage trong khi giữ relevance dưới cùng
latency budget, nhóm nên chuyển sang hướng retrieval-aware keyframe selection.
