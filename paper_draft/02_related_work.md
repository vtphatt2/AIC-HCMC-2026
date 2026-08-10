# Related work and novelty boundary

## 1. Interactive and temporal video retrieval

Temporal queries đã xuất hiện trong các interactive video retrieval systems từ
trước. Multi-Event Video-Text Retrieval còn formalize video chứa nhiều event và
text chỉ mô tả một phần nội dung. Vì vậy paper không được claim “support ordered
events” là contribution duy nhất.

Các công trình gần repo nhất:

| Work | Nội dung gần repo | Hệ quả cho proposal |
|---|---|---|
| Multi-Modal Interactive Video Retrieval with Temporal Queries, MMM 2022 | Interactive temporal query | Temporal UI/query không mới |
| Multi-Event Video-Text Retrieval, ICCV 2023 | Multi-event representation và training objective | Phải phân biệt video-level MeVTR với result-chain diversification |
| DANTE, 2025 | TransNetV2, Milvus, DP cho TRAKE | Không claim DP event-chain matching |
| MADTempo, 2025 | Multi-event temporal retrieval và query augmentation | Không claim sequential search đơn thuần |
| Cross-modal Temporal Event Retrieval, SOICT 2025 | Mỗi event có thể dùng modality khác | Không claim multimodal event fusion |
| MERVIN, 2026 | PE-Core, Milvus, transcript trên Vietnamese news | Không claim stack PE-Core/Milvus/transcript |
| CVPRW IViSE systems, 2025 | TransNetV2, keyframe dedup, reranking, temporal search | Ingestion dedup và temporal reranking đã có prior art gần |

## 2. Result diversification

Maximal Marginal Relevance (MMR) là baseline nền tảng kết hợp relevance và
novelty. xQuAD, DPP và submodular selection là các họ baseline khác cần cân nhắc.

SeqDiv không nên tự mô tả là “MMR nhưng dùng video”. Novelty phải nằm ở:

- đơn vị kết quả là ordered event chain;
- kernel so sánh aligned steps và temporal-gap geometry;
- evaluation labels cho distinct equivalent chains;
- interaction giữa diversification và candidate/latency budget.

Metric diversity nên dựa trên alpha-nDCG và aspect/subtopic coverage. Trong
evaluation, một “aspect” có thể là một human-annotated equivalence cluster của
event chains hoặc một distinct relevant video/scene.

## 3. Near-duplicate video and keyframe deduplication

Near-duplicate detection thường xử lý:

- hai video là copy/near-copy;
- hai keyframe trong cùng shot gần giống nhau;
- loại redundancy trước khi xây index.

SeqDiv xử lý một bài toán khác: hai **answers** có thể là hai chain khác nhau về
frame ID nhưng cùng trả lại một narrative/moment cho người dùng. Ingestion-time
dedup vẫn là baseline bắt buộc, vì reviewer có thể hỏi tại sao không xóa trùng
từ offline pipeline.

## 4. Frame/keyframe selection

Frame selection cho text-video retrieval, video summarization và long-video VLM
đã là một literature lớn. Các paper mới đã dùng clustering, semantic relevance,
diversity, temporal coverage và adaptive stopping.

Do đó retrieval-aware keyframe selection là hướng dự phòng khả thi, nhưng phải
định vị rõ là **offline corpus indexing under a storage budget**, không phải
question-conditioned frame selection để đưa vào một VLM.

## 5. Candidate expansion and reranking

Graph-based Adaptive Re-ranking (GAR) mở rộng pool bằng feedback từ những result
đang được rerank. Repo hiện chỉ lấy trang tiếp theo và exclude frame ID đã thấy.
Vì vậy:

- “paginated refill” là mô tả đúng code hiện tại;
- “adaptive retrieval” chỉ hợp lệ khi proposal thật sự có policy chọn candidate
  dựa trên observed relevance/diversity gain.

## 6. Video processing systems

Scanner đã nghiên cứu sparse compressed-frame access, batching và heterogeneous
video pipelines. VStore tối ưu format/resource xuyên suốt vòng đời video.
NoScope dùng cost-based optimization dưới quality constraints. Vì vậy chỉ
cross-video batching, bounded queues hoặc selective FFmpeg decode chưa đủ làm
systems contribution.

## Novelty statement draft

> Existing work has independently studied ordered temporal video retrieval,
> item-level result diversification, and near-duplicate video filtering. In
> contrast, we study diversity in ranked lists whose atomic results are ordered
> multi-event chains. We model redundancy through aligned event semantics and
> temporal-gap structure, and optimize this objective under the candidate and
> latency budgets of interactive video search.

## Primary links

- MMR: <https://doi.org/10.1145/290941.291025>
- Alpha-nDCG/diversity evaluation: <https://doi.org/10.1145/1390334.1390446>
- Multi-Event Video-Text Retrieval: <https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Multi-Event_Video-Text_Retrieval_ICCV_2023_paper.html>
- Multi-modal temporal queries: <https://doi.org/10.1007/978-3-030-98355-0_44>
- DANTE: <https://arxiv.org/abs/2512.13169>
- MADTempo: <https://arxiv.org/abs/2512.12929>
- Cross-modal temporal event retrieval: <https://arxiv.org/abs/2512.06334>
- MERVIN: <https://arxiv.org/abs/2605.16120>
- CVPRW 2025 unified moment retrieval: <https://openaccess.thecvf.com/content/CVPR2025W/IViSE/html/Tran_Towards_Efficient_and_Robust_Moment_Retrieval_System_A_Unified_Framework_CVPRW_2025_paper.html>
- Adaptive Re-ranking with a Corpus Graph: <https://arxiv.org/abs/2208.08942>
- QSVideo: <https://arxiv.org/abs/2607.04559>
- Scanner: <https://graphics.stanford.edu/papers/scanner/>
- VStore: <https://arxiv.org/abs/1810.01794>
