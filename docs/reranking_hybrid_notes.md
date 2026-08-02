# Reranking hybrid notes

## Mục tiêu

Giữ các phần tốt của những strategy hiện tại, nhưng không coi mỗi strategy là
một pipeline khép kín. Các tín hiệu nên có thể ghép theo thứ tự:

```text
retrieve candidates -> fuse ranks -> learned rerank text evidence
-> temporal sequence search -> final ranking
```

Đây là hướng thử nghiệm, chưa phải cấu hình production. Mọi weight, candidate
window và threshold phải được đo lại trên query/frame label thật.

## Những gì repo đang có

| Thành phần | Điểm tốt | Phần còn thiếu |
|---|---|---|
| PECore cosine ranking | Nhanh, phù hợp semantic query -> image | Không xử lý tốt hard negative; chỉ là một ranking list |
| Stable Fusion | Ghép visual, OCR, transcript; có genre weight | Weighted sum phụ thuộc scale/calibration của từng score |
| Transcript Fusion | RRF ghép visual, OCR, transcript mà không cần cùng score scale | Hiện fuse theo `video_id`, chưa phải frame-level; không có temporal DP |
| Temporal DP | Tìm chuỗi frame tối ưu thỏa offset/span | Chỉ tối ưu tổng PECore score; chưa dùng RRF hoặc learned relevance |
| Temporal heuristic | Có proximity score và missing-step penalty | Công thức cố định, chưa được calibrate bằng label thật |

## Cách ghép nên thử

### 1. RRF trước DP

Không thay trực tiếp cosine score bằng RRF. Cosine tạo ra **một ranked list**;
RRF nhận nhiều ranked list rồi sinh score hợp nhất:

```text
visual cosine rank --\
OCR rank ----------- RRF(frame/video) -> candidate score -> temporal DP
transcript rank ----/
```

Sau khi có frame-level RRF score, DP tối ưu tổng hoặc trung bình RRF score thay
vì chỉ dùng PECore cosine.

### 2. Learned score là thêm một tín hiệu, không thay toàn bộ pipeline

Learned text reranker chấm cặp:

```text
query <-> "OCR: ... Transcript: ..."
```

Nó không đọc ảnh. Với frame không có OCR/transcript, giữ PECore/RRF score và
không phạt chỉ vì thiếu text evidence.

Hai cách ghép cần benchmark:

1. Đưa learned ranking thành một list nữa trong RRF. Đây là mặc định an toàn vì
   không cần giả định raw logit cùng scale với cosine.
2. Weighted sum sau khi calibrate score trên validation set. Chỉ dùng nếu thực
   nghiệm cho thấy tốt hơn RRF.

### 3. DP ở cuối

DP nên chạy sau candidate retrieval/reranking để tìm chuỗi thỏa temporal offset.
Mỗi step dùng score đã hợp nhất, còn DP giữ nhiệm vụ tìm đường đi tối ưu:

```text
best_path = argmax sum(final_candidate_score)
```

## Learned reranker CPU đầu tiên

Thử `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`:

- khoảng 0.1B tham số, multilingual, Apache-2.0;
- nhận `(query, passage)` và trả relevance score;
- model repository có sẵn ONNX INT8 khoảng 119 MB; ưu tiên bản AVX2 cho CPU
  x86 phổ thông, chỉ dùng AVX512/VNNI khi máy hỗ trợ;
- chạy bằng ONNX Runtime trên CPU, phù hợp với dependency `onnxruntime` repo đã có.

Model card:
<https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1>

Quantized artifact:
<https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/blob/main/onnx/model_quint8_avx2.onnx>

Sentence Transformers hỗ trợ chọn thẳng ONNX quantized file bằng `backend="onnx"`
và `model_kwargs.file_name`:
<https://www.sbert.net/docs/cross_encoder/usage/efficiency.html#onnx>

Spike local nên gọi tokenizer + `onnxruntime` trực tiếp để giữ base runtime không
có PyTorch. Không thêm `sentence-transformers` vào local chỉ để bọc một ONNX
session; chỉ cân nhắc wrapper đó nếu code trực tiếp trở thành gánh nặng bảo trì.

Không chọn làm spike đầu tiên:

- `BAAI/bge-reranker-v2-m3`: 568M, nặng hơn cho CPU;
- `jina-reranker-v2-base-multilingual`: 278M và license CC-BY-NC-4.0;
- visual/VLM reranker: cần ảnh, model lớn hơn và chưa cần thiết trước khi text
  reranking chứng minh được lợi ích.

## Spike tối thiểu

1. Retrieve top 50, không chỉ retrieve đúng số kết quả cần trả.
2. Tạo passage từ OCR và transcript nằm gần timestamp của frame.
3. Rerank chỉ những candidate có passage; batch inference trên CPU.
4. Thêm learned ranking vào RRF; giữ visual ranking nguyên vẹn.
5. Trả top K và ghi latency/model score để so sánh.

Không thêm threshold hoặc weight tùy ý trong spike đầu tiên.

## Điều kiện giữ hoặc bỏ

So sánh baseline với hybrid trên cùng tập query được gán nhãn:

- nDCG@10 hoặc MRR@10 phải tăng;
- Recall@50 không được giảm do candidate filtering;
- đo cold-start, warm p50 và warm p95 trên máy CPU mục tiêu;
- visual-only query không được xấu đi;
- nếu chất lượng không tăng đủ để bù latency, bỏ learned reranker và giữ RRF + DP.

## Việc để sau

- Frame-level RRF thay cho video-level RRF hiện tại.
- Caption cho keyframe để learned text reranker hỗ trợ visual-only query.
- Calibrate learned logits và học fusion weights bằng label thật.
- Multimodal reranker chỉ khi benchmark chứng minh text evidence chưa đủ.
