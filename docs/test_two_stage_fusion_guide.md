# Two-Stage Late Fusion — Ma trận Test & Hướng dẫn xác minh Log

## 1. Ma trận Test chức năng

### Kịch bản A: Happy Path (Truy vấn khớp mạnh cả Transcript + Hình ảnh)

**Query:** Semantic: `"cách nấu phở bò truyền thống"`, Text: `"phở bò"`
**Genre:** Ẩm thực

| Giai đoạn | Hành vi mong đợi | Điều kiện kiểm tra |
|-----------|------------------|---------------------|
| Stage 1 | E5 mã hóa `"phở bò"` → tìm trong `transcript_chunks` → trả về các chunk từ video nấu ăn | `stage1_chunks` không rỗng, `stage1_video_ids` chứa `L03_V001` |
| Stage 2 | PE-Core mã hóa semantic query → Milvus search với `expr='video_id in ["L03_V001",...]'` | `stage2_frames` không rỗng, tất cả frames từ các video ứng viên |
| Fusion | α ≈ 0.25 (base Ẩm thực), transcript score ~0.9, visual score ~0.8 | `confidence` > 0.3, kết quả đầu tiên từ `L03_V001` |

**Gọi API:**
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": "two_stage_fusion_strategy",
    "query_groups": [{"semantic_query": "cách nấu phở bò truyền thống", "text_query": "phở bò", "temporal_offset_ms": 0}],
    "top_k": 10,
    "video_genre": "Ẩm thực"
  }'
```

**Log mong đợi:**
```
INFO  Stage 1: transcript search returned 5 chunks, 2 candidate videos
INFO  [TIMER] stage2_vector_search 12.345 ms backend=hnsw group=0 top_k=10 hits=8 expr=True
INFO  TwoStageFusion: alpha=0.257 genre=Ẩm thực transcript_chunks=5 visual_frames=8
INFO  [TIMER] fusion 3.210 ms strategy=TwoStageFusionStrategy results=10
```

---

### Kịch bản B: Truy vấn thiên về hình ảnh (Không khớp lời thoại)

**Query:** Semantic: `"cánh đồng lúa xanh mướt hoàng hôn"`, Text: `""`
**Genre:** Du lịch

| Giai đoạn | Hành vi mong đợi | Điều kiện kiểm tra |
|-----------|------------------|---------------------|
| Stage 1 | Không có text query → bỏ qua Stage 1 | `stage1_chunks` rỗng, `stage1_video_ids` rỗng |
| Stage 2 | PE-Core tìm kiếm visual trên toàn bộ (không có bộ lọc expr) | `stage2_frames` chứa frames từ nhiều video khác nhau |
| Fusion | Fallback về visual-only (alpha không còn ý nghĩa) | `confidence` = visual score, sắp xếp theo độ tương đồng hình ảnh |

**Log mong đợi:**
```
INFO  [TIMER] stage2_vector_search 8.123 ms backend=hnsw group=0 top_k=10 hits=10 expr=False
INFO  TwoStageFusion: Stage 1 empty, falling back to visual-only
INFO  [TIMER] fusion 1.456 ms strategy=TwoStageFusionStrategy results=10
```

---

### Kịch bản C: Không khớp Transcript nào (Truy vấn vô nghĩa)

**Query:** Semantic: `"xyzzy foobar nonsense"`, Text: `"qwerty123"`
**Genre:** All

| Giai đoạn | Hành vi mong đợi | Điều kiện kiểm tra |
|-----------|------------------|---------------------|
| Stage 1 | E5 mã hóa văn bản vô nghĩa → Milvus trả về chunks với điểm rất thấp | `stage1_chunks` có thể có nhưng scores < 0.1 |
| Stage 2 | PE-Core mã hóa văn bản vô nghĩa → Milvus trả về frames với điểm thấp | `stage2_frames` có kết quả nhưng scores < 0.2 |
| Fusion | α động → trọng số transcript thấp (score < 0.05 kích hoạt α=0) | Có kết quả nhưng confidence < 0.2 |

**Log mong đợi:**
```
INFO  Stage 1: transcript search returned 3 chunks, 2 candidate videos
INFO  [TIMER] stage2_vector_search 5.678 ms backend=hnsw group=0 top_k=10 hits=5 expr=True
INFO  TwoStageFusion: alpha=0.000 genre=All transcript_chunks=3 visual_frames=5
INFO  [TIMER] fusion 2.100 ms strategy=TwoStageFusionStrategy results=5
```

---

### Kịch bản D: Kiểm tra ranh giới OCR + Temporal

**Query Step 1:** Semantic: `"bản tin thời sự"`, Text: `"thời sự"`, offset=0
**Query Step 2:** Semantic: `"phóng viên hiện trường"`, Text: `"phóng viên"`, offset=5000
**Genre:** Thời sự

| Giai đoạn | Hành vi mong đợi | Điều kiện kiểm tra |
|-----------|------------------|---------------------|
| Stage 1 | Trả về các chunk khớp chủ đề tin tức/truyền hình | `stage1_chunks` chứa các chunk Thời sự |
| Stage 2 | Tìm kiếm visual được lọc theo các video tin tức ứng viên | `stage2_frames` từ các video tin tức |
| Fusion | α ≈ 0.50 (base Thời sự), OCR boost +5% cho frames khớp text, Temporal boost +5% cho step-2 gần kề | Kết quả đầu có frames khớp OCR được boost |

**Log mong đợi:**
```
INFO  Stage 1: transcript search returned 4 chunks, 1 candidate videos
INFO  [TIMER] stage2_vector_search 15.234 ms backend=hnsw group=0 top_k=10 hits=6 expr=True
INFO  [TIMER] stage2_vector_search 14.567 ms backend=hnsw group=1 top_k=10 hits=4 expr=True
INFO  TwoStageFusion: alpha=0.506 genre=Thời sự transcript_chunks=4 visual_frames=10
INFO  [TIMER] fusion 4.321 ms strategy=TwoStageFusionStrategy results=10
```

---

## 2. Hướng dẫn xác minh Log

### 2.1 Pattern Log thực thi thành công

Khi `TwoStageFusionStrategy` thực thi thành công, bạn sẽ thấy các dòng log theo thứ tự:

```
[TIMER] request_received          ← FastAPI nhận request
Stage 1: transcript search returned N chunks, M candidate videos
[TIMER] transcript_candidate_search X.XXX ms  ← E5 encoding + Milvus search
[TIMER] stage2_vector_search      Y.YYY ms backend=hnsw group=0 top_k=K hits=H expr=True
TwoStageFusion: alpha=A.AAA genre=GENRE transcript_chunks=N visual_frames=M
[TIMER] fusion                    Z.ZZZ ms strategy=TwoStageFusionStrategy results=R
[TIMER] total_request             T.TTT ms strategy=two_stage_fusion_strategy status=ok results=R
```

**Các chỉ số cần kiểm tra:**
- `expr=True` → Stage 2 đã dùng bộ lọc scalar của Milvus (Stage 1 tìm được ứng viên)
- `expr=False` → Stage 1 rỗng, Stage 2 chạy toàn cục
- `alpha` trong [0.0, 1.0] → Trọng số động hợp lệ
- `results > 0` → Pipeline đã tạo ra kết quả
- `total_request < 2000ms` → Nằm trong giới hạn timeout 2 giây

### 2.2 Pattern Log khi Fallback

**Stage 1 rỗng (không có text query hoặc không khớp transcript):**
```
INFO  TwoStageFusion: Stage 1 empty, falling back to visual-only
```

**Stage 2 rỗng (các video ứng viên không có frames khớp):**
```
INFO  Stage 2 returned 0 results, falling back to global visual search
INFO  TwoStageFusion: Stage 2 empty, using transcript-only with frame mapping
```

**Cả hai stage đều rỗng:**
```
INFO  TwoStageFusion: Stage 1 empty, falling back to visual-only
```
(Kết quả sẽ có `confidence=0.0`)

### 2.3 Kiểm tra nhanh hệ thống

```bash
# Kiểm tra strategy đã được đăng ký chưa
curl http://localhost:8000/api/strategies | python -m json.tool | grep -A3 "two_stage"

# Kiểm tra transcript search đã bật chưa
curl http://localhost:8000/api/health | python -m json.tool
```

---

## 3. Hướng dẫn xử lý sự cố

### 3.1 Lỗi cú pháp Milvus `expr`

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|----------|
| `expr failed: syntax error` | `video_id in [...]` bị sai cú pháp | Kiểm tra dấu ngoặc kép chưa escape trong video_id |
| `expr failed: field video_id not found` | Schema collection không khớp | Xác nhận field `video_id` tồn tại trong collection Milvus |
| `expr failed: value type mismatch` | video_id chứa ký tự đặc biệt | Chỉ cho phép video_id dạng chữ-số + underscore |
| Kết quả rỗng với `expr=True` | Không có frames khớp video_id ứng viên | Kiểm tra video_id từ Stage 1 có tồn tại trong collection `video_frames` không |

**Debug chuỗi expr:**
```python
# Thêm tạm thời vào data_provider.py dòng ~309:
logger.info("DEBUG stage2_expr: %s", stage2_expr)
```

### 3.2 Các lỗi thường gặp

| Triệu chứng | Nguyên nhân | Giải pháp |
|-------------|-------------|-----------|
| `Stage 1: transcript search returned 0 chunks` | Collection transcript chưa được index | Chạy `python scripts/index_transcripts.py --sample-root ../AIC2026_sample` |
| `sentence-transformers not found` | Thiếu thư viện | `pip install sentence-transformers>=2.2.0` |
| `Transcript chunk search is not available` | `TRANSCRIPT_CHUNK_SEARCH_ENABLED=false` | Đặt thành `true` trong `.env` |
| `fusion_and_temporal() exceeded 2.0s` | Strategy bị timeout | Kiểm tra độ trễ Milvus; giảm `stage1_top_k` hoặc `top_k` |
| `alpha=0.000` cho mọi query | Transcript scores luôn < 0.05 | Kiểm tra model E5 đã load đúng chưa; xác nhận transcript chunks có điểm có ý nghĩa |
| Kết quả chứa frames từ video không phải ứng viên | Stage 2 expr không được áp dụng | Xác nhận `expr=True` trong log; kiểm tra đường dẫn CAGRA (CAGRA không hỗ trợ expr trực tiếp) |

### 3.3 Lưu ý riêng cho CAGRA

CAGRA (GPU brute-force) không hỗ trợ bộ lọc `expr` của Milvus. Khi `VECTOR_SEARCH_BACKEND=cagra`:
- Stage 2 chạy tìm kiếm CAGRA toàn cục
- Kết quả được lọc bằng Python: `hits = [h for h in hits if h["video_id"] in stage1_video_ids]`
- Cách này kém hiệu quả hơn nhưng vẫn đúng về mặt chức năng
- Log sẽ hiển thị `expr=False` ngay cả khi Stage 1 tìm được ứng viên

---

## 4. Chạy bộ Test Suite

```bash
cd remote-server

# Smoke test nhanh (không cần dependencies)
python scripts/test_two_stage_fusion.py --smoke

# Bộ unit test đầy đủ (mocked, không cần Milvus/PostgreSQL)
python scripts/test_two_stage_fusion.py --unit -v

# Tất cả tests bao gồm async DataProvider tests
python scripts/test_two_stage_fusion.py -v

# Chạy với pytest (nếu đã cài)
pytest scripts/test_two_stage_fusion.py -v
```

### Tổng kết độ phủ Test

| Lớp Test | Số test | Kiểm tra gì |
|----------|---------|-------------|
| `TestDynamicAlpha` | 8 | Tính toán α, giới hạn, mapping genre |
| `TestFusionMath` | 3 | Fusion có trọng số, trường hợp biên α=0/α=1 |
| `TestOCRBoost` | 3 | OCR boost +5%, giới hạn 1.0, không khớp |
| `TestTemporalBoost` | 3 | Temporal boost +5%, ranh giới tolerance |
| `TestFallbackPaths` | 3 | Stage 1 rỗng, Stage 2 rỗng, cả hai rỗng |
| `TestMilvusExprFormatting` | 4 | Xây dựng chuỗi expr |
| `TestResultStructure` | 2 | Tuân thủ schema, thứ tự sắp xếp |
| `TestRankingBuilders` | 3 | Logic ranking transcript/visual |
| `TestDataProviderTwoStageAsync` | 3 | Stage 1 được gọi/bỏ qua, inject expr |
| `TestScenarioA_HappyPath` | 1 | End-to-end happy path |
| `TestScenarioB_VisualHeavy` | 2 | Truy vấn thiên visual, α giảm |
| `TestScenarioC_ZeroTranscript` | 1 | Truy vấn vô nghĩa fallback |
| `TestScenarioD_OCRAndTemporal` | 1 | Multi-step temporal + OCR |
| **Tổng** | **37** | |
