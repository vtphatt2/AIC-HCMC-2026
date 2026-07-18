# Search by Transcript — Topic-Based Chunking & Vector Search

## Tổng quan

Tính năng tìm kiếm transcript bằng **vector embedding** + **phân loại chủ đề tự động**, thay thế hoàn toàn phương pháp keyword-based cũ.

```
Query tiếng Việt → multilingual-e5-small encode → 384-dim vector
  → Dự đoán topic (Ẩm thực / Công nghệ / ...) từ 14 chủ đề
  → Milvus HNSW search trong collection transcript_chunks, filter theo topic
  → PostgreSQL hydrate raw_text
  → Trả về ranked chunks kèm topic, timestamp, text
```

---

## 1. Cài đặt

```bash
cd remote-server
pip install -r requirements.txt   # đã có sentence-transformers>=2.2.0
```

Kiểm tra `.env`:
```env
TRANSCRIPT_CHUNK_SEARCH_ENABLED=true
TRANSCRIPT_MODEL_ID=intfloat/multilingual-e5-small
TRANSCRIPT_MODEL_DEVICE=cpu        # hoặc cuda nếu có GPU
TRANSCRIPT_VECTOR_DIM=384
MILVUS_TRANSCRIPT_COLLECTION=transcript_chunks
```

---

## 2. Index dữ liệu transcript

### 2.1 Dry-run (kiểm tra trước)

```bash
cd remote-server
$env:PYTHONIOENCODING='utf-8'
python scripts/index_transcripts.py --sample-root ../AIC2026_sample --dry-run
```

In script này in ra danh sách chunk mẫu (video, topic dự đoán, khoảng thời
gian, đoạn text) kèm dominant topic mỗi video, và tổng số chunk/video sẽ index
— không ghi gì vào DB.

### 2.2 Index thực tế

```bash
# Lần đầu — tạo collection + nạp toàn bộ
python scripts/index_transcripts.py --sample-root ../AIC2026_sample

# Cập nhật lại (xóa cũ trước)
python scripts/index_transcripts.py --sample-root ../AIC2026_sample --clear-existing
```

Script in tiến độ, số chunk/video đã upsert vào PostgreSQL + index vào Milvus,
và phân bố topic sau khi xong.

---

## 3. Khởi động

```bash
# Terminal 1: Backend
cd remote-server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
cd local-client/frontend
npm run dev
```

Server log sẽ hiện `Transcript chunk search service ready` và liệt kê
`transcript_fusion_strategy` trong danh sách strategy đã discover.

---

## 4. Sử dụng API

### Endpoint

```
POST /api/search/transcript
Content-Type: application/json

{
  "query":        "cách nấu phở bò",    // required
  "top_k":        20,                    // optional, default 100, max 1000
  "topic_filter": ""                     // optional: "Ẩm thực" | "Công nghệ" | ...
                                         // "" = auto-predict từ nội dung query
}
```

### Response

```json
{
  "results": [
    {
      "chunk_id":      42,
      "video_id":      "L03_V001",
      "topic":         "Ẩm thực",
      "start_time_ms": 30000,
      "end_time_ms":   70000,
      "text":          "cách nấu phở bò truyền thống Nam Định...",
      "score":         0.952
    }
  ],
  "total":             15,
  "execution_time_ms": 234
}
```

### Ví dụ curl

```bash
curl -X POST http://localhost:8000/api/search/transcript \
  -H "Content-Type: application/json" \
  -d '{"query":"cách nấu phở bò","top_k":5}'   # thêm "topic_filter":"Kinh tế" để ép topic thủ công
```

---

## 5. Sử dụng UI

Vào http://localhost:3000, chọn tab **Transcripts** (xanh lá).

```
┌──────────────────────────────────────────────────────────────────┐
│ [Search transcript chunks...]  Top K [20] Topic [Auto ▼] [Search Transcripts] │
└──────────────────────────────────────────────────────────────────┘
```

| Thành phần | Chức năng |
|---|---|
| **Search input** | Nhập query tiếng Việt hoặc Anh. Enter để search |
| **Top K** | Số chunk kết quả (1–200). Default 20 |
| **Topic** | **Auto** = backend tự dự đoán. Chọn cụ thể để ép filter |
| **Search Transcripts** | Thực thi search |
| **Topic badge** | Màu riêng cho 14 chủ đề (cam = Ẩm thực, xanh dương = Công nghệ...) |
| **Timestamp** | `00:30 – 01:10` — click để mở YouTube tại giây đó |
| **Score** | Xanh ≥80%, vàng ≥60%, đỏ <60% |

---

## 6. 14 Chủ đề

| Chủ đề | Màu badge | Mô tả |
|---|---|---|
| Ẩm thực | Cam | Nấu ăn, món ăn, nguyên liệu |
| Công nghệ | Xanh dương | AI, robot, chip, phần mềm |
| Du lịch | Cyan | Địa điểm, khách sạn, tour |
| Thể thao | Đỏ | Bóng đá, điền kinh, giải đấu |
| Giáo dục | Tím | Bài giảng, trường học, thi cử |
| Kinh tế | Vàng đậm | Giá cả, thị trường, doanh nghiệp |
| Sức khỏe | Xanh lá | Bệnh viện, thuốc, dinh dưỡng |
| Giải trí | Hồng | Phim ảnh, âm nhạc, gameshow |
| Thời sự | Xám | Tin tức, chính trị, xã hội |
| Văn hóa | Indigo | Nghệ thuật, lễ hội, truyền thống |
| Đời sống | Teal | Gia đình, mua sắm, sinh hoạt |
| Môi trường | Xanh lá đậm | Khí hậu, ô nhiễm, thiên nhiên |
| Giao thông | Vàng | Đường xá, kẹt xe, cầu đường |
| Pháp luật | Nâu | Tòa án, luật, quy định |

---

## 7. Strategy tích hợp

Strategy **Transcript Fusion v1** trong **Frames mode** tự động dùng transcript chunk search:

```
Frames mode → Strategy: Transcript Fusion v1
  Semantic: "phóng viên"        → visual PE-Core search
  Text:     "bản tin thời sự"   → transcript chunk vector search
  Genre:    Thời sự             → filter theo genre
  ↓
RRF fusion (visual + transcript chunks + OCR)
  ↓
Ranked frames kèm confidence
```

Chọn strategy này từ dropdown trong Frames mode để kết hợp visual + transcript.

---

## 8. Genre filter trong Frames mode

Genre của mỗi video được tính tự động từ dominant chunk topic khi chạy `index_transcripts.py`.

Trong Frames mode, dropdown **Genre** cho phép lọc video theo thể loại:
- **All** — không filter
- **Ẩm thực** — chỉ search trong video nấu ăn
- **Thời sự** — chỉ search trong video tin tức
- ...

Genre filter hoạt động ở cả visual search (Milvus `video_id in [...]`) và strategy fusion (dynamic weight tuning).

---

## 9. Kiến trúc dữ liệu

### Milvus: `transcript_chunks`

| Field | Type | Mô tả |
|---|---|---|
| chunk_id | INT64 (PK) | ID tự tăng |
| video_id | VARCHAR(64) | Mã video |
| topic | VARCHAR(64) | Chủ đề dự đoán |
| start_time_ms | INT64 | Thời gian bắt đầu chunk |
| end_time_ms | INT64 | Thời gian kết thúc chunk |
| vector | FLOAT_VECTOR(384) | Embedding từ multilingual-e5-small |

Index: HNSW, COSINE metric, M=16, efConstruction=256

### PostgreSQL: `transcript_chunks_metadata`

| Column | Type | Mô tả |
|---|---|---|
| chunk_id | BIGINT (PK) | Khớp với Milvus chunk_id |
| video_id | VARCHAR(64) | Mã video |
| topic | VARCHAR(64) | Chủ đề |
| start_time_ms | BIGINT | |
| end_time_ms | BIGINT | |
| raw_text | TEXT | Nội dung đầy đủ của chunk |

### PostgreSQL: `videos.genre`

| Column | Type | Mô tả |
|---|---|---|
| genre | VARCHAR(64) | Thể loại chính của video (dominant topic) |

---

## 10. Troubleshooting

| Vấn đề | Giải pháp |
|---|---|
| `Transcript chunk search is not available` | Chạy `index_transcripts.py` để nạp dữ liệu. Kiểm tra `.env` có `TRANSCRIPT_CHUNK_SEARCH_ENABLED=true` |
| `sentence-transformers not found` | `pip install sentence-transformers>=2.2.0` |
| `collection not found` | Backend tự tạo collection khi khởi động nếu `TRANSCRIPT_CHUNK_SEARCH_ENABLED=true` |
| Search chậm | Chưa có INVERTED index trên `topic` (Milvus 2.3.4 không hỗ trợ). Scalar filter vẫn hoạt động với default index |
| Windows console lỗi Unicode | Set `$env:PYTHONIOENCODING='utf-8'` trước khi chạy script |
| NumPy version conflict | `pip install "numpy<2"` nếu gặp lỗi `_ARRAY_API not found` |
