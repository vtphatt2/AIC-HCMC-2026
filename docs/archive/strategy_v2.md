# Strategy V2

> Contract đang chạy trong backend. Cách viết strategy:
> [`strategy_template_v2.md`](strategy_template_v2.md).

## Mục tiêu

Thêm một ý tưởng search mới chỉ bằng một file strategy. Strategy nhận raw query, có thể parse
bằng system prompt, tự chọn data, tự fusion/temporal/rerank và trả kết quả.

## Một API, bốn channel

```python
hits = await context.retrieve(channel, query, top_k=100)
```

| Channel ID | Data | Cách tìm |
|---|---|---|
| `raw.semantic` | Raw keyframe embeddings | PE-Core vector search |
| `subtitled.semantic` | Subtitled keyframe embeddings | PE-Core vector search |
| `transcript.lexical` | Transcript | PostgreSQL full-text; có thể đổi sang BM25 |
| `transcript.semantic` | Transcript | E5 + Milvus vector search |

Channel chỉ là string chọn pipeline, không phải API/class riêng. Không tạo class Raw,
Subtitled hay Transcript cho từng channel.

## Luồng request

```text
POST /api/search
  -> backend chọn strategy
  -> tạo SearchContext cho request
  -> await strategy.run(context)
       -> optional context.parse_json(...)
       -> một hoặc nhiều context.retrieve(...)
       -> optional context.keyframes(video_id, start_ms, end_ms)
       -> built-in/custom fusion và temporal logic
       -> context.results(...)
  -> validate + top_k + response
```

Frontend gửi raw query, không quyết định query thuộc data nào:

```json
{
  "strategy_id": "team_multi_source_v1",
  "config_id": "default",
  "config_overrides": {"event_weights": [1.0, 1.4]},
  "query_groups": [{"query": "người nói về giá xăng", "temporal_offset_ms": 0}],
  "top_k": 100,
  "video_genre": "All"
}
```

`config_id` selects a read-only backend preset. `config_overrides` is the local
frontend draft: the backend validates and merges it for this request only.

## SearchContext

Backend tạo một context mới cho mỗi request. Strategy không tự kết nối database.

```python
class SearchContext:
    def __init__(self, query_groups, top_k, video_genre, data_provider, parser=None, options=None,
                 config_id="default", config_revision=0, duplicate_threshold=None):
        self.query_groups = query_groups
        self.top_k = top_k
        self.video_genre = video_genre
        self._data_provider = data_provider
        self._parser = parser
        self.options = options or {}
        self.duplicate_threshold = duplicate_threshold

    async def retrieve(self, channel, query, *, top_k=None):
        # Paginated internally: each call for the same (channel, query, top_k)
        # fetches a fresh page on top of what was already returned, skipping
        # frame_ids already seen via exclude_frame_ids. Driven by
        # request_more_results() below — a strategy just calls retrieve()
        # again with the same args and gets the next page.
        return await self._data_provider.retrieve(
            channel,
            query,
            top_k=self.top_k if top_k is None else top_k,
            video_genre=self.video_genre,
        )

    def request_more_results(self) -> bool:
        """Advance to the next retrieval page for any channel/query that
        isn't exhausted yet. Returns False if nothing more is available."""
        ...

    async def frame_embeddings(self, frame_ids: list[str]) -> dict[str, list[float]]:
        """Batch-fetch raw vectors for frame_ids (request-scoped cache)."""
        ...

    async def parse_json(self, *, system_prompt, user_input, response_model):
        if self._parser is None:
            raise RuntimeError("Query parser is not configured")
        return await self._parser.parse_json(
            system_prompt=system_prompt,
            user_input=user_input,
            response_model=response_model,
        )

    async def keyframes(self, video_id, start_ms, end_ms, *, limit=20):
        return await self._data_provider.keyframes(video_id, start_ms, end_ms, limit=limit)

    def results(self, hits):
        return self._data_provider.results(hits)
```

`BaseStrategy.search()` tạo object này cho từng request rồi gọi `await strategy.run(context)`.
Các dependency có `_` là private; strategy chỉ dùng fields/methods public phía trên.

## Hit contract

Frame hit bắt buộc có `channel`, `item_id`, `video_id`, `frame_id`, `frame_number`,
`timestamp_ms`, `rank`, `score`.

Transcript chunk hit bắt buộc có `channel`, `chunk_id`, `video_id`, `start_time_ms`,
`end_time_ms`, `text`, `rank`, `score`; không bắt buộc có `frame_id`.
`chunk_id` là ID ổn định, interval nằm trong field riêng thay vì nhét vào chuỗi ID. Strategy
chỉ gọi `keyframes()` nếu muốn chuyển chunk thành frame candidates.

Kết quả cuối giữ format frontend hiện tại và thêm `evidence`; `results()` chịu trách nhiệm
hydrate metadata, dedupe, validate và sort.

## Lọc kết quả gần trùng (`duplicate_threshold`)

`BaseStrategy.search()` không gọi thẳng `run(context)` — nó gọi
`_run_with_filter(context)`, wrap quanh `run()`:

1. Chạy `run(context)` như bình thường, lấy `results`.
2. Nếu request có `duplicate_threshold` (frontend gửi qua `/api/search`,
   default 0.98), lấy embedding từng `frame_id` trong `results` qua
   `context.frame_embeddings(...)`, rồi lọc bằng
   `app/strategies/_similarity_filter.py::filter_similar_results()` — thuật
   toán greedy, giữ thứ tự rank, chỉ loại một result khi **toàn bộ** step
   tương ứng đã similar (cosine, có thể weight theo `event_weights`) với một
   result đã giữ.
3. Nếu số kết quả còn lại chưa đủ `top_k`, gọi `context.request_more_results()`
   (advance sang page tiếp theo của mọi `retrieve()` call chưa exhausted) rồi
   lặp lại `run(context)` — `retrieve()` tự dùng `exclude_frame_ids` nên
   không fetch lại frame đã thấy.

Strategy **không cần biết** cơ chế này tồn tại — không tự gọi
`frame_embeddings()`/`request_more_results()`, không tự filter duplicate.
Toàn bộ nằm trong `SearchContext`/`BaseStrategy`, strategy chỉ cần trả
`results` như thường; vòng lặp filter+refetch là trong suốt với `run()`.

## Code đặt ở đâu?

| Thành phần | File |
|---|---|
| `SearchContext`, `BaseStrategy` | `app/strategies/base_strategy.py` |
| Duplicate-result filter | `app/strategies/_similarity_filter.py` |
| Channel dispatch và keyframes-in-interval | `app/data_provider.py` |
| Raw/subtitled dùng chung `_search_visual()` | `app/data_provider.py` |
| Visual collections | `remote-server/app/db/milvus_client.py` |
| Transcript lexical | `remote-server/app/db/postgres_client.py` |
| Transcript semantic | `remote-server/app/services/transcript_search.py` |
| RRF helper | `app/strategies/_fusion.py` |

Class `Transcript` hiện có trong `transcript_index.py` chỉ parse/lookup transcript để hiển thị
theo timestamp; nó không phải V2 channel.

## Runtime hiện tại

- SAMPLE chạy `raw.semantic`, `subtitled.semantic`, `transcript.lexical`; chưa có local E5 index cho
  `transcript.semantic`.
- SERVER chạy đủ bốn channel sau khi ingest đúng collection/chunk index.
- `parse_json()` khả dụng khi backend có `GEMINI_API_KEY`.
- Bốn `duy_temporal_search*` đã dùng V2: mỗi group retrieve `raw.semantic`, rồi DP trong strategy.
- V1 nằm trong `app/archive_v1/strategies` và không được discovery/import.

## Không làm ở V2 vòng đầu

Không plugin/factory retriever, service locator, generic filters, public trace API,
global StrategyPlan, class riêng cho channel hoặc strategy tự chọn collection/database.

## Done khi

Một file strategy có thể dùng bất kỳ tổ hợp bốn channel, optional system prompt, schema-driven
Tune config, custom fusion/temporal/reranking và trả evidence mà không sửa endpoint hay DataProvider.
SAMPLE smoke test phải pass; SERVER cần smoke test lại trên máy có Milvus/PostgreSQL.
