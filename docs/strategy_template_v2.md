# Strategy V2 Template

> Copy template này để thêm strategy vào backend đang chạy [`strategy_v2.md`](strategy_v2.md).

## Bắt buộc

Một strategy chỉ cần metadata và `async run(context)`. Backend tự gọi `run`, áp timeout,
validate và trả response. Không override `search()` hoặc tự kết nối database.

```python
from app.strategies.base_strategy import BaseStrategy


class RawVisualV1(BaseStrategy):
    name = "Raw Visual v1"
    description = "Search raw keyframes with the user query."
    author = "Your Name"
    version = "1.0"

    async def run(self, context):
        query = context.query_groups[0]["query"]
        hits = await context.retrieve("raw.semantic", query)
        return context.results(hits)
```

## Template multi-source có system prompt

```python
import asyncio
from pydantic import BaseModel, Field

from app.strategies.base_strategy import BaseStrategy
from app.strategies._fusion import rrf


class QueryPlan(BaseModel):
    raw: str
    subtitled: str
    keywords: list[str] = Field(default_factory=list)
    spoken: str


async def frames_for_chunks(context, chunks):
    batches = await asyncio.gather(*[
        context.keyframes(c["video_id"], c["start_time_ms"], c["end_time_ms"])
        for c in chunks
    ])
    return [
        {**frame, "channel": c["channel"], "item_id": c["chunk_id"], "evidence": [c]}
        for c, frames in zip(chunks, batches) for frame in frames
    ]


class MultiSourceV1(BaseStrategy):
    name = "Multi-source v1"
    description = "Plan four queries and fuse their rankings."
    author = "Your Name"
    version = "1.0"

    SYSTEM_PROMPT = """
    Rewrite the request into: raw visible scene, subtitled scene,
    exact transcript keywords, and semantic spoken content.
    Return only the requested JSON structure.
    """

    async def run(self, context):
        user_query = context.query_groups[0]["query"].strip()
        if not user_query:
            raise ValueError("Query must not be empty")

        plan = await context.parse_json(
            system_prompt=self.SYSTEM_PROMPT,
            user_input=user_query,
            response_model=QueryPlan,
        )

        raw, subtitled, lexical, semantic = await asyncio.gather(
            context.retrieve("raw.semantic", plan.raw, top_k=200),
            context.retrieve("subtitled.semantic", plan.subtitled, top_k=200),
            context.retrieve("transcript.lexical", " ".join(plan.keywords), top_k=100),
            context.retrieve("transcript.semantic", plan.spoken, top_k=100),
        )

        lexical, semantic = await asyncio.gather(
            frames_for_chunks(context, lexical),
            frames_for_chunks(context, semantic),
        )

        fused = rrf([raw, subtitled, lexical, semantic], key="frame_id")
        return context.results(fused)
```

Không cần LLM thì bỏ planner và dùng raw query hoặc parser Python.

## API được cấp

| API/data | Dùng để |
|---|---|
| `context.query_groups` | Raw queries và temporal offsets |
| `context.top_k` | Số kết quả cuối request cần |
| `context.video_genre` | Genre filter hiện tại |
| `await context.retrieve(channel, query, top_k=...)` | Lấy ranked hits từ một channel |
| `await context.parse_json(...)` | Optional LLM structured parsing |
| `await context.keyframes(video_id, start_ms, end_ms)` | Lấy keyframes trong interval |
| `rrf(rankings, key="frame_id")` | Fusion rankings khác scale |
| `context.results(frame_hits)` | Hydrate/validate final output |

## Tùy biến trong `run`

Strategy được quyền:

- dùng zero/one/many system prompts hoặc không dùng LLM;
- query một hay nhiều channel, song song hoặc nhiều vòng;
- tự chọn per-channel `top_k`;
- tự fusion, temporal matching, rerank và fallback;
- định nghĩa helper/schema riêng trong cùng file.

Không truyền SQL/DB/collection/credentials hoặc lưu per-request state trong `self`.

## Temporal query

Mỗi query group giữ một ranked list riêng; vị trí trong `rankings` chính là temporal step:

```python
rankings = await asyncio.gather(*[
    context.retrieve(
        "raw.semantic",
        group["query"],
        top_k=200,
    )
    for group in context.query_groups
])
return context.results(my_temporal_algorithm(rankings, context.query_groups))
```

## Khi nào phải sửa backend?

Chỉ khi strategy cần data/model/engine chưa có channel. Khi đó backend thêm một channel vào
`DataProvider.retrieve`; strategy vẫn dùng cùng API.

## Checklist

- Có `name`, `description`, `author`, `version`, `async run(context)`.
- Transcript chunk giữ ID/interval; chỉ gọi `keyframes()` khi cần frame candidates.
- Không cộng trực tiếp raw score khác scale.
- Output qua `context.results()` hoặc đúng canonical format.
- Có một fake-context test kiểm tra query, channel và ranking.
- Bump version khi đổi prompt, parser, retrieval hoặc fusion behavior.
