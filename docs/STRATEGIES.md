# Writing a Search Strategy

A search strategy is one Python file, dropped into
`local-client/local-backend/app/strategies/` (mirrored to
`remote-server/app/strategies/` for production). The backend auto-discovers
it — no endpoint or `DataProvider` changes needed.

## Minimum viable strategy

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

`name`, `description`, `author`, `version` must be non-empty — that's what
makes discovery register it, using the file stem as `strategy_id`. Restart
the backend (or let `--reload` pick up the `.py` change) to see it in the
frontend dropdown.

## Four data channels, one API

```python
hits = await context.retrieve(channel, query, top_k=100)
```

| Channel | Data | How |
|---|---|---|
| `raw.semantic` | Raw keyframe embeddings | PE-Core vector search |
| `subtitled.semantic` | Subtitled keyframe embeddings | PE-Core vector search |
| `transcript.lexical` | Transcript text | PostgreSQL full-text |
| `transcript.semantic` | Transcript text | E5 + Milvus vector search |

`transcript.semantic` / `subtitled.semantic` only work in `LOCAL`/`SERVER`
mode — see [ARCHITECTURE.md](ARCHITECTURE.md#known-limitations-worth-knowing-before-a-demo).

A strategy owns its own fusion, temporal matching, and reranking — call
`context.retrieve()` once or many times, in parallel or in rounds, and
combine results however the idea needs. Near-duplicate filtering and
result-page hydration happen automatically around `run()`; a strategy just
returns `context.results(hits)`.

## Temporal (multi-step) queries

Each entry in `context.query_groups` is one step; its position in the list
you build is the temporal order:

```python
rankings = await asyncio.gather(*[
    context.retrieve("raw.semantic", group["query"], top_k=200)
    for group in context.query_groups
])
return context.results(my_temporal_algorithm(rankings, context.query_groups))
```

## Rules

- Never construct SQL, touch a DB/Milvus collection, or hold credentials —
  everything goes through `context`.
- Don't store per-request state on `self` (the strategy instance is shared
  across concurrent requests).
- Bump `version` when prompt, retrieval, or fusion behavior changes.

## Promoting to production

Copy the file verbatim into `remote-server/app/strategies/`, restart
`remote-server`, select it in the frontend. No other change needed —
`DataProvider` handles the DB connections transparently on both sides.

## Full reference

The complete `SearchContext` API (`parse_json`, `keyframes`,
`request_more_results`, `frame_embeddings`), the hit-shape contract, and a
worked multi-source example with an LLM query planner + RRF fusion:
[archive/strategy_v2.md](archive/strategy_v2.md) ·
[archive/strategy_template_v2.md](archive/strategy_template_v2.md).

Notes on hybrid reranking ideas and transcript-chunk search specifically:
[archive/reranking_hybrid_notes.md](archive/reranking_hybrid_notes.md) ·
[archive/search_by_transcript.md](archive/search_by_transcript.md).
