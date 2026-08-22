# Brainstorm — Video-wise / Multi-detail Temporal Search

Not built yet. Captured 2026-08-22 from a design discussion, for someone
to pick up later — see [[project_aic2026]] memory (video-scoring plan)
for the immediate predecessor of this idea (per-frame second-phase
scoring, already shipped).

## Motivation

Every current strategy (`raw_visual`, `duy_temporal_search`,
`duy_multi_detail_search`, `multi_source`) ranks at the **frame** level —
"video" only exists as a UI grouping applied to already-ranked frames
(Video view). A frame-level top-k can miss a genuinely strong **video**
whose single best frame doesn't quite crack the global top-k, even though
several of that video's frames collectively provide strong evidence.

Two things already built make this tractable without much new machinery:

- **Per-frame event scoring** (`app/strategies/_event_scoring.py`,
  shipped 2026-08-22) — `score(frame) = Σ weight_i · cosine_sim(frame,
  query_event_i)`, computed identically regardless of whether the frame
  ever appeared in a ranking. This is exactly what phase 2 below needs.
- **"All frames of one video" lookup** — `numpy_vector_store.context_frames`
  / `GET /api/video/{id}/context-frames` and `/api/video/{id}` already
  query by `video_id` with no time bound.

## Shape: two phases

**Phase 1 — narrow to candidate videos** (cheap, coarse)

From the full corpus, shortlist a small set of candidate `video_id`s
likely to contain the answer, without fully scoring the whole corpus.

- **Option A (start here):** run the existing per-event `context.retrieve()`
  top-K (same oversampling `duy_multi_detail_search` already does), take
  the **union of `video_id`s** appearing across all M events' top-K. Zero
  new infrastructure — reuses what's already fast (~35ms/1000 results on
  FLAT).
- **Option B (only if A proves too coarse/expensive):** a precomputed
  per-video pooled embedding (mean or max over that video's frame
  embeddings), searched once per event for a genuinely video-level ANN
  pass. New index to build and keep in sync with ingest — real cost, not
  worth it unless profiling says A isn't good enough.

**Phase 2 — score each candidate video as a whole**

For each candidate video: pull *every* indexed keyframe (already-built
query shape, see above), score each against every event via
`event_weighted_scores`, then aggregate to one video score. **Aggregation
function is an open question** — options, not yet decided between:

- `max` — "this video's best moment" (closest to today's de-facto
  behavior, one strong frame carries the video)
- `top-N average` — more robust to a single-frame fluke, natural fit for
  "verify multiple frames were seen" (TRAKE-flavored)
- per-event `max`, then weighted-sum **across** events — "does this video
  have a good moment for detail 1 *and* a good moment for detail 2"
  (relaxes `duy_multi_detail_search`'s "same frame" requirement to
  "somewhere in this video")

Once a video wins, phase 2's already-computed per-frame scores answer
"which frame(s) to actually submit" directly — KIS needs the argmax frame
per event; TRAKE needs an ordered set (see below).

## The two variants asked about

**Video-wise temporal search** — phase 2 uses **ordered** reasoning:
`match_temporal`'s existing DP (`_duy_temporal_core.py`), restricted to
one video's own frame timeline instead of the whole corpus. Answers "does
this video, watched start to end, plausibly contain event1-then-event2-
then-event3 in order" — a *much* cheaper DP once phase 1 has narrowed the
video set (~200 frames for one video vs. tens/hundreds of thousands
corpus-wide).

**Multi-detail temporal search** — same phase 1, but phase 2 uses
`duy_multi_detail_search`'s RRF-style aggregation (unordered, several
details describing one frame) restricted to one video, instead of DP.

Both variants share phase 1 entirely and reuse existing fusion/DP code
unmodified for phase 2 — the new work is the video-narrowing step and the
video-level aggregation choice, not the scoring/matching primitives.

## Open questions to resolve before building

1. **Phase 1 candidate-set size.** Too few videos risks missing the
   answer if it's outside every event's individual top-K; too many
   balloons phase 2's cost (roughly `#candidates × avg_frames_per_video ×
   M events`).
2. **Phase 2 aggregation function** — the three options above interact
   directly with `event_weights`' existing semantics; picking one
   probably needs a few real queries tried against real data, not just
   picked on paper.
3. **New strategy_ids, or fold into existing ones?** The established
   convention here is separate files per variant (`duy_temporal_search`,
   `duy_temporal_search_0_5s`, `_5_10s`, `_10_20s` are already four
   distinct `strategy_id`s, not one strategy with a config flag) — so
   likely `video_wise_temporal_search.py` / `multi_detail_temporal_search.py`
   as new, separate strategies, not a mode switch on the existing ones.
4. **Duplicate filtering** (`_similarity_filter.py`) doesn't map cleanly
   onto a video-level result — a "video" isn't a fixed number of frames
   the way a temporal chain or a single fused frame is. Needs its own
   treatment, not yet designed.
5. **Latency budget.** `EXECUTION_TIMEOUT_SEC = 30s` is a hard ceiling.
   Phase 2's full-video-frame fetch + per-event cosine scoring, times the
   candidate-video count from phase 1, needs to comfortably clear that —
   worth a rough back-of-envelope check against real candidate counts
   before committing to an aggregation function that's expensive to
   compute.

## Reusable building blocks (nothing here needs reinventing)

| Need | Already exists as |
|---|---|
| Score one frame against M query events | `app/strategies/_event_scoring.py::event_weighted_scores` |
| All frames of one video, no time bound | `numpy_vector_store.context_frames` (local) / `query_frames_in_time_range` (remote), see `/api/video/{id}/context-frames` |
| Ordered multi-event matching (DP) | `app/strategies/_duy_temporal_core.py::match_temporal` |
| Unordered multi-event fusion | `app/strategies/_fusion.py::rrf` |
| Per-event weight config + UI (tuning page, step-count auto-sync) | `TEMPORAL_CONFIG_SCHEMA` / `event_weights` — reuse the exact field name, don't invent a new one (see [[project_aic2026]] for why that matters to the frontend) |
