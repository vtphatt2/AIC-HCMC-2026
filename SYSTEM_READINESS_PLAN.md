# System readiness: reliable search and display

## Goal and operating rules

Release a system people can use end to end: search L/M/N/S, see the correct
source picture, play and seek the matching video, and create valid submissions.
Passing unit tests or structural counts alone do not establish readiness.

Preserve original archives, canonical IDs, and working L/M/S sampling, vectors,
and behavior. Source frame identity, PTS, time base, and checksum are authoritative;
images, embeddings, cards, and playback files are derived. Reuse a derivative only
when its source/settings fingerprint and required identity checks still match;
rebuild the affected dependency chain when they do not. Stage and validate before
publication, and retain rollback data. First release covers server and proxied
clients; standalone N ZIP clients are deferred.

Diagnose patterns across affected videos before choosing a fix. Prefer a verified
shared fix when the same cause applies; use a video-specific profile only for a
verified edge case. If media appears defective, compare a fresh organizer download
and test suitable processing before exclusion. Exclude its data only when the
source problem remains reproducible and cannot be fixed without assumptions.

Organizer rule: N videos are VFR. Submit source PTS × 1000 milliseconds for KIS
and QA; reject N in TRAKE. Keep L/M/S submissions in frames. Organizer acceptance
of the format remains unverified.

## Current checkpoint — 2026-09-25

Work is paused. At the last local check, backend and frontend were not running and
no real client session had been verified. No new N generation was published.

| Area | Status |
|---|---|
| Source inventory and integrity | Complete: 21 source ZIPs pass CRC; 614 videos are inventoried. All 298 N videos passed independent replay and timing audit. Official N001 and N031 archive redownloads matched local copies; no organizer corruption or new exclusion is established. |
| N timeline and sampling | Complete: source PTS/time-base maps and configurable two-second presentation-time selections are staged for 298 N videos (89,795 pictures). Selections deduplicate source IDs and reuse valid scene boundaries without rerunning scene detection. N031 non-increasing timestamps are explicitly omitted while source IDs remain stable; N selection never relies on nominal frame/FPS seeking. |
| Shared processing policy | Implemented: source maps, selected images, embeddings, and playback use the same verified decoder setting. N duration comes from MP4 timing. |
| N vectors and result packages | In progress: duration-aware main stage paused at 170/291 videos. Five unpublished result ZIPs pass packaging checks (153 videos/45,518 vectors); the first also passed every ingest row and exact vector comparison. Seven source-profile videos remain release-blocked pending complete derivatives. |
| N cards and playback | In progress: exposed-card audit paused at 19/291 videos. Playback validation paused at 17 copies across two partitions (13/288 and 4/145). |
| M/S picture identity | In progress: source maps paused at 143/316 videos. Pilot source-picture and PE-Core checks passed; full coverage remains. |
| Search recall | Open: HNSW missed two exact self-matches in 316 M/S queries; FLAT found all 316. Compare real client recall, ranking, and latency before changing the default. |
| Structural and live records | Complete for current artifacts: all 21 source/result ZIPs and 614 videos were audited; exports and live PostgreSQL/populated Milvus records agree structurally. Picture semantics, playback, client flow, and final candidate publication remain open. |
| Memory | During background processing, 17–18 GiB was available with negligible swap. Memory under real client load is unmeasured. |

## Required work, in priority order

1. **Prove a real L/M/S user flow.** Start required services and use the real
   client to search, scroll, open result cards, inspect the source picture, and
   play/seek the matching video. Repeat through the proxy. Fix user-visible
   correctness or availability failures before expanding low-impact checks.
2. **Settle search behavior.** Compare HNSW and FLAT on representative searches;
   verify expected IDs, ranking, and top-100 recall in the client, then measure
   end-to-end latency. Preserve the existing L/M/S baseline; assess added N
   candidates separately.
3. **Finish N derivatives from checkpoints.** Complete the 291-video vector
   stage using verified adoption where source pictures, selection rows, scenes,
   model, and preprocessing match. Recompute missing or mismatched vectors only.
   Complete all exposed-card checks and prepare browser-compatible 720p H.264,
   yuv420p playback copies (libx264, veryfast, CRF 23, two encoder threads by
   default). Preserve presentation timing and source-to-playback origin; validate
   duration, seeks, PTS, and picture alignment before serving by HTTP Range.
4. **Finish source-picture checks.** Complete deterministic M/S checks across all
   316 videos, including selected pictures and scene boundaries. For every
   generated or repaired N vector, verify its source picture. Check each N video's
   start, middle, end, and known timestamp discontinuities; audit every exposed
   selected card. Record sampled versus exhaustive coverage clearly.
5. **Complete display metadata and proxy behavior.** Support both organizer
   metadata archive naming conventions. Resolve underscore/hyphen aliases only
   when unambiguous and preserve canonical IDs. Refresh M/S titles and links only
   from authoritative metadata without re-embedding; N metadata/link absence is
   expected. Do not prefer unverified YouTube alignment (including S01-V005/
   S01-V011). Forward timing,
   playback origin, availability, frame IDs, and media Range requests through the
   proxy.
6. **Finish timeline and submission handling.** Keep explicit frame IDs and
   presentation times in the versioned timeline API; expose submission units and
   playback origin without breaking older clients. Store units in versioned
   sidecars while preserving organizer CSV columns. Make modal selection, manual
   edits, review pictures, and CSV round-trips unit-aware. Convert legacy N frame
   rows only with verified maps; flag unresolved rows. Keep L/M/S neighbor fill at
   ±15 frames; use verified N pictures near ±500 ms with bounds and deduplication.
   Test strict numeric parsing, QA quoting, and N TRAKE rejection.
7. **Complete the all-artifact audit and publish consistently.** The machine-
   readable manifest must cover every M/N/S source and derivative, including
   unchanged data, and detect missing, extra, or duplicate items. Cross-check ZIP
   entries/CRC, scenes, selected IDs, PTS/checksum maps, embedding rows and model
   settings, result ZIPs, NumPy exports, PostgreSQL/vector indexes, cards/caches,
   playback copies, and submission round-trips. Check picture identity separately
   from counts. Validate a candidate generation before atomically publishing it;
   verify indexes and exports after publication, then clear only evidenced blocks.
   Exercise legacy and versioned artifacts, interrupted/resumed processing, and
   selective invalidation after a settings change.
8. **Exercise real use under load.** Run two clients during warming, scrolling,
   playback, and submissions. Bound background jobs; use a 32 GiB card-cache budget
   with safe misses and eviction. Keep conversions off the request path. Check
   deadlocks, broken images/seeks, OOM, and sustained swap growth. Targets: warm
   top-100 search p95 ≤1 s and cached first
   viewport ≤2 s locally; measure WAN separately.

## Completed shared fixes to preserve

- Ready-queue final rescan, warmer waits outside cache locks, and protected
  metadata/generation invalidation have regression coverage.
- Decoder settings and source-map fingerprints bind maps, images, vectors, and
  playback. N duration uses the MP4 track timeline; ingest buffering is bounded.
- Safe vector adoption requires matching source pictures, selected rows, scenes,
  model/preprocessing settings, and stored vectors. It avoids full recomputation
  only when those checks pass.

## Release gates and deferred scope

- Search IDs, pictures, timelines, playback seeks, and submitted timestamps agree.
- Cards are readable and correctly associated; media is browser-compatible,
  source-aligned, and works through Range and proxy requests.
- ZIP entry IDs/offsets/sizes and CRCs; scenes; vector shape, finiteness, and
  normalization; exports; metadata; and populated indexes agree. Find missing,
  extra, and duplicate records. Legacy sessions and versioned artifacts both work.
- Any excluded video has a reproducible, documented source defect after archive
  comparison and suitable processing checks. Keep originals and report exceptions.
- Organizer acceptance is unverified; no competition submission is included.

Defer model/ranking redesign, OCR/transcript expansion, broad optimization, and
standalone N ZIP clients until correctness and user-visible readiness are achieved.
