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

Organizer metadata clarification: media-info entries for `S01-V0*.mp4` should be named `S01-V0*.json`; the supplied table incorrectly uses `S01_V0*.json`. Resolve underscore/hyphen aliases only when unique; preserve canonical IDs.

Organizer rule: N videos are VFR. Submit source PTS × 1000 milliseconds for KIS
and QA; reject N in TRAKE. Keep L/M/S submissions in frames. Organizer acceptance
of the format remains unverified.

## Current checkpoint — 2026-09-25

The remote server, proxy, and frontend are running on ports 8000, 8001, and
3000; preprocessing is stopped. The proxy reports `ENV_MODE=ZIP` and
`ZIP_MEDIA_SOURCE=remote`: its `.env` overrides shell variables, so the attempted
process-only LOCAL setting did not take effect. Fresh search returned 100
results. Logs show 1,643 frame-image requests and 1,488 HTTP 502 responses, with
739 failures for S01-V010; these include retries and are not unique-frame counts.
Organizer archive range requests returned HTTP 206, so this check does not show
archive corruption. RAM snapshot: 14 GiB available of 25 GiB, 25 MiB swap used;
GPU: 5.6/16.3 GiB, 6% utilization. No preprocessing or N publication.

| Area | Status |
|---|---|
| Real client flow | Passed: search returned 100; all 100 cards loaded while scrolling. L21-V008, M09-V028 and S01-V005 played from selected pictures; frame submissions matched canonical video/frame IDs. N010-V002 picture and submission used verified source PTS milliseconds correctly. |
| Proxy media fix | Implemented and focused tests passed when configured in LOCAL mode. Current running instance is ZIP mode because the local `.env` takes precedence over shell settings; media is fetched from organizer ZIP endpoints. Restart with actual LOCAL configuration before evaluating the proxy fix. Full local suite remains to run. |
| Current slow/missing cards | User reports S cards are extremely slow/unavailable and some M cards fail; L is slower than the prior system. Current proxy log confirms repeated 502 frame responses (many for S01-V010 and S01-V007; also M06_V001 and one L23_V021). Quick check only: cause may be the active ZIP/remote decode path and request volume; linear full-video decoding is not established. |
| N seeking | Open, user-visible: Chrome cannot seek in original N010-V001/002/003 (decode error); start playback works. A lossless remux experiment did not establish a repair and was removed. These are not excluded; prepare verified playback copies and retest random seeks. |
| Search | Open decision: FLAT returned all 316 exact M/S self-matches; HNSW missed 2. Real proxied warm top-100 p95: FLAT 290 ms, HNSW 270 ms; overlap was 95–100/100 across six text queries. Default not changed. |
| N derivatives | In progress, resumable: vectors 180/291, verified playback copies 17/288, selected-card audit 29/291. Existing stage has 89,795 selected pictures across 298 N videos. No N release/publication. |
| M/S source-picture audit | In progress at 143/316 source maps; pilot checks passed. |
| Sources and live data | Earlier structural audits passed for 21 ZIPs and 614 M/N/S videos; N001/N031 redownloads matched. No organizer corruption or new exclusion established. Structural counts do not prove image identity. |

## Required work, in priority order

1. **Keep the verified L/M/S flow and resolve N seeking.** L/M/S search, all 100
   cards, selected-picture playback, and frame submissions passed in the proxied
   browser. N010-V002 picture and PTS-millisecond submission passed; Chrome
   random seeking failed on N010-V001/002/003 originals. Prepare verified copies,
   then retest seeks and check representative N videos through the proxy.
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
