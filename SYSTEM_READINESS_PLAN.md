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

No offline preprocessing job is running. The last known web stack uses ports
8000, 8001, and 3000. The idle machine has about 20/25 GiB RAM available and no
swap pressure. No N publication has occurred.

| Area | Status |
|---|---|
| Real client flow | Passed for released data: a top-100 search produced 54 L/M/S groups. Activating every group rendered 1,215 cards; all 125 requested images returned HTTP 200, with no broken/unavailable cards and 6 ms UI response. Search-result and nearby S cards now use exact frame IDs. |
| Display/search lag | Fixed shared causes: LOCAL proxies now forward search, scoring, and media instead of loading a second model; context is centered on the best hit and bounded to about 24 cards rather than thousands across long S videos. Five uncached top-100 searches took 56–246 ms under playback conversion. |
| Weak-network cards | Constrained connections request 640px WebP cards while retaining the exact video/timestamp/frame identity. Browser simulation requested 10/10 small cards and no originals; representative L/M/S cards were 31–61 KB instead of 78–211 KB originals. |
| N submissions | Live web round-trip passed: N010-V001 source frame 100 became verified source-PTS position 4199 ms, its exact WebP review card returned HTTP 200, reload preserved both identities, and the temporary session was removed. Review/manual-entry labels distinguish N milliseconds from L/M/S frames. |
| N seeking | Shared fallback verified: all N originals require validated copies. N010-V001/002/003 copies passed exhaustive source-picture/output-PTS validation and Chrome playback at start, midpoint, and near end through the proxy. |
| Tests | Remote 183/183, local 65/65, frontend 49/49, production build, and real browser flows pass. The local suite declares deployment mode so LOCAL `.env` does not alter standalone tests. |
| Search | FLAT is the release default: it returned all 316 exact M/S self-matches while HNSW missed 2. Real proxied warm top-100 p95 was 290 ms for FLAT and 270 ms for HNSW, with 95–100/100 overlap across six text queries. HNSW remains selectable. |
| N derivatives | Resumable: all 298 videos have a verified vector generation across main/recovery stages. Thirteen decoder-sensitive videos have verified one-thread profiles; four new playback failures need the same profile/recovery (`N027-V002`, `N029-V001`, `N029-V003`, `N030-V003`). All failures retain exact PTS/counts and match the known checksum-only threading pattern; no organizer corruption is established. Playback has 90/298 current validated copies. Only 32 current exact-image manifests remain; earlier 116-video evidence must be rebuilt and checked for JPEG plus WebP. Seventeen videos remain release-blocked through final validation. |
| Release safety | Final audit now fails when any N video lacks a validated browser copy. Quarantine release is atomic, evidence-bearing, and refuses changed or still-blocked entries. Add an offline evidence-bound clearance for the 17 recovered blocks before index ingestion; clearing them does not expose N until final release. Previous full source-CRC and live-index audits took 97 s and 176 s. Removed 16.5 GB/123 stale playback files from superseded signatures; current copies and active work were preserved. |
| M/S source-picture audit | In progress at 143/316 source maps; pilot checks passed. |
| Sources and live data | Earlier structural audits passed for 21 ZIPs and 614 M/N/S videos; N001/N031 redownloads matched. No organizer corruption or new exclusion established. Structural counts do not prove image identity. |
| Organizer metadata | The supplied B2 archive passed ZIP CRC/JSON checks and contains 304 M plus 12 S entries. Both archive naming conventions and unique underscore/hyphen aliases are supported and tested, so erroneous `S01_V0*.json` names map to canonical `S01-V0*` IDs. PostgreSQL titles/YouTube IDs now match all 316 entries with timing fields unchanged. All 12 authoritative S MP4s passed Chrome start/middle/end seeks; S uses MP4 first because YouTube timeline alignment is not yet proven. N metadata/YouTube absence is expected. |

## Required work, in priority order

1. **Finish N derivatives from checkpoints.** Recover the four pending decoder
   profiles and affected derivatives. Resume selected-card checks and the
   playback-copy job. Preserve the passing L/M/S and N010 client flow.
2. **Settle search behavior.** Compare HNSW and FLAT on representative searches;
   verify expected IDs, ranking, and top-100 recall in the client, then measure
   end-to-end latency. Preserve the existing L/M/S baseline; assess added N
   candidates separately.
3. **Complete browser-compatible N playback.** Continue the 720p H.264/yuv420p
   job (libx264, veryfast, CRF 23, two encoder threads). On this 16-logical-CPU
   host use four low-priority workers (`nice 8`); concurrency changes scheduling,
   not artifacts. Preserve VFR timing and source-to-playback origin; validate all
   source pictures, output PTS, duration and seeking before HTTP Range serving.
   If the release window is limited to three hours, first benchmark 480p
   H.264/yuv420p NVENC on ordinary, text/detail, night/motion, and discontinuity
   sources. Accept it only when exhaustive picture/PTS checks, Chrome seeking,
   readable medium/large details, and about 220 remaining copies/hour aggregate
   pass. Keep the 90 valid 720p copies and use exact source-resolution JPEGs when
   small details must be inspected.
4. **Finish source-picture checks.** Complete deterministic M/S checks across all
   316 videos, including selected pictures and scene boundaries. For every
   generated or repaired N vector, verify its source picture. Check each N video's
   start, middle, end, and known timestamp discontinuities; audit every exposed
   selected JPEG and weak-network WebP card. Record sampled versus exhaustive
   coverage clearly.
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
   clear recovered blocks only from complete offline evidence, ingest HNSW and
   FLAT with bounded batches and one final flush/load, then verify indexes and
   exports before the atomic release.
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

With a three-hour release window, defer the unfinished exhaustive M/S semantic
audit after recording its 143/316 checkpoint: released L/M/S has passed current
search, card, submission, and playback checks with no known functional defect.
Prioritize decoder recovery, N playback/cards, candidate publication, and the
final live browser/submission audit.
