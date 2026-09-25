# System readiness plan: reliable L/M/N/S search and display

Source: user-approved readiness plan, 2026-09-24. Status values: pending, in progress,
complete, blocked. Evidence is recorded here as work progresses. Generated data is
excluded from code commits; existing uncommitted work must be preserved.

## Goal and compatibility

Make search results, source images, playback and submission exports reliable.
Preserve source ZIPs, canonical IDs, working L/M/S sampling, model and ranking.
Source frame identity, PTS and checksum are authoritative; thumbnails, embeddings
and playback copies are independent derivatives. Centralize configurable policies,
version new metadata, preserve legacy defaults, fingerprint sources and settings,
and stage intentional replacements before publication. First release targets the
server and proxied clients; independent N support on standalone ZIP clients is deferred.

## Tracking

| Task | Status | Evidence / next action |
|---|---|---|
| Save plan and maintain short readiness note | complete | Plan and short note updated on 2026-09-25. |
| Ready queue: rescan after producer completion | complete | Deterministic final-publish race test passes. |
| Warmer: idle waits outside cache locks, including recovery | complete | Concurrent lock/idle regression test passes. |
| Protect verified metadata from Phase 1 and policy invalidation | complete | Preflight guard test proves no artifact deletion on a protected generation. |
| Staged generations and deterministic processing regression tests | in progress | Independent source-fingerprinted N stage; no publication. Pulled changes add immutable candidates, locks, artifact digests and interrupted-metadata recovery; complete remote tests pass. More end-to-end interruption tests remain. |
| Reuse decoded source frame/PTS/time-base/checksum maps | complete | All 298 N videos received complete independent replay. Seven existing one-thread profile maps pass exact replay (100,107 frames). Among the remaining maps, 32 differ only across one-/four-thread settings; every fresh four-thread repeat reproduces its authoritative four-thread map exactly. All frame IDs and PTS agree. Cross-setting variation is ignored because the unified pipeline uses the map setting. |
| Isolated organizer redownload of N031-N040 | complete | Fresh 10,924,602,184-byte ZIP from the supplied URL has the same SHA-256 as local (`f66be6b1bb0189e86f6b8e2df3b704b8e5b85a9d8a82ed806873eb01005551aa`); all 30 entry records match and fresh full CRC passes. No source or derivative replacement is warranted. |
| N configurable two-second presentation sampling, retain valid selections, deduplicate, no scene cap | complete | 298 N videos / 89,795 staged pictures; stage manifest records identities and omissions. |
| Recompute all selected N vectors with existing PE-Core preprocessing | in progress | N031-V003's 309 and the N007-V002/V003/N009-V002 recovery generation's 935 vectors pass exhaustive selected-source checks. All derivatives use their authoritative map's decoder setting. Main stage paused at 170/291 completed videos; unchanged vectors are adopted only after exhaustive selected-source, row, scene and encoder-setting validation. |
| N031 non-increasing timestamp exclusions | complete | All 298 N timelines pass exhaustive numeric/source-ID audit; only N031-V001/V002/V003 omit entries (four each). All six adjacent retained pictures pass source PTS/checksum and visual continuity checks, with matching exclusions in validated playback copies. See n_timeline_audit.json and n031_discontinuity_picture_audit.json. |
| Sequential selected 640px thumbnails and full-resolution inspection | in progress | All 309 N031-V003 and 935 combined N001 recovery selected JPEGs/cards pass source identity and exposed-card comparison (N007 max RGB MAE 2.90; N009 2.85). Main exposed-card audit paused at 19/291 completed videos. |
| Validate staged archives, exports, metadata and indexes | in progress | Recovered N001-N010 ZIP stages all 30 videos/9,181 vectors; full ZIP CRC, ingest dry run and NumPy export agree on every ID, source-PTS millisecond timestamp and vector (max component error 5.96e-8). Offline audit temporarily bypassed the public blocklist; live publication is still prohibited. Superseded 29-video candidate and export files were removed after validation. |
| Reusable validated 720p H.264/yuv420p MP4 playback | in progress | Recovery copies passed exhaustive source PTS/checksum and output PTS/codec checks. Unified-policy conversion paused at 13/288 early and 4/145 late completed copies; interrupted scratch copies were removed. N015-V001 and N019-V003 exposed the shared two-thread/four-thread mismatch; the global policy fix covers both. Publication is pending. |
| Preserve VFR timing and playback origin independently of source pictures | in progress | All 298 N MP4 track durations agree with decoded presentation spans within 240 ms; fingerprinted per-video evidence is in n_duration_audit.json. Nominal frame/FPS duration was wrong by >10 s for 15 videos (worst 70.37 s). New staged metadata, timeline API and ingest use the source track duration; old artifacts keep legacy defaults. Playback origin remains a separate field. |
| Proxy media, timing and availability | in progress | Local HTTP tests preserve timing capabilities, explicit timeline IDs, Range headers and playback bytes; metadata forwarding now includes the tunnel bypass header. Live two-client browser verification remains. |
| Organizer archive naming, unambiguous ID aliases, M/S title/link refresh | in progress | Both archive naming forms and ambiguity-safe aliases implemented/tested. Local organizer metadata contains L only; M/S source titles/links remain blocked pending authoritative metadata, with no embedding change needed. N absence is expected. |
| Verify YouTube alignment before preferring it | blocked | S01-V005 and S01-V011 have no authoritative YouTube links in available metadata; do not invent or prefer a link. |
| Compatible timing capabilities and versioned explicit-ID timeline | in progress | Version 2 API and frontend parser tested; end-to-end proxy test pending. |
| Versioned submission sidecars, unchanged CSV columns | in progress | Implementation and focused frontend tests pass; organizer acceptance unverified. |
| Unit-aware modal, edits, review pictures and CSV round-trips | in progress | Submission handler integration verifies modal frame conversion, manual ms edits, QA quoting, sidecars, CSV round-trips, legacy L/N migration and unresolved N rows. Fixed rounded endpoint rejection and concurrent add/rename/delete races. Browser review-image checks remain. |
| Neighbor fill and strict numeric/QA quoting tests | complete | L/M/S and N unit tests, strict parser and quoted QA tests pass. |
| Machine-readable audit of every M/N/S source/artifact | in progress | Full source/result structural manifest and live PostgreSQL/vector-index manifest written with zero reported structural issues. Picture and playback audits remain. |
| Exhaustive structural and cross-artifact audit | in progress | All 21 source ZIPs and 21 result ZIPs pass full CRC; 614 source/result videos and 221,760 M/N/S rows align. Packaged vectors match processing output exactly; NumPy export differs by at most 1.19e-7 per component. All live M/N/S PostgreSQL and populated Milvus rows agree. |
| Picture identity checks | in progress | Every newly repaired N vector passed; main N pass paused at 170/291. A deterministic M/S semantic audit inventories all 316 videos/2,991 sampled selected rows, including scene boundaries. The M01_V001 and S01-V001 pilots pass checksum-verified source seeking and fresh PE-Core comparison (minimum cosine 0.999987); CPU source maps paused at 143/316, and full PE-Core sampling follows the N GPU job. |
| Thumbnail/cache/playback/submission artifact validation | pending | Check generation, images/dimensions, compatibility/timing/seeks, units/round-trips. |
| Legacy/versioned/interrupted/selective-invalidation tests | pending | Preserve rollback before derivative replacement. |
| Every N start/middle/end/discontinuity and all exposed thumbnails | pending | Release quarantine only with evidence. |
| Unchanged L/M/S vectors and baseline search | in progress | Exact stored-vector queries found two HNSW top-100 self-recall misses among 316 M/S videos; FLAT found all 316. Assess a FLAT default for remote search with end-to-end latency and ranking checks before changing behavior. Mixed-data ranking remains separate. |
| Two-client load during warming and resource bounds | pending | Local warm top-100 p95 ≤1s, cached first viewport ≤2s; 32 GiB card budget, no deadlock/OOM/swap growth; WAN separately. |
| Maintenance publication, indexes/exports, quarantine release | pending | Consistent validated generation and rollback. |
| Exceptions, final evidence and focused commits | in progress | Only the seven previously verified one-thread profile videos remain release-blocked pending derivatives/publication. Temporary blocks for 32 stable current maps were cleared after exact same-setting replay. Original sources are retained; organizer corruption is not established. |

## Shared decoder recovery

- **Complete (implementation):** replaced video-name decoder exceptions with a
  source-bound profile registry. Source maps, images, embeddings and playback
  share the verified decoder setting. Profiles require full independent replay;
  stale source/map identities and mismatched evidence are rejected. Atomic writes
  and a file lock preserve concurrent registrations. See
  [DECODER_RECOVERY.md](docs/DECODER_RECOVERY.md).
- **In progress (data):** complete staged publication for the seven already
  verified profiles. The other 32 cross-setting differences have exact current-map
  repeats and require no recovery. Decoder reproducibility does not by itself
  establish intended picture identity or release readiness.
- **Complete (compatibility check):** all seven profiles resolve to their existing
  source map paths. Verified vectors and exact images remain reusable. Playback
  copies require the new source-map/decoder fingerprint and are revalidated.
- **Complete (tests for shared policy):** 39 focused decoder/timing/image/playback/
  staging/quarantine tests pass. The pulled tree passed 160 remote tests, and the
  first full run with cached map identities passed 161; a final full run follows
  the current changes. Local backend has 56 passing tests; frontend has 48 and a
  passing production build.

## Audit requirements

- Source ZIPs/video entries: inventory, CRC, readability, uniqueness, offsets/sizes;
  reuse prior evidence only if tied to unchanged fingerprints.
- Scenes: source, ordered bounded ranges, completeness/settings; investigate decoder
  versus container counts.
- Selections/maps: unique ordered frame identities, scenes, timestamps, embedding
  rows; fingerprints, time bases, coverage, discontinuities and exclusions.
- Embeddings: dimensions, counts, finite normalized vectors, model/preprocess provenance,
  picture association. Results and numpy must agree in layout, metadata and values.
- PostgreSQL/populated indexes: live counts, extras/missing/duplicates, metadata,
  timestamps and vector agreement.
- Images/playback: readable correctly associated cards, dimensions/cache generation;
  complete compatible video copies with correct timing, seeking and pictures.
- Submissions: units/migration, correct review pictures, CSV/ZIP round-trips/packaging.

Matching counts do not prove picture identity. New N sampling is expected to add
approximately 88,000 pictures (~21% more total vectors, ~430 MiB raw vectors/copy).
Recover every N video where possible; quarantine only documented unresolved defects.
Organizer acceptance remains unverified until reference evidence or an acceptance
result exists. No competition submission is authorized by this plan.

## Findings and changes

- 2026-09-24: Sources, GPU, existing outputs and previous verification evidence are
  present. Initial note reports 295 quarantined N videos and two reproducible races.
  No repository AGENTS.md found (database volume directories are not readable).
- 2026-09-24: User instructed us to exclude a video if organizer data is bad
  and cannot be repaired without assumptions. N031-V003 first appeared to meet
  that condition: PTS 4881870 had three checksums under earlier software
  passes. The 10.9 GB ZIP passed CRC and the fresh official copy is identical,
  so transport corruption is not established. No N generation was published.
- 2026-09-24: At the user's request, downloaded the organizer's N031-N040 ZIP
  into a separate verification directory and compared it with the local source.
  Both full SHA-256 hashes are
  `f66be6b1bb0189e86f6b8e2df3b704b8e5b85a9d8a82ed806873eb01005551aa`;
  all 30 ZIP entry records match and the fresh copy passes full CRC. The
  N031-V003 video bytes are identical (entry CRC `fa480330`). Re-downloading
  or reprocessing from the same source cannot, by itself, fix a decoder issue.
  Existing source and shared derivatives were untouched;
  the redundant verification download was removed after recording the result.
- 2026-09-24: A new four-thread decode matched the saved N031-V003 source map
  at 14,946/14,947 frames but yielded an all-zero selected picture at frame
  12165 (PTS 4881870/10000). The saved map instead reused frame 10978's
  checksum. Two independent full one-thread decodes agree at all 14,947
  frames, including a plausible target picture (`c3dba613`). The
  independently extracted target JPEG is byte-for-byte the earlier one-thread
  JPEG. The hardware decoder's JPEG has the same ~6.09 RGB mean absolute
  difference from the target as it does from the preceding one-thread picture,
  consistent with a decoder/color-conversion offset; it does not establish a
  different picture. The working hypothesis is a software decoder threading
  problem, not proven organizer corruption. A separate one-thread source map
  and embedding path are being verified. Evidence is in
  `zip_embeddings/verification/n031_v003_identity_audit.json`.
  N031-V003 remains search-quarantined and release-blocked until the full
  source, selected vectors, thumbnails and playback pass. The other 297 N
  videos continue through isolated verification.
- 2026-09-24: Full structural audit passed across all 21 M/N/S source ZIPs,
  all 21 result ZIPs and all 614 videos, with 221,760 packaged selected rows.
  The export's one-ULP float32 renormalization was previously reported as a
  mismatch; the corrected audit accepts up to 1e-6 component drift and found
  a maximum of 1.19e-7. TransNet's scene conversion intentionally omits
  1,252 transition pictures in 1,211 short gaps; no selected picture lies in
  a gap. These are recorded separately from source-picture identity, which
  remains unverified for most videos. The subsequent live audit found zero
  issues across all M/N/S PostgreSQL records and both populated Milvus
  indexes; SCANN is unpopulated. See the two readiness audit JSON manifests.
- 2026-09-24: N031-V003's new map matches an independent complete one-thread
  decode at every one of 14,947 frame IDs, PTS values and checksums. All 309
  selected source JPEGs and 640px cards passed identity/visual checks; card
  RGB MAE after resizing was at most 3.03. Its rebuilt 720p/yuv420p playback
  copy verified every one of 14,943 retained source pictures before encoding
  and all output frame PTS exactly. The target playback image matches the
  verified source. Four backward-PTS frames were omitted while preserving
  their original IDs; neighboring pictures passed visual checks. Embedding
  verification is running. The public block remains until the new vectors
  and all indexes/exports are published consistently.
- 2026-09-24: The wider N embedding pass found a selected-frame mismatch in
  N007-V002 at frame 535, source PTS 522020/10000. Four-thread source map,
  two-thread embedding decode and one-thread independent decode yielded
  distinct checksums `EA3273BD`, `56D2B881`, `C0EF871A` respectively;
  one-thread repeats agree. Hardware decoder also shows different picture
  content. The N001-N010 ZIP and its N007-V002 entry passed full CRC. Exact
  source-picture identity is unresolved, so the whole video is release-blocked
  while we assess it; organizer corruption is not proven. Other N videos
  continue through the isolated embedding audit. See
  `zip_embeddings/verification/n007_v002_identity_audit.json`.
- 2026-09-24: Following the user's archive-wide verification instruction,
  downloaded a fresh official N001-N010 ZIP (10,549,210,472 bytes) and
  compared it to the local original. Both full SHA-256 hashes are
  `5ed6e3a8f02889d8624ae47a35d24c949b4c9991512ef229607d309fc9ad0cf7`;
  all 30 entry records match and fresh full CRC passes. Thus a bad local
  download cannot explain N007-V002. The redundant 10.5 GB copy was removed
  after writing `zip_embeddings/verification/n001_archive_redownload_audit.json`.
  A separate one-thread full-frame PTS/checksum audit is running across all
  30 videos in that archive before any final exclusion decision.
- 2026-09-24: The N001-N010 replacement result ZIP has been staged offline
  from verified generations: 29 videos, 8,871 source-selected vectors,
  N007-V002 excluded. Full ZIP CRC, the existing ingest parser's dry run, and
  an independent NumPy export round-trip verify IDs, frame numbers, source-PTS
  milliseconds and vector values (max component error zero). The source
  archive-wide independent decoder audit has checked 16/30 videos with exact
  frame/PTS/checksum agreement and is continuing. No live data was changed.
- 2026-09-24: After adding source-picture alignment checks to offline playback
  conversion, the full remote backend suite passed 127 tests; local backend
  passed 51 tests, frontend passed 39 tests, and its production build passed.
  `git diff --check` passed. The user revoked the temporary stop request, so
  bounded staging and verification continue.
- 2026-09-24: The N001-N010 archive-wide one-thread audit found N007-V002's
  old four-thread map differs at 9/6,445 pixel checksums and N007-V003's at
  5/6,318, while every decoded frame ID and PTS matches. The V003 mismatch
  was discovered after its prior stage and replacement ZIP were made; V003 is
  now release-blocked and that ZIP is invalidated for publication. Separate
  one-thread maps and independent repeat decodes are being prepared for both
  videos. This is a decoder-setting problem under investigation, not evidence
  of organizer corruption. The fresh N001-N010 ZIP is already byte-identical
  to local source and has passed CRC, so no second download is needed.
- 2026-09-24: Separately fingerprinted one-thread maps for N007-V002 and
  N007-V003 passed independent complete replay: respectively 6,445 and 6,318
  frame IDs, source PTS values and pixel checksums agree exactly. Their
  isolated two-second selections contain 310 and 317 pictures with no PTS
  omissions. Selected full-resolution JPEG and served 640px card verification
  is running; embeddings and playback still need one-thread verification.
- 2026-09-24: The same complete N001-N010 audit detected N009-V002's old
  four-thread map differs at one of 18,169 pixel checksums, with every frame
  ID and PTS unchanged. It is now release-blocked. A separately fingerprinted
  one-thread map and repeat audit are underway; no organizer corruption has
  been established. Four previously prepared playback copies passed the new
  exhaustive source-picture pre-encode and output-timing checks. One bounded
  offline conversion job continues for the other nonblocked browser-incompatible
  N sources, skipping N010 originals already verified browser compatible.
- 2026-09-24: Full N001-N010 one-thread replay completed: 27/30 existing
  four-thread maps match on every frame ID, source PTS and checksum. The only
  differences are N007-V002 (9 pixels), N007-V003 (5) and N009-V002 (1);
  their separately fingerprinted one-thread maps all pass independent full
  replay. All 627 selected N007-V002/V003 source JPEGs and exposed 640px
  cards pass PTS/checksum and image association checks (maximum card RGB MAE
  2.90). Disputed frames viewed under one-, two-, four-thread and (for V002)
  hardware decoding show decoder-dependent moving-object placement. A nearby
  picture in unaffected N007-V001 also has CCTV compression ghosting, so
  visual artifacts alone do not prove these specific organizer videos are
  unusable. The three remain blocked pending complete playback/embedding
  verification. Archive-wide evidence supports a selective one-thread
  exception, not an unverified blanket change for all N videos.
- 2026-09-24: N007-V002/V003 and N009-V002 passed 935 selected one-thread
  PE-Core vectors and all 935 full-size JPEG/served-card comparisons in a
  combined isolated generation. A new offline N001-N010 result ZIP contains
  all 30 videos and 9,181 vectors; full ZIP CRC, existing ingest dry run and
  NumPy export round-trip agree on all IDs, source-PTS milliseconds and vectors
  (maximum component difference 5.96e-8). An audit-only empty blocklist was
  used for the parser/export checks; public blocks remain. The superseded
  29-video ZIP and large test export were removed. Main N manifest now excludes
  all four currently release-blocked videos, leaving 294/88,551; its image
  audit has started and embedding resumes after an M/S semantic pilot.
- 2026-09-24: Added deterministic M/S semantic audit spanning all 316 videos,
  with a planned 2,991 source-verified sampled rows near selected frame and
  scene boundaries. A pilot detected an audit-script assumption that an exact
  sparse map stored only target rows; the map also stores neighbors. The
  reader now selects requested IDs explicitly. This was a test harness issue,
  not an M source finding. The corrected M01_V001 and S01-V001 pilots pass
  checksum-verified source PTS/pixel identity and fresh PE-Core checks on 10
  and 11 selected pictures (minimum cosine 0.999987). The audit now reuses a
  fingerprinted S selected map and accepts fast source seeking only on exact
  PTS/checksum agreement, falling back to sequential decode otherwise. Main N
  embedding resumed at its verified markers; only one GPU model job runs.
- 2026-09-24: A local ZIP video lookup could resolve a compact alias to a
  release-blocked canonical ID after the initial access check. The resolved
  ID is now checked too; the targeted regression and all 51 local backend
  tests pass. Frontend's 39 tests and production build pass after the latest
  timing/submission changes. The complete remote backend run passed 132 tests.
- 2026-09-25: Resumed interrupted N embedding (61 saved videos), selected-card
  audit (24), full N031-N040 replay (21) and playback (18). Jobs now have
  detached log/exit-status records in verification/running_jobs_2026-09-25.json.
  The N031-V001/V002 source map contains retained frames sharing a PTS with an
  omitted frame. Verified sequential JPEG generation now selects decoded
  frame IDs for ambiguous timestamps and still requires exact PTS/checksums.
  All 610 selected cards for these two videos pass; adjacent retained pictures
  around all three N031 discontinuities also pass source and visual checks.
- 2026-09-25: Seven submission API integration tests cover source-frame/modal
  conversion, manual ms edits, versioned sidecars, QA quoting, CSV round-trips,
  legacy migration, unavailable maps, TRAKE rejection and concurrent clients.
  They exposed rounded endpoint rejection and lost concurrent updates; fixed
  both, including rename destination reservation and deletion during a save.
  Frontend now passes 46 tests and production build. Per-session serialization
  assumes one frontend server owns the local submission directory.
- 2026-09-25: Added immutable staged-metadata selection and strict resume-marker
  validation. A repaired map with unchanged source bytes now chooses a separate
  generation; complete marker settings and artifact hashes prevent reuse of
  stale or changed vectors. Legacy markers are adopted only after matching
  metadata/settings and complete vector validation. Three regression tests pass.
  The already-running N process will adopt this guard on its next restart.
- 2026-09-25: Full N031-N040 replay found N039-V001 differs at four of 18,081
  pixel checksums with exact frame IDs/PTS; it is release-blocked immediately.
  A separately fingerprinted one-thread source map and independent replay are
  being prepared. The archive's official redownload is already byte-identical,
  so no duplicate download is needed. No organizer corruption is established.
  Other videos continue through bounded processing. CPU-only sampled source-map
  preparation has also begun across all 316 M/S videos to support the later
  GPU semantic audit without loading a second model.
- The previous 640px cache warmer reached about 7.9 GiB in its 8 GiB soft
  limit. Host memory still had about 18 GiB available; most apparent use was
  reclaimable file cache. To honor the stop instruction and lower load, the
  warmer and the new staging/conversion jobs were stopped. Afterward host
  memory was about 20 GiB available, with low memory pressure and 277 MiB swap.
- Verification completed before stop: 115 remote backend tests passed,
  39 frontend tests passed, frontend production build passed, and targeted
  source/playback tests passed. Full M/N/S artifact and browser audits remain.
- Deferred: model/ranking redesign, OCR/transcripts, extensive optimization, standalone
  N ZIP clients, and competition submission.

- 2026-09-25: N031-N040 full replay completed with 27 exact current maps and
  three pixel-only disagreements: N039-V001 (4/18,081), N040-V002 (1/18,069),
  N040-V003 (2/18,078). N031-V003 uses its already-recovered map. The three
  newly affected videos were release-blocked immediately. N039-V001 and
  N040-V002 candidate one-thread maps independently replay exactly; N040-V003
  recovery also passed all 18,078 rows. No further redownload is needed for the already identical
  official archive. New recovery profiles preserve all previous map/generation
  names. Background jobs are being restarted from verified checkpoints to use
  the shared policy and refreshed 291-video main manifest.
- 2026-09-25: All seven registered decoder profiles now pass independent full
  replay, covering 100,107 source pictures. Comparing one-/four-thread maps
  identifies 23 differing pixel checksums; every frame ID and PTS agrees. These
  are shared decoder-setting symptoms, not confirmed organizer corruption.
  Full backend suite: 141 tests pass; focused shared-policy suite: 29 tests pass.
  Real migration checks preserve old map identities and four existing verified
  playback copies. See decoder_pixel_patterns.json and decoder_profile_migration.json.
- 2026-09-25: Recovery stage now contains N031-V003/N039-V001/N040-V002/V003
  (1,242 selected pictures). N039-V001's 318 vectors and all 318 source/cards
  pass; the remaining recovery vectors/images/playback continue. Then the same
  bounded queues resume the 291-video main stage and normal playback. Main image
  verification and M/S source-map preparation also continue with durable logs.
  Removed 320,715,208 bytes of abandoned unpublished scratch only after checking
  age and live process references; see scratch_cleanup_2026-09-25.json.

- 2026-09-25: Reconciled organizer guidance already carried in the handoff:
  N001–N100 are VFR, organizer frame_idx is approximate, KIS/QA may submit
  pts_time × 1000 milliseconds, and TRAKE excludes N. The implemented source-PTS
  policy matches this statement. Nominal frame/FPS disagreement is expected,
  not grounds for source exclusion. Decoder pixel disagreements at equal PTS
  remain a separate verification question. Exact statement and distinctions
  from finals DRES/preliminary CSV guidance are preserved in
  docs/ORGANIZER_TIMING_GUIDANCE.md. No acceptance or rounding convention is
  inferred beyond the supplied guidance.
- 2026-09-25: Reviewed the three pulled readiness commits. Their immutable
  staged candidates, exact-image verification, source-map-bound playback,
  source-PTS ingest/export and explicit frame URLs are structurally sound; the
  pulled tree passed 160 remote, 56 local and 48 frontend tests plus the frontend
  production build. A performance defect remained: normal N image and playback
  requests repeatedly hashed complete frame-map files. Source-map digests are now
  cached by path/inode/size/mtime/ctime, retaining replacement invalidation while
  avoiding repeated multi-megabyte reads. Cache-generation regression tests pass.
- 2026-09-25: Playback evidence exposed N015-V001 and N019-V003 with the same
  complete-row/exact-PTS but different-pixel pattern when the two-thread
  derivative decoder was compared with the four-thread authoritative map. This
  revealed a global policy inconsistency. Source maps, PE-Core inputs, exact
  images, cards and playback source decoding now share one source-bound decoder
  setting; the planned two threads still bound the H.264 encoder. Decoder
  provenance invalidates old normal derivatives and cache entries. The archive
  audit can atomically release-block every complete replay mismatch, while
  source-specific one-thread profiles remain reserved for independently verified
  unstable maps. Organizer VFR guidance remains handled separately through PTS:
  KIS/QA use source PTS milliseconds and TRAKE rejects N.
- 2026-09-25: Completed archive-wide replay across all 298 N videos. Thirty-two
  additional videos showed small pixel-only differences between one- and
  four-thread decoding, with zero frame-ID or PTS differences. Per the user's
  severity guidance, these were not treated as defects automatically. A new
  global stability gate repeated each authoritative four-thread map with four
  threads: all 32 matched every row and checksum exactly. Their conservative
  release blocks were cleared, no profiles were created, and four interrupted
  one-thread candidate maps were removed. N059-V002's earlier random-seek concern
  also passes full sequential replay. The seven older verified profiles remain;
  no organizer corruption or additional exclusion is established.
- 2026-09-25: Added source/map fingerprint checks to the saved same-setting
  decoder replay so a changed map cannot inherit old evidence. Added safe vector
  adoption across N metadata generations: selected rows, scenes, source
  checksums, model/preprocessing and stored vector bytes must validate before a
  new marker is written. This avoids needless PE-Core recomputation when a
  decoder-policy change altered only unselected pictures. Focused tests pass;
  the real 291-video N stage has adopted matching vectors and recomputed the
  first missing set. CPU-only M/S source maps and one-job playback conversion
  run concurrently with memory monitored; no live publication has occurred.
- 2026-09-25: An archive/ingest review found a global N duration defect:
  `num_frames / fps` can understate the source presentation length by 70 seconds.
  Across all 298 N videos, MP4 track duration agrees with the decoded first-to-
  last PTS span within 240 ms. New stage metadata records track duration in
  milliseconds, ingest uses it for N, and packaging checks it against the
  current MP4 and timeline. Existing vector rows are safely adopted because
  duration does not affect picture selection or PE-Core inputs. The 291-video
  stage resumed from checkpoints; the seven recovery videos were revalidated
  and adopted without recomputing identical vectors. Focused duration and
  packaging tests pass. An isolated N001-N010 candidate had previously passed
  all 30 videos/9,181 vectors; it was superseded by the duration-aware candidate.
  The latter passed ZIP CRC, all 30 video metadata/9,181 ingest rows, source
  duration and frame ID/timestamp checks, and exact normalized vector comparison
  (maximum component error zero). The obsolete isolated candidate was removed;
  no live archive or index was changed.
- 2026-09-25: Five duration-aware N result archives now pass packaging validation
  (153 videos/45,518 vectors). The main exposed-card audit is underway. Fifty-
  three old playback copies have pre-encode alignment markers but no source-map
  digest, so they cannot safely inherit the new signature. Conversion continues
  with two disjoint single-job partitions while host memory remains about 17–18
  GiB available and memory pressure reads zero. Each job still uses the planned
  720p libx264/yuv420p, veryfast, CRF 23 and two encoder threads.
- 2026-09-25: The ingest buffer setting contradicted its bounded-memory comment:
  it retained 120 batches of Python vector records before writing. It now keeps
  four batches (1,024 rows at the default 256-row batch size), lowering peak
  Python memory during the later full-index rebuild without changing record
  content. The 56 local-backend tests also pass after the shared timing change.
- 2026-09-25: Stored-vector baseline search across the first selected picture
  of every M/S video found two HNSW top-100 self-recall misses out of 316:
  M09_V013_000002 and S01-V001_000025. FLAT found all 316 at rank one; its
  database search p95 was about 58 ms under background processing. An expanded
  HNSW search still missed S01-V001_000025. The 75-query L/M/S comparison had
  median top-100 overlap 100 and minimum 91. This is a search-index quality
  issue, not evidence of source corruption. Assess a FLAT default after an
  end-to-end latency/client test; no live search setting was changed.
- 2026-09-25: User requested a temporary stop. Interrupted the N vector stage
  at 170/291 videos, M/S source maps at 143/316, N exposed-card audit at 19/291,
  and the two disjoint playback partitions at 13/288 and 4/145 completed copies.
  No processing or ffmpeg job remains. Removed four incomplete playback scratch
  files (601,391,904 bytes); kept completed validated derivatives and all source
  archives. Five unpublished N result ZIPs (153 videos/45,518 vectors) remain
  validated, with the first independently checked against every ingest row.
  This checkpoint leaves picture, playback, client, index/export and publication
  work in progress. Recent audit/ingest/packaging edits passed 17 focused tests;
  the last full remote/local/frontend suites passed 172/56/48 tests before those
  small edits. Resume from saved generations and logs, then rerun final suites.
