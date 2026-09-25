# Current readiness — 2026-09-25

The readiness fixes pass 172 remote, 56 local and 48 frontend tests plus the
frontend build. N maps and all picture derivatives use the same verified decoder
setting; map hashing is stat-cached. A second global fix uses the source MP4
duration for N: nominal frame/FPS duration was short by up to 70 seconds.

All 298 N videos completed full replay. Seven source-bound one-thread profiles
pass complete verification. Another 32 videos differ across one-/four-thread
decoding, but every same-setting four-thread repeat is exact; those variations
are harmless under the unified policy and their temporary blocks were cleared.
No organizer corruption or new exclusion is established. Official N001-N010 and
N031-N040 redownloads matched and redundant copies were removed. KIS/QA use
verified source-PTS milliseconds; TRAKE excludes N.

All 21 source/result ZIPs and the 614-video structural/live audit pass. The
291-video duration-aware N stage safely adopts unchanged source-verified vectors
and recomputes the rest. Seven recovery videos are revalidated. The first
five N result ZIPs (153 videos/45,518 vectors) pass packaging validation; the
first also passed all-row ingest checks. M/S sample maps, exposed-image audit
and one-job playback conversion run. Remaining: N derivatives, M/S semantics,
image/playback and browser/load checks, indexes/exports and publication. Memory
has about 18 GiB available with negligible swap use; the card cache is disk-backed.
Details and evidence: [SYSTEM_READINESS_PLAN.md](SYSTEM_READINESS_PLAN.md).
