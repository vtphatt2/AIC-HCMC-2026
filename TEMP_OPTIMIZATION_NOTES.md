# Current readiness — 2026-09-25

The pulled readiness fixes passed 160 remote, 56 local and 48 frontend tests plus
the frontend build. Review found and fixed a global decoder-policy mismatch:
normal N maps used four threads while embeddings/playback used two. Every
derivative now uses its source map's verified decoder setting, with provenance
and cache invalidation; H.264 encoding remains bounded to two threads. Repeated
full frame-map hashing is now stat-cached. Thirty-nine focused tests pass.

All 298 N videos completed full replay. Seven source-bound one-thread profiles
pass complete verification. Another 32 videos differ across one-/four-thread
decoding, but every same-setting four-thread repeat is exact; those variations
are harmless under the unified policy and their temporary blocks were cleared.
No organizer corruption or new exclusion is established. Official N001-N010 and
N031-N040 redownloads matched and redundant copies were removed. KIS/QA use
verified source-PTS milliseconds; TRAKE excludes N.

All 21 source/result ZIPs and the 614-video structural/live audit pass. Remaining:
finish archive replay, rebuild invalidated N derivatives, finish M/S semantics,
browser/load checks, consistent indexes/exports and staged publication. Memory is
about 19 GiB available with zero swap use and no pressure; the 25 GiB card cache
is disk-backed and the RAM JPEG cache is capped at 128 MiB.
Details and evidence: [SYSTEM_READINESS_PLAN.md](SYSTEM_READINESS_PLAN.md).
