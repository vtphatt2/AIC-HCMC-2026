# Current readiness — 2026-09-25

Shared decoder recovery is implemented: source-bound profiles replace video-ID
exceptions, and maps/images/embeddings/playback use the same verified setting.
Complete replay is required; source/map changes invalidate verification.
All seven profiles pass complete independent replay (100,107 source frames).
These videos remain release-blocked until derivatives and publication pass.
No organizer corruption is established. Official N001-N010 and N031-N040
redownloads matched the originals; redundant downloads were removed.

Both affected archives have completed full replay. Main N staging now covers
291 nonblocked videos/87,618 selected pictures, resuming verified checkpoints.
All 1,244 previously recovered vectors/cards and four recovery playback copies
pass; N039-V001 adds 318 verified vectors/cards. Other recovery work continues. Full image/playback work and M/S source-map preparation continue.
All 21 source/result ZIPs and the 614-video live structural audit pass.

Tests: 141 remote backend, 54 local backend, 46 frontend and production build
pass; 29 focused shared-policy/timing/image/playback/staging tests also pass.
Remaining: complete N derivatives, all-video M/S sampled semantic checks,
browser/load validation, consistent indexes/exports and staged publication.
M/S title/YouTube refresh awaits authoritative organizer metadata.

Memory: about 19 GiB available, 2.3 GiB swap, no sustained memory pressure.
Jobs have logs and exit records in verification/running_jobs_2026-09-25.json.
Details and evidence: [SYSTEM_READINESS_PLAN.md](SYSTEM_READINESS_PLAN.md).
