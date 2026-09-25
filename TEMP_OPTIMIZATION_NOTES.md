# Readiness checkpoint — 2026-09-25

Work stopped at the user's request. Remote server, proxy and frontend are stopped; Docker databases remain up. No data generation was published. RAM: ~19 GiB available after stopping.

- Browser verified search, all 100 cards, L/M/S playback and frame submissions. N010 picture and source-PTS-millisecond submission passed; Chrome random seeking fails on N010-V001/002/003 originals.
- Fixed LOCAL proxy routing so every lot's cards and Range playback use the search server. Focused proxy tests passed; full suite pending. Temporary env edit was restored.
- Resume stages: N vectors 180/291; playback copies 17/288; N card audit 29/291; M/S source maps 143/316. Resume from existing checkpoints; do not publish until validated.
- FLAT exact self-search found 316/316 vs HNSW 314/316. Real p95: 290 ms vs 270 ms; choose whether recall gain warrants default change.
- No source corruption/exclusion established. Organizer says S01 metadata filename uses hyphens; N KIS/QA use source PTS milliseconds; N TRAKE excluded.

Next: finish N copies and verify Chrome seeking; rerun full proxy/client suites; decide search backend; finish picture identity and submission checks; publish only a validated consistent generation. Details and remaining release gates: [SYSTEM_READINESS_PLAN.md](SYSTEM_READINESS_PLAN.md).
