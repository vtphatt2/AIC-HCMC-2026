# Readiness checkpoint — 2026-09-25

Work stopped at the user's request. Remote server, proxy and frontend are stopped; Docker databases remain up. No N generation was published. RAM after stopping: ~19 GiB available; 26 MiB swap used. Branch `zip-decode/fix` is clean at `14cccbe`.

- Browser passed search (100 results), loading all 100 cards, and selected-picture playback plus frame submissions for L21-V008, M09-V028 and S01-V005 (underscore lookup resolved to canonical hyphen ID). N010-V002 selected picture and source-PTS millisecond submission passed. Chrome random seeking fails on original N010-V001/002/003; they are not excluded.
- Commit `14cccbe` fixes LOCAL proxy media routing for every lot and includes five focused proxy tests. Five proxy tests and ten existing frame HTTP tests passed. Full local suite remains pending. `local-client/local-backend/.env` was restored to its original ZIP settings; proxy browser testing required temporary `ENV_MODE=LOCAL`, `REMOTE_SERVER_URL=http://127.0.0.1:8000`, and frontend API URLs on ports 8001. Those launch settings are not persisted.
- Resume checkpoints: N vectors 180/291 (`verification/n_resume_2026-09-25.log`); N playback copies 17/288 (`verification/playback_resume_2026-09-25.log`); N card audit 29/291 (`verification/n_images_resume_2026-09-25.log`); M/S source maps 143/316 (earlier checkpoint). Stages are resumable; no N data is published.
- Search comparison: FLAT exact self-search 316/316; HNSW 314/316. Proxied warm p95 was 290 ms vs 270 ms across tested top-100 queries; default unchanged pending recall/ranking decision.
- No source corruption/exclusion established. Organizer: S01 video metadata JSON names use hyphens; N KIS/QA positions use source PTS milliseconds; N is excluded from TRAKE.

Next: if continuing, resume N copies and verify browser seeking; finish staged vectors/cards and M/S picture checks; run full suites; settle search backend; validate submission and cross-artifact consistency; publish only after checks pass. Full release gates: [SYSTEM_READINESS_PLAN.md](SYSTEM_READINESS_PLAN.md).
