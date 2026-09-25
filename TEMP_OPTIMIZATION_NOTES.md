# Readiness checkpoint — 2026-09-25

App is running: web `http://127.0.0.1:3000`, remote server port 8000, proxy port 8001. Preprocessing remains stopped; Docker is up. Proxy `.env` is now set to LOCAL (ignored file) and `/api/health` confirms it. This fixes the earlier ZIP-mode launch mismatch. Reload the web page before checking.

- Before the config fix, logs showed 1,643 frame requests / 1,488 HTTP 502s, including retries; 739 were S01-V010. Organizer ZIP range requests were HTTP 206, so corruption is not established. After the fix, L23_V021 and M06_V001 spot images returned HTTP 200 in 2.4 s and 2.7 s; S01-V010 returned cached HTTP 200. Too few checks to call S/M fully resolved. User also reports the results page stays laggy after search even when search itself is quick; L is slower than the old system, S is very slow/missing, and some M cards are missing. Treat page/render lag separately from search latency. Linear full-video decode is unverified.
- Hardware snapshot after proxy restart: 17/25 GiB RAM available, 25 MiB swap; GPU 3.2/16.3 GiB at 7%. Earlier ZIP-mode snapshot was 14 GiB available and GPU 5.6 GiB.
- Browser previously passed 100-result search, all 100 cards, representative L/M/S selected-picture playback and frame submissions; N010-V002 picture and source-PTS-ms submission passed. N010-V001/002/003 original random seeking failed. No N data published.
- Resume checkpoints: N vectors 180/291 (`verification/n_resume_2026-09-25.log`); playback 17/288 (`verification/playback_resume_2026-09-25.log`); card audit 29/291 (`verification/n_images_resume_2026-09-25.log`); M/S source maps 143/316. FLAT exact self-search 316/316 vs HNSW 314/316; p95 290 vs 270 ms; default unchanged.

Next: reload and quickly check representative L/M/S cards in LOCAL mode; investigate any remaining 502/load issue; then resume staged audits and N playback work. Do not exclude media from the pre-fix 502s. Full gates: [SYSTEM_READINESS_PLAN.md](SYSTEM_READINESS_PLAN.md).
