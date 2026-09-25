# DONE

- M0: recorded `main` at `4a77cf9`; preserved existing uncommitted build/document files.
- M0: confirmed startup with `source local-client/local-backend/.venv/bin/activate` and no installs.
- M0: manual baseline passed on port 8011: ZIP mode, 193,508 vectors, 9 strategies; `raw_visual` returned 5 real candidates in 634 ms.
- M1: traced the critical frontend/backend/retrieval/render/media/transcript path and wrote `docs/agent/MVP.md`.
- Integration decision: current Codex CLI 0.157.0 is installed and authenticated; backend venv has no OpenAI/Codex/MCP package. Use bounded CLI structured output plus direct VORTA HTTP tool bridge for M2.
- M2: added validated Search Agent plans, bounded Codex CLI planner, `search_vorta` HTTP bridge, orchestration CLI and tests under `agent/`.
- M2 unit test: `source local-client/local-backend/.venv/bin/activate && python -m unittest agent.tests.test_search_agent -v` — PASS, 4/4.
- M2 real slice: two-event mango query -> Codex `duy_temporal_search` plan -> 3 real candidates with two ordered frame steps — PASS. Planning 9,062 ms; retrieval 687 ms; total 9,749 ms.
- Manual regression after agent run: `raw_visual` returned 3 real candidates in 43 ms — PASS.
- Syntax/import check: `python -m compileall -q agent` — PASS.
- M3: added one in-memory Search Agent job API (`agent/server.py`, `agent/state.py`), bounded Codex process cancellation, Next.js proxy, compact shared Search Agent panel, and optional startup in `scripts/start-local.sh`. Manual `/api/search` code was not changed.
- M3 live end-to-end: `POST /api/agent-search` via Next.js -> Codex -> VORTA returned 3 real temporal candidates for the mango query in 10,764 ms (9,059 ms planning; 1,704 ms retrieval). A train query returned 3 candidates in 7,291 ms.
- M3 concurrent manual search: `POST /api/search` returned 3 real candidates in 375 ms while the agent was planning. Two identical client starts shared one `query_id`; cancel returned `cancelled` and cleared results. Unit tests cover late-result suppression on reset/new query.
- M3 agent-off fallback: stopped agent service; Next.js agent proxy returned HTTP 503 while manual `/api/search` still returned 3 results in 195 ms.
- M3 checks: `local-client/local-backend/.venv/bin/python -m unittest discover -s agent/tests -v` — PASS 8/8; `npm run build` in `local-client/frontend` — PASS; `bash -n scripts/start-local.sh` and `git diff --check` — PASS. No tools or dependencies installed.
- M5: implemented bounded `inspect_candidate`, separate Codex Verify context, evidence-bound `MATCH/MISMATCH/UNKNOWN` checks, shared serialized queue/API, Next.js proxy, Verify buttons on manual/agent results, and shared status/checks panel. No DRES action was added.
- M5 tests: `local-client/local-backend/.venv/bin/python -m unittest discover -s agent/tests -v` — PASS 14/14; `npm run build` in `local-client/frontend` — PASS. Actual Codex Verify run with synthetic transcript returned transcript `MATCH`, visual `UNKNOWN`, overall `INSPECT` in 8,174 ms. No dataset host used.
- M6 local tests: two API clients shared one Verify queue; two distinct candidates ran serially; cancel/new official query suppressed stale results; mismatched old-query Verify requests returned HTTP 409; missing transcript and malformed output stayed UNKNOWN/error. Full suite later passed 18/18.
- M5/M6 API + Codex integration with synthetic evidence: `POST /api/agent/verify` returned 200; shared state reached `done` in 10,573 ms with two visual UNKNOWN checks and one transcript MATCH. No dataset host used.
- M7 partial replay (planning only, no retrieval): four representative queries chose `raw_visual`, `raw_visual`, `duy_temporal_search` (two correct event steps), and transcript `fuzzy`; planning times 8,246/8,845/8,666/14,265 ms.
- M8 available checks: final `python -m unittest discover -s agent/tests -v` through backend `.venv` PASS 18/18; Next.js production build PASS; Python compile/import, shell syntax and `git diff --check` PASS; `scripts/start-local.sh --help` PASS after making the script executable. No dependencies installed.
- M8 local proxy smoke with the real agent service and unavailable VORTA backend: production Next.js `/api/agent-verify` accepted a request, reached `done` with only visual UNKNOWN checks, and reset to idle; page `/` stayed HTTP 200. After stopping agent service, Verify proxy returned HTTP 503 while page `/` stayed HTTP 200. The UI now marks missing VORTA evidence explicitly.
- Added `docs/agent/USAGE.md` with existing-venv/Codex checks, one-command and manual startup, Windows, LAN/tunnel wiring, UI steps, smoke commands and failure recovery. Routed the two agent API paths through Next.js in `scripts/share-proxy.cjs` so tunnel users can reach them.
- Documentation/routing check: Node stdlib proxy smoke with fake frontend/backend returned the expected upstream for 6/6 paths (`/api/agent-search`, `/api/agent-verify`, manual search/media, submission, page) — PASS. No package installation.
- Final-round DRES: read `HD-ChungKet-2026.pdf` from `origin/zip-decode/fix`; added operator-triggered candidate submission from `/submissions`, server-only DRES session token, submit PIN, active task check, FPS-based KIS/Q&A timestamps, TRAKE frame-number format, duplicate guard, and explicit DRES result. Search/Verify Agents still never submit.
- DRES tests: `npm run build` PASS; `npm test` PASS 36/36; `node scripts/test-dres-smoke.cjs` PASS with fake DRES/VORTA (status, PIN, stale row/task, missing metadata, exact payload, duplicate guard and task rollover). No request was sent to BTC.
- Added [final-round setup and operator guide](../DRES.md). Existing CSV/ZIP export remains available.
- Pre-merge guard: reject `N` video DRES payloads until verified PTS timing is available. Frontend build PASS; `npm test` 36/36 PASS; Agent backend tests 18/18 PASS; fake DRES/VORTA smoke PASS; `git diff --check` PASS. No BTC request made.

# DOING

- Merge the Agent branch into `main` only; `zip-decode/fix` remains separate by operator choice.

# NEXT

- Deferred M4 test: prove real frame-image transport to Verify Agent with one candidate when the main host is available. Enable visual verdicts only after this passes.
- M6: test two actual browsers when browser tooling is available without installing packages.
- M7: replay representative queries with real retrieval and measure manual latency with the main dataset host. M8: verify restart, LAN/Cloudflare access and agent-off/manual-only mode on the deployment host.

# BLOCKERS

- No M3 code blocker. Actual browser interaction is unverified: Playwright CLI is absent and its wrapper would install a package, contrary to the no-install constraint. Next.js build and HTTP UI/API checks passed.
- The main hosting device is occupied; M4 live visual transport test is temporarily deferred.
- No browser automation executable is installed; two actual browser sessions remain unverified under the no-install constraint.
- M7 retrieval replay and M8 deployment/restart checks cannot run on the occupied main host. Planning-only replay and local startup/build checks passed.
- Live DRES rehearsal requires BTC session credentials, a running evaluation/task, and host access. The fake-server test proves local wiring only.

# DECISIONS

- Manual `/api/search` remains untouched and independent.
- No dependency installation; run Python through the existing local-backend `.venv`.
- Do not use removed `codex mcp-server`; defer App Server session wiring until UI work.
- No fallback retrieval based on score because existing similarity scores are not calibrated; fallback only on empty results or retrieval error where safe.
- Agent Search is shared by query, while each browser retains its own manual search state. A new query can replace a running agent query; cancel/reset increment the generation and discard late results.
- Run only one `agent.server` process because state is in memory. The agent service is optional; the Next.js proxy reports its failure without affecting manual search.
- Per operator request, defer only the host-dependent M4 live test. Keep visual checks `UNKNOWN` until real image transport is proven; transcript/metadata work can continue.
- Verify uses a separate ephemeral Codex context, one serialized in-memory queue and bounded evidence from a single video. Its visual verdicts remain UNKNOWN; no automatic submission is implemented.
- DRES submission is a separate human action on one dashboard row. Team session ID stays on the Next.js server; the submit PIN is configured there and entered per browser tab. The DRES payload includes the active task ID. TRAKE uses numeric frame numbers (operator confirmed). Duplicate protection is in memory for one Next.js process.
- `main` does not contain `zip-decode/fix`; per operator direction merge Agent alone. Block DRES submission for `N001–N100` in this branch until verified source PTS timing is integrated. A future merge with `zip-decode/fix` still needs to combine test scripts and adapt `readCandidateForDres` to version-2 session metadata and millisecond rows.
