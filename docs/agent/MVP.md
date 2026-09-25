# VORTA Competition Agent MVP

Operator setup and UI steps: [USAGE.md](USAGE.md).

## 1. Actual source map

- `local-client/frontend/src/pages/index.tsx`: browser state, manual search lifecycle, cancellation, result selection.
- `local-client/frontend/src/lib/api.ts`: manual `/api/search`, transcript, context-frame and media API calls.
- `local-client/frontend/src/components/ResultCard.tsx`, `ResultGrid.tsx`, `VideoModal.tsx`: candidate rendering, frame/video inspection and transcript display.
- `local-client/local-backend/main.py`: local FastAPI schemas and endpoints. Manual search enters at `POST /api/search`.
- `local-client/local-backend/app/strategies/base_strategy.py`: `SearchContext`, retrieval limits, result hydration and duplicate filtering.
- `local-client/local-backend/app/data_provider.py`: ZIP numpy/Milvus access or LOCAL proxy to `remote-server`.
- `local-client/local-backend/app/strategies/`: existing raw, multi-detail, multi-source and temporal retrieval.
- `remote-server/main.py` and `remote-server/app/`: matching production API/strategy contract backed by Milvus/CAGRA/PostgreSQL.

Critical path: query controls -> `runSearch()` -> `POST /api/search` -> `SearchRequest` -> discovered `BaseStrategy` -> `SearchContext.retrieve()` -> existing result dictionaries -> `SearchResponse` -> `ResultGrid`/`ResultCard` -> `VideoModal`. Neighbor evidence uses `GET /api/video/{id}/context-frames`; transcript evidence uses `GET /api/transcript/{id}`. Frames/video use the existing ZIP/YouTube endpoints.

## 2. Reused surfaces

- Search APIs and all current strategies; no retrieval math is copied.
- Existing result fields (`video_id`, `frame_id`, frame/timestamp, score/confidence, image URL, optional temporal `steps`).
- Existing context-frame, transcript, frame-image and video endpoints.
- Existing frontend result cards/modal for agent candidates and human visual inspection.
- Existing `codex` CLI authentication and runtime. No package installation.

## 3. Minimal architecture

`agent/` is a separate thin controller. Codex converts an official query into a bounded structured search plan. `search_vorta` validates that plan and calls the existing HTTP API. The manual browser path continues to call `/api/search` directly. An optional FastAPI agent service keeps one Search job and one serialized Verify queue in memory; Next.js `/api/agent-search` and `/api/agent-verify` proxy browser requests so two operators see the same agent state.

Both roles use bounded Codex CLI structured output because the installed backend venv has no Codex SDK/MCP packages and current official docs state that `codex mcp-server` was removed. The UI talks to the optional local agent service; no App Server integration is required for this MVP. This bridge is replaceable without touching retrieval.

## 4. Search Agent contract

Input: raw official query, backend capabilities and a top-k limit.

Output: `query`, short `plan_summary`, selected existing strategy/mode, one or more concise event queries with offsets, real VORTA candidates, timing, and optional error. At most two broad retrieval attempts; fallback is used only when the first attempt returns no candidates. It never verifies or submits.

## 5. Verify Agent contract

Input: active query plus one human-selected existing result (including temporal steps).

Output: per-requirement `MATCH | MISMATCH | UNKNOWN`, short evidence notes, and an overall operator action. It may fetch a small matched/neighbor frame bundle and transcript window. It never broad-searches or submits. If reliable image transport is unavailable, visual checks remain `UNKNOWN` and human-driven.

## 6. Backend/tool contracts

- `search_vorta(plan, top_k)`: frame mode maps to existing `POST /api/search`; transcript mode maps to `POST /api/search/transcript`. Responses retain original result metadata.
- `inspect_candidate(candidate)`: bounded composition of existing context-frame and transcript endpoints for one video. It returns frame metadata and nearby transcript segments; image pixels are not yet transported to Codex.
- All inputs and Codex outputs are validated. Timeouts/errors stay inside the agent process.

## 7. UI changes

M3 adds a compact Search Agent panel with official query input, shared status, candidates, cancel and reset. M5 adds `Verify with Agent` beside manual and agent candidates plus a shared Verify status/checks panel. Manual state and controls remain unchanged.

## 8. State model

In-memory state only: query id/generation, Search Agent status/results/thread, Verify Agent status/queue/results. A generation captured at job start must still match before publishing. Candidate verification keys use query generation plus candidate/frame identity.

## 9. Cancellation/reset

New query/reset increments generation, interrupts or ignores old work, clears agent results, and creates fresh role contexts. Codex planning uses a bounded subprocess that is terminated on cancellation. HTTP calls have bounded timeouts. Late results are discarded by generation comparison.

## 10. Concurrency

Manual requests retain the existing direct API. Agent calls have small top-k, one Search job and one serialized Verify worker. No global backend lock. Two browser tabs keep independent manual state. If latency rises, reduce agent top-k/fallback before changing manual search.

## 11. Milestones

- M0–M1: baseline, critical-path audit and this plan — done.
- M2: terminal Search Agent with real VORTA candidates — done.
- M3: shared Search UI, cancellation and generation safety — done.
- M4: real image transport proof — deferred while the main host is occupied; text fallback active.
- M5: human-triggered Verify UI/API with transcript/metadata evidence — done.
- M6: queue/reset/stale-result concurrency checks — local tests done; two actual browsers pending.
- M7: representative planning replay done; retrieval/latency replay on the dataset pending host access.
- M8: feature freeze, startup/manual-only checks; deployment checks pending host access.

## 12. Acceptance tests

- Existing manual visual search returns real results before and after agent work.
- Codex receives a query, emits a valid bounded plan, and `search_vorta` returns real dataset candidates.
- Agent failure/cancellation cannot alter manual search.
- Manual and agent requests can overlap; stale agent results cannot publish after reset.
- Verify handles real evidence, missing evidence and malformed responses without inventing claims.
- Two browser tabs can search while agent work runs; restart preserves manual-only operation.

## 13. Risks and fallbacks

- Codex unavailable/auth timeout: show agent error; manual remains usable.
- App Server integration cost: keep the tested CLI structured-output bridge for competition.
- ZIP mode lacks semantic transcript channels: use the existing dedicated fuzzy/lexical transcript endpoint or visual strategies.
- Main host is occupied, so the M4 live image transport test is deferred. The implemented Verify fallback uses transcript/metadata, forces visual checks to `UNKNOWN`, and leaves visual inspection in `VideoModal`.
- Shared retrieval contention: cap agent top-k and serialize only agent jobs.

## M3 local startup

`scripts/start-local.sh` starts the optional Search Agent service on `127.0.0.1:8012` using `local-client/local-backend/.venv`; Next.js proxies to it through `AGENT_SERVER_URL`. To start separately from the repository root:

```bash
VORTA_BACKEND_URL=http://127.0.0.1:8000 local-client/local-backend/.venv/bin/python -m uvicorn agent.server:app --host 127.0.0.1 --port 8012
```

Set `AGENT_SERVER_URL` on the frontend host if agent service is elsewhere. Use one agent service process for the shared in-memory state. Manual VORTA continues through `NEXT_PUBLIC_API_URL` when this service is down.
