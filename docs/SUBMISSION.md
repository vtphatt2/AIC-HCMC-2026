# Submission Dashboard

How the team builds and exports AIC26 submission CSVs during the contest.
Final-round, operator-triggered DRES submission: [DRES.md](DRES.md).
Contest rules (filename format, row format, packaging) are
[SUBMISSION_RULES.md](SUBMISSION_RULES.md) — this page is about the tool,
not the rules.

The CSV/ZIP workflow below is for the earlier upload format. In the final round,
use the DRES panel on this dashboard to submit one reviewed candidate at a time.

## Sessions = files

A **session** is one working area for one BTC query, and its name **is**
the exported filename (`{session}.csv`) — name it exactly what BTC's query
file is called (e.g. `query-p1-11-kis`, not `query-1`), so export always
matches without any separate number field to keep in sync.

Open it two ways:
- `/submissions` — the full dashboard, every session at once
- The 🗳 button in the main search page header — a compact single-session
  panel, for quickly adding a result you just found without leaving search

Both talk to the same backend (`pages/api/submission.ts`) and the same
storage, so they always agree.

## Storage

`.runtime/submissions/{session}.csv` + `{session}.meta.json` — gitignored,
local to whichever machine is running the frontend. The `.csv` **is** the
literal export file (no header, comma-delimited, CRLF), not a derived view
of some other model — row order on disk is rank order, and what you'd get
from "Download CSV" is a byte-for-byte copy. The `.meta.json` sidecar only
holds what a CSV can't: `queryType`, `draftRowIndex` (TRAKE's in-progress
candidate pointer), timestamps.

One row = one CSV line = one candidate:
- **KIS** — `video_id,frame`
- **QA** — `video_id,frame,answer` — every row has its *own* answer, not
  one shared across the session
- **TRAKE** — `video_id,frame_1,...,frame_N`, all frames from one video —
  a row has exactly one `videoId` field, so mixing videos into a candidate
  is impossible by construction, not just a warning

## Editing a session — two modes, same data

Toggle per session on `/submissions`: **▦ Frame grid** ⇄ **📝 Raw CSV**.

**Frame grid** (default) — thumbnails, drag a row to rerank, inline-edit
any frame number or `video_id`, "+ frame" to append another frame to an
existing TRAKE candidate, "✕" to remove a row or a frame. The
`video_id, frame(s)` box on every session adds a row by typing it directly
— no search needed.

**Raw CSV** — a textarea on the literal file content. Save re-parses it
against the session's `queryType` and replaces every row at once; a bad
line is rejected with its line number, without discarding what you typed.
Useful for pasting in results computed elsewhere, or bulk-editing rows by
hand.

Both modes hit the same backend actions, so switching mid-edit never loses
anything already saved.

## KIS neighbor filler and grid review

For a KIS session with at least one manually ranked candidate, **Filler to
100** preserves all existing rows and appends nearby frames until the
organizer's 100-row limit is reached. It walks each original candidate in
round-robin order using offsets `-15, +15, -30, +30, ...`, skips negative
frames and duplicate `video_id,frame` pairs, and never changes the original
rank order. The operation is deliberately unavailable for QA and TRAKE:
duplicating a QA answer is not necessarily valid, while TRAKE requires an
exact event/frame count per candidate.

**Review grid ↗** opens that session in a separate browser tab and displays
every CSV frame as a responsive thumbnail grid in CSV rank order. It works
for all query types and refreshes from the stored session every five seconds,
so it can remain open while teammates continue editing the submission.

To keep a filled session manageable, the dashboard shows only its first five
ranked candidates by default whenever it contains more than five rows. Use
**Show all N** to expand that session and **Collapse to 5** to reduce it again.
This is display-only: Raw CSV, Review grid, exports, and the stored submission
always retain every candidate.

## Find and jump to a video

The `Title or video ID…` box in the main search page header (next to 🗳)
opens the highest-ranked indexed video by organizer title, exact `video_id`,
or ID prefix. ID separators are ignored for prefix matching, so `L0_` finds
the `L01_…`–`L09_…` groups; title matching is case- and accent-insensitive.

The request and response stay on the old `GET /api/video/{lookup}` contract:
exact IDs use the original fast path, while a non-exact lookup is resolved to
the first ranked canonical ID before the backend returns that video's
`fps`/`youtube_id` and first indexed keyframe. Local-backend needs ZIP-mode
exported vectors (`numpy_vector_store`), the same as regular search; its titles
come from the organizer's `media-info*.zip` archive.

## Migrating older sessions

Before 2026-08-21, sessions were stored as one big `{session}.json` per
session (entries + a separate row-order list + per-entry group index).
Existing sessions in that format aren't read by the current code — `git
pull` won't touch `.runtime/` (gitignored), so on any machine that already
has old-format sessions, run once after pulling:

```bash
node scripts/migrate-submissions-to-csv.cjs
```

Safe to run even if there's nothing to migrate (no-ops if `.runtime/submissions/`
holds only the new format already).
