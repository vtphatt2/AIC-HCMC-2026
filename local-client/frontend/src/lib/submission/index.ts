import type { SearchResult, SubmissionEntry, SubmissionQueryType, SubmissionSessionSummary, SubmissionState } from "@/types";
import { apiUrl } from "@/lib/api";
import { sortByRowOrder } from "./types";

export { reorderKeys } from "./types";
import * as kis from "./kis";
import * as qa from "./qa";
import * as trake from "./trake";

export { groupKey as trakeGroupKey } from "./trake";

// Talks to pages/api/submission.ts — a same-origin Next.js route, NOT the
// FastAPI backend, so these calls deliberately do not go through apiUrl().
// scripts/share-proxy.cjs routes /api/submission to the Next side for the
// same reason it already does for /api/tuning-draft.

async function parseError(res: Response, fallback: string): Promise<never> {
  const data = await res.json().catch(() => ({}));
  const err = new Error(data.error || fallback) as Error & { status?: number };
  err.status = res.status;
  throw err;
}

export async function fetchSubmissionSessions(): Promise<SubmissionSessionSummary[]> {
  const res = await fetch("/api/submission", { cache: "no-store" });
  if (!res.ok) return parseError(res, "Failed to list submission sessions");
  return (await res.json()).sessions;
}

export async function createSubmissionSession(
  session: string,
  queryType: SubmissionQueryType = "kis",
  queryNumber: number = 1,
): Promise<SubmissionState> {
  const res = await fetch("/api/submission", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "create", session, queryType, queryNumber }),
  });
  if (!res.ok) return parseError(res, "Failed to create submission session");
  return res.json();
}

export async function fetchSubmission(session: string): Promise<SubmissionState> {
  const res = await fetch(`/api/submission?session=${encodeURIComponent(session)}`, { cache: "no-store" });
  if (!res.ok) return parseError(res, "Failed to load submission session");
  return res.json();
}

export async function deleteSubmissionSession(session: string): Promise<void> {
  const res = await fetch(`/api/submission?session=${encodeURIComponent(session)}`, { method: "DELETE" });
  if (!res.ok) return parseError(res, "Failed to delete submission session");
}

async function postAction(session: string, body: Record<string, unknown>): Promise<SubmissionState> {
  const res = await fetch(`/api/submission?session=${encodeURIComponent(session)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) return parseError(res, "Submission update failed");
  return res.json();
}

export function addSubmissionEntry(
  session: string,
  videoId: string,
  frame: number,
  fps?: number,
  youtubeId?: string,
  groupIndex?: number,
): Promise<SubmissionState> {
  return postAction(session, { action: "add", videoId, frame, fps, youtubeId, groupIndex });
}

export function editSubmissionEntryFrame(session: string, id: string, frame: number): Promise<SubmissionState> {
  return postAction(session, { action: "editFrame", id, frame });
}

export function editSubmissionEntryGroup(session: string, id: string, groupIndex: number): Promise<SubmissionState> {
  return postAction(session, { action: "editGroup", id, groupIndex });
}

export function removeSubmissionEntry(session: string, id: string): Promise<SubmissionState> {
  return postAction(session, { action: "remove", id });
}

export function reorderSubmissionRows(session: string, rowOrder: string[]): Promise<SubmissionState> {
  return postAction(session, { action: "reorderRows", rowOrder });
}

// The session name is the exported filename (session.csv) — this moves
// the backing .runtime/submissions/*.json file too, not just a label.
export function renameSubmissionSession(session: string, newSession: string): Promise<SubmissionState> {
  return postAction(session, { action: "rename", newSession });
}

// No stored thumbnail — always decode fresh from videoId/frame/fps via the
// same route VideoModal/ResultCard already use, so editing the frame number
// (or moving it to a different candidate) never leaves a stale image
// behind. `|| 25` only covers entries added before fps was captured.
export function submissionEntryThumbUrl(entry: SubmissionEntry): string {
  const fps = entry.fps || 25;
  const timestampMs = Math.round((entry.frame / fps) * 1000);
  return apiUrl(`/api/zip-frame/${encodeURIComponent(entry.videoId)}/${timestampMs}`);
}

// Reconstructs the SearchResult shape VideoModal needs from a stored
// submission entry, so the dashboard can reopen the same modal the search
// grid uses.
export function submissionEntryToSearchResult(entry: SubmissionEntry): SearchResult {
  const fps = entry.fps || 25;
  return {
    video_id: entry.videoId,
    youtube_id: entry.youtubeId,
    frame_id: entry.id,
    frame_number: entry.frame,
    timestamp_ms: (entry.frame / fps) * 1000,
    confidence: 1,
    frame_image_url: submissionEntryThumbUrl(entry),
    fps,
  };
}

export function setSubmissionMeta(
  session: string,
  patch: Partial<Pick<SubmissionState, "queryType" | "queryNumber" | "answer">>,
): Promise<SubmissionState> {
  return postAction(session, { action: "setMeta", ...patch });
}

export function newSubmissionCandidate(session: string): Promise<SubmissionState> {
  return postAction(session, { action: "newCandidate" });
}

export function resetSubmission(session: string): Promise<SubmissionState> {
  return postAction(session, { action: "reset" });
}

// ── Per-type dispatch ───────────────────────────────────────────────────────

// Organizer rule: every TRAKE candidate must have frames from one video
// only. `null` when there's nothing to warn about.
export function trakeCandidateVideoMismatch(state: SubmissionState): number[] | null {
  if (state.queryType !== "trake") return null;
  const { videoMismatch } = trake.validate(state.entries);
  return videoMismatch.length ? videoMismatch : null;
}

// Organizer rule: every TRAKE candidate must have the same number of frames
// (matching the query's event count). `null` when consistent.
export function trakeCandidateSizeMismatch(state: SubmissionState): number[] | null {
  if (state.queryType !== "trake") return null;
  const { sizeMismatch } = trake.validate(state.entries);
  return sizeMismatch.length ? sizeMismatch : null;
}

// Entries grouped by candidate, frames ascending within each, candidates in
// rowOrder — same order the CSV export uses, for UI display.
export function trakeSortedGroups(entries: SubmissionEntry[], rowOrder: string[]): SubmissionEntry[][] {
  return trake.sortedGroups(entries, rowOrder);
}

// KIS/QA's flat-list equivalent: entries in rowOrder, for UI display.
export function orderedEntries(state: SubmissionState): SubmissionEntry[] {
  return sortByRowOrder(state.entries, state.rowOrder, (e) => e.id);
}

// ── CSV export ───────────────────────────────────────────────────────────
// No auto-quoting — fields are written exactly as typed. Quoting a QA
// answer (e.g. one containing a comma) is the user's own call to make in
// the answer text itself, not something this tool infers.

function csvRow(fields: (string | number)[]): string {
  return fields.join(",");
}

export function buildSubmissionCsv(state: SubmissionState): { filename: string; rows: number; content: string } {
  const rows =
    state.queryType === "trake" ? trake.buildRows(trake.toGroups(state.entries, state.rowOrder)) :
    state.queryType === "qa" ? qa.buildRows(qa.toGroups(state.entries, state.rowOrder), state.answer) :
    kis.buildRows(kis.toGroups(state.entries, state.rowOrder));

  const csvRows = rows.map(csvRow);
  return {
    // BTC's actual query filenames don't follow a plain query-{N}-{type}
    // pattern (e.g. query-p1-11-kis.txt) — the session name IS the exact
    // filename to match, so name the session exactly what BTC's file is
    // called (minus extension) and this always matches.
    filename: `${state.session}.csv`,
    rows: csvRows.length,
    content: csvRows.length ? `${csvRows.join("\r\n")}\r\n` : "",
  };
}

function triggerDownload(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function downloadCsv(filename: string, content: string): void {
  triggerDownload(filename, new Blob([content], { type: "text/csv;charset=utf-8" }));
}

// ── submission.zip — a `submission/` folder of query CSVs, per the AIC26
// organizer's required upload shape. Hand-rolled, uncompressed (STORE) ZIP
// writer: the CSVs are a few KB each, so there's nothing to gain from
// DEFLATE, and Node/the browser have no built-in ZIP *container* writer to
// reach for (zlib only does raw compression, not the archive format).

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i++) crc = CRC_TABLE[(crc ^ data[i]) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function u16(n: number): number[] { return [n & 0xff, (n >>> 8) & 0xff]; }
function u32(n: number): number[] { return [n & 0xff, (n >>> 8) & 0xff, (n >>> 16) & 0xff, (n >>> 24) & 0xff]; }

function dosDateTime(d: Date): { time: number; date: number } {
  return {
    time: ((d.getHours() & 0x1f) << 11) | ((d.getMinutes() & 0x3f) << 5) | ((d.getSeconds() >> 1) & 0x1f),
    date: ((Math.max(0, d.getFullYear() - 1980) & 0x7f) << 9) | (((d.getMonth() + 1) & 0xf) << 5) | (d.getDate() & 0x1f),
  };
}

export function buildSubmissionZip(states: SubmissionState[]): Blob | null {
  const enc = new TextEncoder();
  // Filenames come from the session name (see buildSubmissionCsv), and
  // session names are already enforced unique at creation time (409 on
  // duplicate), so two sessions can never collide on the same zip entry.
  const files = states
    .filter((s) => s.entries.length > 0)
    .map((s) => {
      const { filename, content } = buildSubmissionCsv(s);
      return { name: enc.encode(`submission/${filename}`), data: enc.encode(content) };
    });
  if (files.length === 0) return null;

  const { time, date } = dosDateTime(new Date());
  const parts: BlobPart[] = [];
  const centralParts: BlobPart[] = [];
  let offset = 0;
  let centralSize = 0;

  for (const file of files) {
    const crc = crc32(file.data);
    const size = file.data.length;

    const local = new Uint8Array([
      ...u32(0x04034b50), ...u16(20), ...u16(0), ...u16(0),
      ...u16(time), ...u16(date),
      ...u32(crc), ...u32(size), ...u32(size),
      ...u16(file.name.length), ...u16(0),
    ]);
    parts.push(local, file.name, file.data);

    const central = new Uint8Array([
      ...u32(0x02014b50), ...u16(20), ...u16(20), ...u16(0), ...u16(0),
      ...u16(time), ...u16(date),
      ...u32(crc), ...u32(size), ...u32(size),
      ...u16(file.name.length), ...u16(0), ...u16(0), ...u16(0), ...u16(0),
      ...u32(0), ...u32(offset),
    ]);
    centralParts.push(central, file.name);
    centralSize += central.length + file.name.length;

    offset += local.length + file.name.length + file.data.length;
  }

  const end = new Uint8Array([
    ...u32(0x06054b50), ...u16(0), ...u16(0),
    ...u16(files.length), ...u16(files.length),
    ...u32(centralSize), ...u32(offset), ...u16(0),
  ]);

  return new Blob([...parts, ...centralParts, end], { type: "application/zip" });
}

export function downloadSubmissionZip(states: SubmissionState[], zipName: string = "submission.zip"): boolean {
  const blob = buildSubmissionZip(states);
  if (!blob) return false;
  triggerDownload(zipName, blob);
  return true;
}

if (process.env.NODE_ENV !== "production") {
  console.assert(csvRow(["plain", 1]) === "plain,1", "csvRow: no auto-quoting");
  console.assert(csvRow(['"a,b"']) === '"a,b"', "csvRow: pre-quoted values pass through untouched");
}
