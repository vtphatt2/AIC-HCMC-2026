import { serializeSubmissionRow, validateSubmissionRow } from "./format";
import { useEffect, useState } from "react";
import type { SearchResult, SubmissionQueryType, SubmissionRow, SubmissionSessionSummary, SubmissionState } from "@/types";
import { apiUrl, fetchVideoById } from "@/lib/api";

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
): Promise<SubmissionState> {
  const res = await fetch("/api/submission", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "create", session, queryType }),
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

// Quick-add from VideoModal. rowIndex targets a specific existing TRAKE
// candidate (reviewing it from the dashboard); omitted, it lands wherever
// the session's own draftRowIndex points.
export function addSubmissionRowFrame(
  session: string,
  videoId: string,
  frame: number,
  rowIndex?: number,
): Promise<SubmissionState> {
  return postAction(session, { action: "add", videoId, frame, rowIndex, sourceFrameInput: videoId.startsWith("N") });
}

// Manual add: type a video_id + frame(s) directly, no search needed.
export function addSubmissionRow(
  session: string,
  videoId: string,
  frames: number[],
  answer?: string,
): Promise<SubmissionState> {
  return postAction(session, { action: "addRow", videoId, frames, answer });
}

export function editRowVideoId(session: string, rowIndex: number, videoId: string): Promise<SubmissionState> {
  return postAction(session, { action: "editVideoId", rowIndex, videoId });
}

export function editRowFrame(session: string, rowIndex: number, frameIndex: number, frame: number): Promise<SubmissionState> {
  return postAction(session, { action: "editFrame", rowIndex, frameIndex, frame });
}

export function removeRowFrame(session: string, rowIndex: number, frameIndex: number): Promise<SubmissionState> {
  return postAction(session, { action: "removeFrame", rowIndex, frameIndex });
}

export function removeRow(session: string, rowIndex: number): Promise<SubmissionState> {
  return postAction(session, { action: "removeRow", rowIndex });
}

// Per-row QA answer — every candidate gets its own answer text, not one
// answer shared across the whole session.
export function setRowAnswer(session: string, rowIndex: number, answer: string): Promise<SubmissionState> {
  return postAction(session, { action: "setAnswer", rowIndex, answer });
}

export function setQueryType(session: string, queryType: SubmissionQueryType): Promise<SubmissionState> {
  return postAction(session, { action: "setQueryType", queryType });
}

export function newTrakeCandidate(session: string): Promise<SubmissionState> {
  return postAction(session, { action: "newCandidate" });
}

export function resetSubmission(session: string): Promise<SubmissionState> {
  return postAction(session, { action: "reset" });
}

// order[i] = the current-array index that should land at position i — the
// same shape Array.prototype.map(( _, i) => order[i]) consumes, so callers
// building it from a drag-and-drop move just splice indices, not row data.
export function reorderSubmissionRows(session: string, order: number[]): Promise<SubmissionState> {
  return postAction(session, { action: "reorderRows", order });
}

// Raw CSV textarea save — replaces every row at once. Throws with a
// line-numbered message (from the server's parser) on malformed input, so
// the editor can show it without discarding what the user typed.
export function replaceSubmissionCsv(session: string, content: string): Promise<SubmissionState> {
  return postAction(session, { action: "replaceRaw", content });
}

export function fillSubmissionNeighbors(session: string): Promise<SubmissionState> {
  return postAction(session, { action: "fillNeighbors" });
}

// The session name is the exported filename (session.csv) — this moves
// the backing .runtime/submissions/*.csv/.meta.json files too, not just a label.
export function renameSubmissionSession(session: string, newSession: string): Promise<SubmissionState> {
  return postAction(session, { action: "rename", newSession });
}

// Move an item within an array — the primitive behind drag-and-drop row
// reordering (source/target are positions in the *displayed* order).
export function moveItem<T>(items: T[], from: number, to: number): T[] {
  const copy = [...items];
  const [moved] = copy.splice(from, 1);
  copy.splice(to, 0, moved);
  return copy;
}

// ── Video info cache ────────────────────────────────────────────────────
// Rows don't store fps/youtube_id — nothing here is knowledge the row itself
// carries, it's a property of the video, recomputable via the same
// /api/video/{id} lookup the "jump to video" feature uses. Cached in memory
// (per videoId, across the whole session) since a row's thumbnail and
// several other rows of the same video all want the same answer.
const videoInfoCache = new Map<string, Promise<{ fps: number; youtubeId?: string }>>();

export function getVideoInfo(videoId: string): Promise<{ fps: number; youtubeId?: string }> {
  let cached = videoInfoCache.get(videoId);
  if (!cached) {
    cached = fetchVideoById(videoId)
      .then((r) => ({ fps: r.fps, youtubeId: r.youtube_id }))
      .catch(() => ({ fps: 25, youtubeId: undefined }));
    videoInfoCache.set(videoId, cached);
  }
  return cached;
}

// One fetch per distinct videoId across however many rows reference it —
// getVideoInfo's own cache absorbs repeat calls across renders/components,
// this just turns "the set of videos currently shown" into React state.
export function useVideoInfo(videoIds: string[]): Record<string, { fps: number; youtubeId?: string }> {
  const [info, setInfo] = useState<Record<string, { fps: number; youtubeId?: string }>>({});
  const key = Array.from(new Set(videoIds)).sort().join(",");
  useEffect(() => {
    if (!key) return;
    let cancelled = false;
    Promise.all(key.split(",").map((id) => getVideoInfo(id).then((v) => [id, v] as const))).then((pairs) => {
      if (cancelled) return;
      setInfo((prev) => ({ ...prev, ...Object.fromEntries(pairs) }));
    });
    return () => { cancelled = true; };
  }, [key]);
  return info;
}

export function rowThumbUrl(videoId: string, frame: number, fps: number, sourceFrame?: number): string {
  if (videoId.startsWith("N")) {
    if (sourceFrame === undefined) return "";
    return apiUrl(`/api/zip-frame/${encodeURIComponent(videoId)}/${frame}?frame_number=${sourceFrame}&width=640`);
  }
  const timestampMs = Math.round((frame / (fps || 25)) * 1000);
  return apiUrl(`/api/zip-frame/${encodeURIComponent(videoId)}/${timestampMs}`);
}

// Reconstructs the SearchResult shape VideoModal needs, so the dashboard can
// reopen the same modal the search grid uses.
export function rowFrameToSearchResult(
  videoId: string,
  frame: number,
  fps: number,
  youtubeId?: string,
  sourceFrame?: number,
): SearchResult {
  if (videoId.startsWith("N") && sourceFrame === undefined) throw new Error("N review needs verified source frame identity");
  const source = sourceFrame ?? frame;
  return {
    video_id: videoId,
    youtube_id: youtubeId,
    frame_id: `${videoId}_${String(source).padStart(6, "0")}`,
    frame_number: source,
    timestamp_ms: videoId.startsWith("N") ? frame : (frame / (fps || 25)) * 1000,
    confidence: 1,
    frame_image_url: rowThumbUrl(videoId, frame, fps, sourceFrame),
    fps: fps || 25,
  };
}

// Organizer rule: every TRAKE candidate must have the same number of frames
// (matching the query's event count). The "frames from one video" rule is
// impossible to violate now — a row has exactly one videoId, not one per
// frame — so there's nothing left to check for that. `null` when consistent.
export function trakeSizeMismatch(state: SubmissionState): number[] | null {
  if (state.queryType !== "trake") return null;
  const sizes = new Set(state.rows.map((r) => r.frames.length));
  return sizes.size > 1 ? Array.from(sizes).sort((a, b) => a - b) : null;
}

// ── CSV export ───────────────────────────────────────────────────────────
function csvRow(state: SubmissionState, row: SubmissionRow): string {
  return serializeSubmissionRow(state.queryType, row);
}

export function buildSubmissionCsv(state: SubmissionState): { filename: string; rows: number; content: string } {
  state.rows.forEach(row => validateSubmissionRow(state.queryType, row, true));
  const csvRows = state.rows.map((r) => csvRow(state, r));
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
    .filter((s) => s.rows.length > 0)
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
