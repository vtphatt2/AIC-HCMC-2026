import type { SearchResult, SubmissionEntry, SubmissionQueryType, SubmissionSessionSummary, SubmissionState } from "@/types";

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
  imageUrl?: string,
  fps?: number,
  youtubeId?: string,
): Promise<SubmissionState> {
  return postAction(session, { action: "add", videoId, frame, imageUrl, fps, youtubeId });
}

export function editSubmissionEntryFrame(session: string, id: string, frame: number): Promise<SubmissionState> {
  return postAction(session, { action: "editFrame", id, frame });
}

export function removeSubmissionEntry(session: string, id: string): Promise<SubmissionState> {
  return postAction(session, { action: "remove", id });
}

// Reconstructs the SearchResult shape VideoModal needs from a stored
// submission entry, so the dashboard can reopen the same modal the search
// grid uses. `|| 25` only covers entries added before fps was captured.
export function submissionEntryToSearchResult(entry: SubmissionEntry): SearchResult {
  const fps = entry.fps || 25;
  return {
    video_id: entry.videoId,
    youtube_id: entry.youtubeId,
    frame_id: entry.id,
    frame_number: entry.frame,
    timestamp_ms: (entry.frame / fps) * 1000,
    confidence: 1,
    frame_image_url: entry.imageUrl || "",
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

// ── CSV export, matching Python's csv.QUOTE_MINIMAL ────────────────────────

function csvField(value: string | number): string {
  const s = String(value);
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function csvRow(fields: (string | number)[]): string {
  return fields.map(csvField).join(",");
}

export function buildSubmissionCsv(state: SubmissionState): { filename: string; rows: number; content: string } {
  let rows: string[];

  if (state.queryType === "trake") {
    const groups = new Map<number, typeof state.entries>();
    for (const entry of state.entries) {
      if (!groups.has(entry.groupIndex)) groups.set(entry.groupIndex, []);
      groups.get(entry.groupIndex)!.push(entry);
    }
    rows = Array.from(groups.entries())
      .sort(([a], [b]) => a - b)
      .map(([, entries]) => {
        const sorted = [...entries].sort((a, b) => a.addedAt - b.addedAt);
        return csvRow([sorted[0].videoId, ...sorted.map((e) => e.frame)]);
      });
  } else if (state.queryType === "qa") {
    rows = state.entries.map((e) => csvRow([e.videoId, e.frame, state.answer]));
  } else {
    rows = state.entries.map((e) => csvRow([e.videoId, e.frame]));
  }

  return {
    filename: `query-${state.queryNumber}-${state.queryType}.csv`,
    rows: rows.length,
    content: rows.length ? `${rows.join("\r\n")}\r\n` : "",
  };
}

export function downloadCsv(filename: string, content: string): void {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

if (process.env.NODE_ENV !== "production") {
  console.assert(csvField("plain") === "plain", "csvField: plain passthrough");
  console.assert(csvField("a,b") === '"a,b"', "csvField: comma quoted");
  console.assert(csvField('say "hi"') === '"say ""hi"""', "csvField: embedded quote doubled");
}
