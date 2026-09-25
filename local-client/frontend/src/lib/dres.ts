import type { SubmissionQueryType, SubmissionRow } from "@/types";

const VIDEO_ID = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/;

export function buildDresAnswer(
  queryType: SubmissionQueryType,
  row: SubmissionRow,
  fps?: number,
): { mediaItemName: string; start: number; end: number } | { text: string } {
  if (!VIDEO_ID.test(row.videoId)) throw new Error("Invalid video ID");
  if (!Array.isArray(row.frames) || row.frames.length === 0 ||
      row.frames.some((frame) => !Number.isSafeInteger(frame) || frame < 0)) {
    throw new Error("Candidate needs valid frame numbers");
  }
  const isNVideo = row.videoId.startsWith("N");

  if (queryType === "trake") {
    if (isNVideo) throw new Error("N videos are unavailable for TRAKE");
    if (new Set(row.frames).size !== row.frames.length) throw new Error("TRAKE frames must be distinct");
    return { text: `TR-${row.videoId}-${row.frames.join(",")}` };
  }

  if (row.frames.length !== 1) throw new Error("KIS and Q&A need exactly one frame");
  let timestampMs: number;
  if (isNVideo) {
    if (row.unit !== "milliseconds" || row.timingStatus !== "verified") {
      throw new Error("N videos need verified source PTS timing before DRES submission");
    }
    timestampMs = row.frames[0];
  } else {
    if (row.unit === "milliseconds") throw new Error("Frame unit does not match video timing");
    if (!Number.isFinite(fps) || !fps || fps <= 0) throw new Error("Video FPS is unavailable");
    timestampMs = Math.round((row.frames[0] / fps) * 1000);
  }
  if (!Number.isSafeInteger(timestampMs)) throw new Error("Frame timestamp is invalid");

  if (queryType === "kis") {
    return { mediaItemName: row.videoId, start: timestampMs, end: timestampMs };
  }
  const answer = row.answer?.trim();
  if (!answer || answer.length > 100 || /[\r\n]/.test(answer)) {
    throw new Error("Q&A needs an answer of 1–100 characters without line breaks");
  }
  return { text: `QA-${answer}-${row.videoId}-${timestampMs}` };
}

export function buildDresSubmission(
  taskId: string,
  queryType: SubmissionQueryType,
  row: SubmissionRow,
  fps?: number,
) {
  if (!taskId) throw new Error("No active DRES task");
  return { answerSets: [{ taskId, answers: [buildDresAnswer(queryType, row, fps)] }] };
}
