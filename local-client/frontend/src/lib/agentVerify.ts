import type { SearchResult, TranscriptChunkResult } from "@/types";

export type VerifyCandidate = SearchResult | TranscriptChunkResult;
export type VerifyStatus = "MATCH" | "MISMATCH" | "UNKNOWN";

export interface VerifyCheck {
  requirement: string;
  modality: "visual" | "transcript" | "unknown";
  status: VerifyStatus;
  evidence_quote: string | null;
  note: string;
}

export interface VerifyResult {
  query: string;
  candidate_id: string;
  candidate: { candidate_id: string; video_id: string; frame_id?: string; chunk_id?: number };
  status: "done" | "error";
  checks?: VerifyCheck[];
  overall?: "PROMISING" | "UNLIKELY" | "INSPECT";
  next_action?: string;
  evidence?: { visual_content_available: boolean; errors: string[] };
  timing_ms?: number;
  error?: string;
}

export interface VerifyState {
  query: string;
  generation: number;
  status: "idle" | "verifying" | "done" | "error" | "cancelled";
  current_candidate: { candidate_id: string; video_id: string; frame_id?: string; chunk_id?: number } | null;
  queue: Array<{ candidate_id: string; video_id: string }>;
  results: Record<string, VerifyResult>;
  error: string | null;
}

function compactCandidate(candidate: VerifyCandidate): object {
  if ("chunk_id" in candidate) {
    return {
      video_id: candidate.video_id,
      chunk_id: candidate.chunk_id,
      frame_number: candidate.frame_number,
      nearest_timestamp_ms: candidate.nearest_timestamp_ms,
      start_time_ms: candidate.start_time_ms,
      end_time_ms: candidate.end_time_ms,
    };
  }
  const frame = (item: SearchResult) => ({
    video_id: item.video_id,
    frame_id: item.frame_id,
    frame_number: item.frame_number,
    timestamp_ms: item.timestamp_ms,
  });
  return { ...frame(candidate), steps: candidate.steps?.slice(0, 4).map(frame) };
}

async function request(method: "GET" | "POST" | "DELETE", body?: object): Promise<VerifyState> {
  const response = await fetch("/api/agent-verify", {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    cache: "no-store",
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || payload.detail || "Verify Agent request failed");
  return payload as VerifyState;
}

export const fetchVerify = () => request("GET");
export const startVerify = (query: string, candidate: VerifyCandidate) =>
  request("POST", { query, candidate: compactCandidate(candidate) });
export const cancelVerify = () => request("POST", { action: "cancel" });
export const resetVerify = () => request("DELETE");
