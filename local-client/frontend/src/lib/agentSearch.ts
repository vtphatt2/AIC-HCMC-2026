import type { SearchResult, TranscriptChunkResult } from "@/types";

export interface AgentSearchState {
  query_id: string | null;
  generation: number;
  query: string;
  status: "idle" | "planning" | "searching" | "done" | "error" | "cancelled";
  plan_summary: string;
  plan: { primary: { mode: "frames" | "transcript"; strategy: string } } | null;
  results: SearchResult[] | TranscriptChunkResult[];
  total: number;
  timing_ms: { planning: number; retrieval: number; total: number } | null;
  error: string | null;
}

async function request(method: "GET" | "POST" | "DELETE", body?: object): Promise<AgentSearchState> {
  const response = await fetch("/api/agent-search", {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    cache: "no-store",
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || payload.detail || "Search Agent request failed");
  return payload as AgentSearchState;
}

export const fetchAgentSearch = () => request("GET");
export const startAgentSearch = (query: string, topK = 20) =>
  request("POST", { query, top_k: topK });
export const cancelAgentSearch = () => request("POST", { action: "cancel" });
export const resetAgentSearch = () => request("DELETE");
