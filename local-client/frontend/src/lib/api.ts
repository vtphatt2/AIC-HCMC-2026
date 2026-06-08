import type { Strategy, QueryGroup, SearchResponse } from "@/types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export function apiUrl(path: string): string {
  return `${API_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

export async function fetchStrategies(): Promise<Strategy[]> {
  const res = await fetch(apiUrl("/api/strategies"));
  if (!res.ok) throw new Error("Failed to fetch strategies");
  return res.json();
}

export async function runSearch(strategyId: string, queryGroups: QueryGroup[], topK: number): Promise<SearchResponse> {
  const payload = {
    strategy_id: strategyId,
    query_groups: queryGroups.map(g => ({
      semantic_query: g.semanticQuery,
      text_query: g.textQuery,
      temporal_offset_ms: g.temporalOffsetMs
    })),
    top_k: topK
  };

  const res = await fetch(apiUrl("/api/search"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Search failed");
  }

  return res.json();
}
