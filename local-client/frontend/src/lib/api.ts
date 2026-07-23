import type {
  Strategy,
  QueryGroup,
  SearchResponse,
  StrategyConfigPreset,
  StrategyConfigResponse,
  StrategyConfigValue,
  TranslationResponse,
  TranscriptChunkSearchResponse,
  TranscriptResponse,
  VectorSearchAlgorithmResponse,
} from "@/types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");
export const strategyConfigUpdateKey = (strategyId: string, configId: string) =>
  `aic2026-strategy-config-updated:${strategyId}:${configId}`;

export function apiUrl(path: string): string {
  return `${API_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

export async function fetchStrategies(): Promise<Strategy[]> {
  const res = await fetch(apiUrl("/api/strategies"));
  if (!res.ok) throw new Error("Failed to fetch strategies");
  return res.json();
}

export async function fetchStrategyConfigs(strategyId: string): Promise<StrategyConfigResponse> {
  const res = await fetch(apiUrl(`/api/strategies/${encodeURIComponent(strategyId)}/configs`));
  if (!res.ok) throw new Error("Failed to fetch strategy configs");
  return res.json();
}

export async function saveStrategyConfig(
  strategyId: string,
  configId: string,
  weights: Record<string, StrategyConfigValue>,
): Promise<StrategyConfigPreset> {
  const res = await fetch(
    apiUrl(`/api/strategies/${encodeURIComponent(strategyId)}/configs/${encodeURIComponent(configId)}`),
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ weights }),
    },
  );
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Failed to save strategy config");
  }
  return res.json();
}

export async function deleteStrategyConfig(strategyId: string, configId: string): Promise<void> {
  const res = await fetch(
    apiUrl(`/api/strategies/${encodeURIComponent(strategyId)}/configs/${encodeURIComponent(configId)}`),
    { method: "DELETE" },
  );
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Failed to delete strategy config");
  }
}

export async function fetchVectorSearchAlgorithms(): Promise<VectorSearchAlgorithmResponse> {
  const res = await fetch(apiUrl("/api/vector-search-algorithms"));
  if (!res.ok) throw new Error("Failed to fetch vector search algorithms");
  return res.json();
}

export async function warmupTextEncoder(): Promise<void> {
  const res = await fetch(apiUrl("/api/warmup_text_encoder?passes=1"), {
    method: "POST",
  });
  if (!res.ok) throw new Error("Failed to warm up text encoder");
}

export async function translateTexts(texts: string[]): Promise<TranslationResponse> {
  const res = await fetch(apiUrl("/api/translate"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texts }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Translation failed");
  }

  return res.json();
}

export async function runSearch(
  strategyId: string,
  queryGroups: QueryGroup[],
  topK: number,
  videoGenre: string = "All",
  vectorSearchAlgorithm?: string,
  configId: string = "default",
): Promise<SearchResponse> {
  const payload: Record<string, unknown> = {
    strategy_id: strategyId,
    config_id: configId,
    query_groups: queryGroups.map(g => ({
      query: [g.semanticQuery, g.textQuery]
        .map(value => value.trim())
        .filter(Boolean)
        .join(" "),
      temporal_offset_ms: g.temporalOffsetMs
    })),
    top_k: topK,
    ...(vectorSearchAlgorithm ? { vector_search_algorithm: vectorSearchAlgorithm } : {})
  };
  if (videoGenre && videoGenre !== "All") {
    payload.video_genre = videoGenre;
  }

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

export async function searchTranscriptChunks(
  query: string,
  topK: number,
  topicFilter?: string,
): Promise<TranscriptChunkSearchResponse> {
  const payload: Record<string, unknown> = { query, top_k: topK };
  if (topicFilter) payload.topic_filter = topicFilter;

  const res = await fetch(apiUrl("/api/search/transcript"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Transcript chunk search failed");
  }

  return res.json();
}

export async function fetchTranscript(videoId: string): Promise<TranscriptResponse> {
  const res = await fetch(apiUrl(`/api/transcript/${videoId}`));
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Failed to fetch transcript");
  }
  return res.json();
}
