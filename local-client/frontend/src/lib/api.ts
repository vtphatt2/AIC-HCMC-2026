import type {
  ContextFramesResponse,
  Strategy,
  QueryGroup,
  SearchResponse,
  SearchResult,
  StrategyConfigDraft,
  StrategyConfigResponse,
  StrategyConfigValue,
  TranslationResponse,
  TranscriptChunkSearchResponse,
  TranscriptResponse,
  VectorSearchAlgorithmResponse,
} from "@/types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

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

function draftUrl(strategyId: string, configId: string): string {
  const query = new URLSearchParams({ strategy_id: strategyId, config_id: configId });
  return `/api/tuning-draft?${query}`;
}

export async function fetchStrategyConfigDraft(
  strategyId: string,
  configId: string,
): Promise<StrategyConfigDraft> {
  const res = await fetch(draftUrl(strategyId, configId), { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to load local tuning draft");
  return res.json();
}

export async function saveStrategyConfigDraft(
  strategyId: string,
  configId: string,
  overrides: Record<string, StrategyConfigValue>,
): Promise<StrategyConfigDraft> {
  const res = await fetch(draftUrl(strategyId, configId), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ overrides }),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.error || "Failed to save local tuning draft");
  }
  return res.json();
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
  configOverrides: Record<string, StrategyConfigValue> = {},
  duplicateThreshold: number = 0.98,
): Promise<SearchResponse> {
  const payload: Record<string, unknown> = {
    strategy_id: strategyId,
    config_id: configId,
    config_overrides: configOverrides,
    query_groups: queryGroups.map(g => ({
      query: [g.semanticQuery, g.textQuery]
        .map(value => value.trim())
        .filter(Boolean)
        .join(" "),
      temporal_offset_ms: g.temporalOffsetMs
    })),
    top_k: topK,
    duplicate_threshold: duplicateThreshold,
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

// Neighboring indexed keyframes around [startMs, endMs] — for Video view's
// "expand a sparse strip with real nearby frames" feature, not a search.
export async function fetchContextFrames(
  videoId: string,
  startMs: number,
  endMs: number,
  expand: number = 20,
): Promise<ContextFramesResponse> {
  const params = new URLSearchParams({
    start_ms: String(Math.round(startMs)),
    end_ms: String(Math.round(endMs)),
    expand: String(expand),
  });
  const res = await fetch(apiUrl(`/api/video/${encodeURIComponent(videoId)}/context-frames?${params}`));
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Failed to load context frames");
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

// Jump straight to a video by its exact video_id (e.g. one found via
// transcript grep) instead of going through a search box — opens the same
// VideoModal a search hit would, seeded from whatever indexed keyframe the
// backend picks as the video's first one.
export async function fetchVideoById(videoId: string): Promise<SearchResult> {
  const res = await fetch(apiUrl(`/api/video/${encodeURIComponent(videoId)}`));
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Failed to look up video");
  }
  const info = await res.json();
  return {
    video_id: info.video_id,
    youtube_id: info.youtube_id ?? undefined,
    frame_id: info.frame_id,
    frame_number: info.frame_number,
    timestamp_ms: info.timestamp_ms,
    confidence: 1,
    frame_image_url: apiUrl(`/api/zip-frame/${info.video_id}/${info.timestamp_ms}`),
    fps: info.fps,
  };
}
