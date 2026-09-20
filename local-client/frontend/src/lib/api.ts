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
  TranscriptSearchAlgorithmId,
  TranscriptSearchAlgorithmResponse,
  TranscriptResponse,
  VectorSearchAlgorithmResponse,
} from "@/types";
import { buildTranscriptSearchPayload } from "@/lib/transcriptSearch";
import { RecentContextCache } from "@/lib/recentContext";

type ScoredContext = ContextFramesResponse & { scores: Record<string, number> | null };
const recentContexts = new RecentContextCache<ScoredContext>();
const recentTranscripts = new RecentContextCache<TranscriptResponse>();

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

export async function fetchTranscriptSearchAlgorithms(): Promise<TranscriptSearchAlgorithmResponse> {
  const res = await fetch(apiUrl("/api/transcript-search-algorithms"));
  if (!res.ok) throw new Error("Failed to fetch transcript search algorithms");
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
  signal?: AbortSignal,
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
    body: JSON.stringify(payload),
    signal,
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
  algorithm: TranscriptSearchAlgorithmId,
  topicFilter?: string,
  signal?: AbortSignal,
): Promise<TranscriptChunkSearchResponse> {
  const payload = buildTranscriptSearchPayload(query, topK, topicFilter, algorithm);

  const res = await fetch(apiUrl("/api/search/transcript"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
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
  signal?: AbortSignal,
): Promise<ContextFramesResponse> {
  const params = new URLSearchParams({
    start_ms: String(Math.round(startMs)),
    end_ms: String(Math.round(endMs)),
    expand: String(expand),
  });
  const res = await fetch(apiUrl(`/api/video/${encodeURIComponent(videoId)}/context-frames?${params}`), { signal });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Failed to load context frames");
  }
  return res.json();
}

// Second-phase Video-view scoring: re-scores an already-assembled frame
// set (a search's own matches + context-frames' expanded neighbors)
// against the query events that produced that search, on one consistent
// scale. Returns null on failure so existing context frames stay visible — this is a display
// enhancement, never something a missing/failed score should block on.
export async function fetchFrameScores(
  queryEvents: string[],
  frameIds: string[],
  eventWeights?: number[],
  duplicateThreshold?: number,
  signal?: AbortSignal,
): Promise<Record<string, number> | null> {
  if (queryEvents.length === 0 || frameIds.length === 0) return {};
  try {
    const res = await fetch(apiUrl("/api/frame-scores"), {
      method: "POST",
      signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query_groups: queryEvents.map((query) => ({ query })),
        event_weights: eventWeights ?? null,
        frame_ids: frameIds,
        // Same slider that drove the original search's own dedup — the
        // backend defaults to 0.98 only when this is omitted, so an
        // omitted value here would silently use a *different* threshold
        // than the search this view is showing.
        ...(duplicateThreshold !== undefined ? { duplicate_threshold: duplicateThreshold } : {}),
      }),
    });
    if (!res.ok) return null;
    return (await res.json()).scores ?? null;
  } catch {
    return null;
  }
}

export async function fetchScoredContext(
  videoId: string, startMs: number, endMs: number, expand: number,
  frames: { frame_id: string; frame_number: number }[], queryEvents: string[],
  eventWeights?: number[], duplicateThreshold?: number, signal?: AbortSignal,
  onCached?: (context: ScoredContext) => void,
): Promise<ScoredContext> {
  const url = apiUrl(`/api/video/${encodeURIComponent(videoId)}/context-scores`);
  const body = JSON.stringify({
    start_ms: Math.round(startMs), end_ms: Math.round(endMs), expand,
    frame_ids: frames.map(f => f.frame_id),
    frame_numbers: Object.fromEntries(frames.map(f => [f.frame_id, f.frame_number])),
    query_groups: queryEvents.map(query => ({ query })),
    event_weights: eventWeights ?? null, duplicate_threshold: duplicateThreshold ?? 0.98,
  });
  const key = JSON.stringify([url, body]);
  if (onCached && !signal?.aborted) {
    const cached = recentContexts.get(key);
    if (cached) onCached(cached);
  }
  // Always revalidate: cached neighbors/scores only bridge the network delay.
  try {
    const res = await fetch(url, {
      method: "POST", signal,
      headers: { "Content-Type": "application/json" }, body,
    });
    if (res.ok) {
      const context: ScoredContext = await res.json();
      if (!signal?.aborted) {
        if (context.scores !== null && context.scores !== undefined) recentContexts.set(key, context);
        else recentContexts.delete(key);
      }
      return context;
    }
    recentContexts.delete(key);
    // Keep compatibility with the local backend and older remote deployments.
    if (res.status !== 404 && res.status !== 405) throw new Error("Failed to load context scores");
    const context = await fetchContextFrames(videoId, startMs, endMs, expand, signal);
    const matched = new Set(frames.map(f => f.frame_id));
    const additions = [...context.before, ...context.middle, ...context.after].filter(f => !matched.has(f.frame_id));
    const ids = [...frames, ...additions].sort((a, b) => a.frame_number - b.frame_number).map(f => f.frame_id);
    const scores = await fetchFrameScores(queryEvents, ids, eventWeights, duplicateThreshold, signal);
    return { ...context, scores };
  } catch (error) {
    if (!signal?.aborted) recentContexts.delete(key);
    throw error;
  }
}

export async function fetchTranscript(
  videoId: string, onCached?: (transcript: TranscriptResponse) => void,
): Promise<TranscriptResponse> {
  const url = apiUrl(`/api/transcript/${videoId}`);
  if (onCached) {
    const cached = recentTranscripts.get(url);
    if (cached) onCached(cached);
  }
  try {
    const res = await fetch(url);
    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || "Failed to fetch transcript");
    }
    const transcript: TranscriptResponse = await res.json();
    if (transcript.video_id === videoId && Array.isArray(transcript.segments)) {
      recentTranscripts.set(url, transcript);
    } else recentTranscripts.delete(url);
    return transcript;
  } catch (error) {
    recentTranscripts.delete(url);
    throw error;
  }
}

// Keeps the original one-video response contract, while the backend also
// accepts a title or compact ID prefix and resolves it to the highest-ranked
// canonical video ID. Exact IDs remain the common fast path (including calls
// from the submission dashboard).
export async function fetchVideoById(lookup: string): Promise<SearchResult> {
  const res = await fetch(apiUrl(`/api/video/${encodeURIComponent(lookup)}`));
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
