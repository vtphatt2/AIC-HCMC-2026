export interface Strategy {
  id: string;
  name: string;
  description: string;
  author: string;
  version: string;
  configurable?: boolean;
}

export interface StrategyConfigField {
  type?: "number" | "number_list";
  label: string;
  item_label?: string;
  default: number | number[];
  min: number;
  max: number;
  step: number;
  min_items?: number;
  max_items?: number;
}

export type StrategyConfigValue = number | number[];

export interface StrategyConfigPreset {
  id: string;
  strategy_id: string;
  strategy_version: string;
  revision: number;
  weights: Record<string, StrategyConfigValue>;
}

export interface StrategyConfigResponse {
  strategy_id: string;
  schema: Record<string, StrategyConfigField>;
  configs: StrategyConfigPreset[];
}

export interface StrategyConfigDraft {
  revision: number;
  overrides: Record<string, StrategyConfigValue>;
}

export interface VectorSearchAlgorithm {
  id: string;
  name: string;
  available: boolean;
  description: string;
}

export interface VectorSearchAlgorithmResponse {
  default: string;
  algorithms: VectorSearchAlgorithm[];
}

export interface QueryGroup {
  semanticQuery: string;
  textQuery: string;
  temporalOffsetMs: number;  // ms after the previous group's result — 0 for the first group
}

export interface TranslationResponse {
  translations: string[];
}

export interface SearchResult {
  video_id: string;
  youtube_id?: string;
  frame_id: string;
  frame_number: number;
  timestamp_ms: number;
  confidence: number;
  frame_image_url: string;
  fps: number;
  // Optional low-res placeholder to show while frame_image_url is still
  // extracting. No backend sets it today; the components degrade to showing
  // nothing until the real frame arrives.
  frame_preview_url?: string;
  // Present only for multi-step temporal matches: every frame in the matched
  // chain, in step order (this result itself is the chain's closing frame).
  steps?: SearchResult[];
}

// ── Submission sessions ───────────────────────────────────────────────────
// A "session" is a named working folder a searcher creates or joins, so
// multiple teammates can each solve a different organizer query at once
// without stepping on each other. Backed by pages/api/submission.ts.

export type SubmissionQueryType = "kis" | "qa" | "trake";

export interface SubmissionEntry {
  id: string;
  videoId: string;
  frame: number;
  imageUrl?: string;
  fps: number; // captured at add-time so a later dashboard view can reopen VideoModal
  youtubeId?: string;
  groupIndex: number; // TRAKE candidate grouping; always 0 for kis/qa
  addedAt: number;
}

export interface SubmissionState {
  session: string;
  revision: number;
  queryType: SubmissionQueryType;
  queryNumber: number;
  answer: string; // qa only
  nextGroupIndex: number;
  // Ranked row order for CSV export: entry.id per row for kis/qa, `g${groupIndex}`
  // per candidate for trake. Server keeps this in sync (pages/api/submission.ts's
  // syncRowOrder) — stale keys drop out, new ones append at the end.
  rowOrder: string[];
  entries: SubmissionEntry[];
  createdAt: number;
  updatedAt: number;
}

export interface SubmissionSessionSummary {
  session: string;
  queryType: SubmissionQueryType;
  queryNumber: number;
  entryCount: number;
  createdAt: number;
  updatedAt: number;
}

export interface SearchResponse {
  results: SearchResult[];
  strategy_id: string;
  total: number;
  execution_time_ms: number;
  config_id?: string;
  config_revision?: number;
  effective_config?: Record<string, StrategyConfigValue>;
  duplicate_threshold?: number;
}

export interface TranscriptChunkResult {
  chunk_id: number;
  video_id: string;
  youtube_id: string;
  topic: string;
  start_time_ms: number;
  end_time_ms: number;
  text: string;
  score: number;
  frame_image_url: string;
  frame_preview_url?: string;
  frame_number: number;
  nearest_timestamp_ms: number | null;
}

export interface TranscriptChunkSearchResponse {
  results: TranscriptChunkResult[];
  total: number;
  execution_time_ms: number;
}

export interface TranscriptSegment {
  start_ms: number;
  end_ms: number;
  text: string;
  speaker: string | null;
}

export interface TranscriptResponse {
  video_id: string;
  segments: TranscriptSegment[];
}
