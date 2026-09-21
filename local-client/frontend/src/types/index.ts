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
  /** English result kept separate from the query the user entered. */
  translatedSemanticQuery?: string;
  /** When enabled, search submits translatedSemanticQuery instead of semanticQuery. */
  translationEnabled?: boolean;
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

// A row IS a CSV export row — kis/qa: one frame; trake: every frame of one
// candidate, all from `videoId`. No id/fps/youtubeId here: nothing is stored
// that the backend can recompute (fps, youtube_id) or that's implied by the
// row's own position (rank = array order), so the on-disk file is the
// literal export CSV, not a separate JSON model of it.
export interface SubmissionRow {
  videoId: string;
  frames: number[];
  answer?: string; // qa only
}

export interface SubmissionState {
  session: string;
  queryType: SubmissionQueryType;
  // TRAKE only: row index "Add to submission" appends the next frame into.
  // rows.length once "+ New candidate" has been pressed (nothing to append
  // to yet — the next add starts a fresh row there).
  draftRowIndex: number;
  rows: SubmissionRow[];
  createdAt: number;
  updatedAt: number;
}

export interface SubmissionSessionSummary {
  session: string;
  queryType: SubmissionQueryType;
  rowCount: number;
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
  algorithm: TranscriptSearchAlgorithmId;
}

export type TranscriptSearchAlgorithmId = "semantic" | "lexical" | "fuzzy";

export interface TranscriptSearchAlgorithm {
  id: TranscriptSearchAlgorithmId;
  name: string;
  available: boolean;
  supports_topic_filter: boolean;
  description: string;
}

export interface TranscriptSearchAlgorithmResponse {
  default: TranscriptSearchAlgorithmId;
  algorithms: TranscriptSearchAlgorithm[];
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

// ── Video view strip context frames ─────────────────────────────────────
// Neighboring indexed keyframes around a video's matched-frame cluster —
// not search hits, just what's available in the system to browse when a
// video only matched a tight handful. See VideoGroupGrid.tsx.

export interface ContextFrame {
  frame_id: string;
  video_id: string;
  frame_number: number;
  timestamp_ms: number;
  youtube_id: string;
}

export interface ContextFramesResponse {
  fps: number;
  before: ContextFrame[];
  // Every indexed frame between the requested start/end — including
  // whatever frame(s) sit exactly at those bounds, i.e. the caller's own
  // matched frames if start/end came from their min/max timestamp. The
  // caller already knows those frame_ids and dedupes against them; this
  // response doesn't try to guess which of the in-range frames it is.
  middle: ContextFrame[];
  after: ContextFrame[];
}
