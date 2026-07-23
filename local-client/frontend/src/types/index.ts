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
  // Present only when FRAME_IMAGE_SOURCE=youtube_precise: a fast, blurry
  // placeholder to show immediately while frame_image_url is still extracting.
  frame_preview_url?: string;
  // Present only for multi-step temporal matches: every frame in the matched
  // chain, in step order (this result itself is the chain's closing frame).
  steps?: SearchResult[];
}

export interface SearchResponse {
  results: SearchResult[];
  strategy_id: string;
  total: number;
  execution_time_ms: number;
  config_id?: string;
  config_revision?: number;
  effective_config?: Record<string, StrategyConfigValue>;
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
