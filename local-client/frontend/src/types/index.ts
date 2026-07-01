export interface Strategy {
  id: string;
  name: string;
  description: string;
  author: string;
  version: string;
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
  translateSemantic: boolean;
  translatedQuery: string;
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
}

export interface SearchResponse {
  results: SearchResult[];
  strategy_id: string;
  total: number;
  execution_time_ms: number;
}

export interface TranscriptResult {
  video_id: string;
  youtube_id: string;
  start_time_ms: number;
  end_time_ms: number;
  text: string;
  score: number;
  nearest_frame_id: string | null;
  nearest_timestamp_ms: number | null;
  frame_image_url: string | null;
  normalized_query?: string;
  match_type?: string;
  window_text?: string;
  window_start_time_ms?: number | null;
  window_end_time_ms?: number | null;
}

export interface TranscriptSearchResponse {
  results: TranscriptResult[];
  total: number;
  execution_time_ms: number;
}
