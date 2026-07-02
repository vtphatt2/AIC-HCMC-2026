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

export interface TranscriptChunkResult {
  chunk_id: number;
  video_id: string;
  topic: string;
  start_time_ms: number;
  end_time_ms: number;
  text: string;
  score: number;
}

export interface TranscriptChunkSearchResponse {
  results: TranscriptChunkResult[];
  total: number;
  execution_time_ms: number;
}
