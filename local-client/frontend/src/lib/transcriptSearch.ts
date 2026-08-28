import type { TranscriptSearchAlgorithmId } from "@/types";

export function buildTranscriptSearchPayload(
  query: string,
  topK: number,
  topicFilter: string | undefined,
  algorithm: TranscriptSearchAlgorithmId,
): Record<string, unknown> {
  return {
    query,
    top_k: topK,
    ...(topicFilter ? { topic_filter: topicFilter } : {}),
    algorithm,
  };
}
