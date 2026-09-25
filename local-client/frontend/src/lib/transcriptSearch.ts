import type {
  SearchResult,
  TranscriptChunkResult,
  TranscriptSearchAlgorithmId,
} from "@/types";

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

export function transcriptChunksToFrameResults(
  chunks: ReadonlyArray<TranscriptChunkResult>,
): SearchResult[] {
  return chunks.map((chunk) => {
    const midMs = (chunk.start_time_ms + chunk.end_time_ms) / 2;
    const computedFrameNumber = Math.floor((midMs / 1000) * 25);
    const frameNumber = chunk.frame_number > 0 ? chunk.frame_number : computedFrameNumber;
    const timestampMs = chunk.nearest_timestamp_ms ?? Math.round(midMs);
    const path = `/api/zip-frame/${encodeURIComponent(chunk.video_id)}/${timestampMs}`;
    // N is VFR: a nominal frame number derived from the transcript midpoint
    // cannot identify a source picture. Only a nearest indexed frame can.
    const fallbackImageUrl = chunk.video_id.startsWith("N")
      ? (chunk.nearest_timestamp_ms !== null && Number.isSafeInteger(chunk.frame_number) &&
          chunk.frame_number >= 0 ? `${path}?frame_number=${chunk.frame_number}&v=7` : "")
      : path;

    return {
      video_id: chunk.video_id,
      youtube_id: chunk.youtube_id,
      frame_id: `${chunk.video_id}_${String(frameNumber).padStart(6, "0")}`,
      frame_number: frameNumber,
      timestamp_ms: timestampMs,
      confidence: chunk.score,
      frame_image_url: chunk.video_id.startsWith("N") ? fallbackImageUrl :
        (chunk.frame_image_url || fallbackImageUrl),
      frame_preview_url: chunk.frame_preview_url,
      fps: 25,
    };
  });
}

export interface HighlightedTranscriptPart {
  text: string;
  highlighted: boolean;
}

function normalizeVietnameseWord(value: string): string {
  return value
    .toLocaleLowerCase("vi")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/g, "d");
}

function levenshteinDistance(left: string, right: string): number {
  if (left === right) return 0;
  if (!left) return right.length;
  if (!right) return left.length;

  let previous = Array.from({ length: right.length + 1 }, (_value, index) => index);
  for (let leftIndex = 1; leftIndex <= left.length; leftIndex += 1) {
    const current = [leftIndex];
    for (let rightIndex = 1; rightIndex <= right.length; rightIndex += 1) {
      current[rightIndex] = Math.min(
        current[rightIndex - 1] + 1,
        previous[rightIndex] + 1,
        previous[rightIndex - 1] + (left[leftIndex - 1] === right[rightIndex - 1] ? 0 : 1),
      );
    }
    previous = current;
  }
  return previous[right.length];
}

function wordsAreSimilar(word: string, queryWord: string): boolean {
  if (word === queryWord) return true;
  const shortest = Math.min(word.length, queryWord.length);
  const longest = Math.max(word.length, queryWord.length);
  if (shortest < 4) return false;
  const allowedDistance = longest <= 6 ? 1 : Math.max(1, Math.floor(longest * 0.25));
  return levenshteinDistance(word, queryWord) <= allowedDistance;
}

export function highlightTranscriptText(
  text: string,
  query: string,
): HighlightedTranscriptPart[] {
  const queryWords = query
    .split(/[^0-9A-Za-zÀ-ỹĐđ]+/)
    .map(normalizeVietnameseWord)
    .filter(Boolean);
  if (queryWords.length === 0) return [{ text, highlighted: false }];

  return text
    .split(/([0-9A-Za-zÀ-ỹĐđ]+)/)
    .filter((part) => part.length > 0)
    .map((part) => {
      const normalized = normalizeVietnameseWord(part);
      const highlighted = /^[a-z0-9]+$/.test(normalized)
        && queryWords.some((queryWord) => wordsAreSimilar(normalized, queryWord));
      return { text: part, highlighted };
    });
}
