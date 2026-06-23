import { useMemo } from "react";
import type { SearchResult } from "@/types";
import ResultCard from "./ResultCard";

interface Props {
  results: SearchResult[];
  total: number;
  executionTimeMs: number;
  onCardClick: (result: SearchResult) => void;
}

const FILLER_INTERVAL = 10;

function buildFillerFrameUrl(videoId: string, frameNumber: number): string {
  const padded = String(frameNumber).padStart(6, "0");
  return `/static/frames/${videoId}/${padded}.jpg`;
}

interface DisplayFrame {
  result: SearchResult;
  isQueried: boolean;
}

function buildVideoGroups(results: SearchResult[]): Map<string, DisplayFrame[]> {
  const groups = new Map<string, SearchResult[]>();
  for (const r of results) {
    const list = groups.get(r.video_id) || [];
    list.push(r);
    groups.set(r.video_id, list);
  }

  const output = new Map<string, DisplayFrame[]>();

  for (const [videoId, frames] of Array.from(groups.entries())) {
    const sorted = [...frames].sort((a, b) => a.frame_number - b.frame_number);
    const queriedSet = new Set(sorted.map((f) => f.frame_number));
    const displayFrames: DisplayFrame[] = [];

    for (let i = 0; i < sorted.length; i++) {
      const current = sorted[i];

      if (i > 0) {
        const prev = sorted[i - 1];
        const gap = current.frame_number - prev.frame_number;

        if (gap > FILLER_INTERVAL) {
          for (
            let fn = prev.frame_number + FILLER_INTERVAL;
            fn < current.frame_number;
            fn += FILLER_INTERVAL
          ) {
            if (!queriedSet.has(fn)) {
              displayFrames.push({
                result: {
                  video_id: videoId,
                  frame_id: `${videoId}_${String(fn).padStart(6, "0")}`,
                  frame_number: fn,
                  timestamp_ms: Math.round((fn / current.fps) * 1000),
                  confidence: -1,
                  frame_image_url: buildFillerFrameUrl(videoId, fn),
                  fps: current.fps,
                },
                isQueried: false,
              });
            }
          }
        }
      }

      displayFrames.push({ result: current, isQueried: true });
    }

    output.set(videoId, displayFrames);
  }

  return output;
}

export default function VideoGroupGrid({
  results,
  total,
  executionTimeMs,
  onCardClick,
}: Props) {
  const videoGroups = useMemo(() => buildVideoGroups(results), [results]);
  const sortedVideoIds = useMemo(
    () =>
      Array.from(videoGroups.keys()).sort((a, b) => {
        const aMax = Math.max(
          ...videoGroups.get(a)!.filter((f) => f.isQueried).map((f) => f.result.confidence)
        );
        const bMax = Math.max(
          ...videoGroups.get(b)!.filter((f) => f.isQueried).map((f) => f.result.confidence)
        );
        return bMax - aMax;
      }),
    [videoGroups]
  );

  return (
    <div className="space-y-4">
      {/* Stats bar */}
      <div className="flex items-center gap-4 text-sm text-slate-400">
        <span>
          <span className="text-white font-semibold">{total}</span> results
        </span>
        <span>·</span>
        <span>
          <span className="text-white font-semibold">{executionTimeMs}</span> ms
        </span>
        <span>·</span>
        <span>
          <span className="text-white font-semibold">{sortedVideoIds.length}</span> videos
        </span>
      </div>

      {/* Video groups */}
      {sortedVideoIds.map((videoId) => {
        const frames = videoGroups.get(videoId)!;
        const queriedCount = frames.filter((f) => f.isQueried).length;
        const bestScore = Math.max(
          ...frames.filter((f) => f.isQueried).map((f) => f.result.confidence)
        );

        return (
          <div
            key={videoId}
            className="border border-slate-700 rounded-xl overflow-hidden bg-slate-800/50"
          >
            {/* Video header */}
            <div className="px-4 py-2 bg-slate-800 border-b border-slate-700 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="font-mono text-sm text-white font-semibold">
                  {videoId}
                </span>
                <span className="text-xs text-slate-400">
                  {queriedCount} matched frame{queriedCount !== 1 ? "s" : ""}
                </span>
              </div>
              <span className="text-xs text-emerald-400 font-semibold">
                best: {(bestScore * 100).toFixed(1)}%
              </span>
            </div>

            {/* Horizontal scrollable frame strip */}
            <div className="flex overflow-x-auto gap-2 p-3 scrollbar-thin">
              {frames.map((df, i) => (
                <div
                  key={df.result.frame_id + (df.isQueried ? "" : "-filler")}
                  className={`shrink-0 w-44 rounded-lg overflow-hidden ${
                    df.isQueried
                      ? "ring-2 ring-yellow-400 ring-offset-1 ring-offset-slate-900"
                      : "opacity-60"
                  }`}
                >
                  <ResultCard
                    result={df.result}
                    rank={df.isQueried ? i + 1 : -1}
                    onClick={onCardClick}
                    hideBadge={!df.isQueried}
                    compact
                  />
                </div>
              ))}
            </div>
          </div>
        );
      })}

      {results.length === 0 && (
        <p className="text-slate-500 text-center py-16">
          No results. Try a different query.
        </p>
      )}
    </div>
  );
}
