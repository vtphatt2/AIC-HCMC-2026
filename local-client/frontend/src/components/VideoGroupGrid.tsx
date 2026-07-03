import { useEffect, useMemo, useRef, useState } from "react";
import type { SearchResult } from "@/types";
import ResultCard from "./ResultCard";

interface Props {
  results: SearchResult[];
  total: number;
  executionTimeMs: number;
  onCardClick: (result: SearchResult) => void;
}

interface DisplayFrame {
  result: SearchResult;
  rankInVideo: number;
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
    const ranked = [...frames].sort((a, b) => {
      if (b.confidence !== a.confidence) return b.confidence - a.confidence;
      return a.timestamp_ms - b.timestamp_ms;
    });
    const rankByFrameId = new Map(
      ranked.map((result, index) => [result.frame_id, index + 1]),
    );

    output.set(
      videoId,
      sorted.map((result) => ({
        result,
        rankInVideo: rankByFrameId.get(result.frame_id) || 0,
      })),
    );
  }

  return output;
}

function frameHighlightClass(rankInVideo: number): string {
  if (rankInVideo === 1) {
    return "ring-2 ring-emerald-400 ring-offset-2 ring-offset-slate-950";
  }
  if (rankInVideo >= 2 && rankInVideo <= 5) {
    return "ring-2 ring-cyan-400/90 ring-offset-1 ring-offset-slate-950";
  }
  return "ring-1 ring-slate-700";
}

function frameBadge(rankInVideo: number): string | null {
  if (rankInVideo === 1) return "Best";
  if (rankInVideo >= 2 && rankInVideo <= 5) return `Top ${rankInVideo}`;
  return null;
}

type BestDirection = "left" | "right" | "visible";

interface VideoGroupSectionProps {
  videoId: string;
  frames: DisplayFrame[];
  onCardClick: (result: SearchResult) => void;
}

function VideoGroupSection({ videoId, frames, onCardClick }: VideoGroupSectionProps) {
  const elementRef = useRef<HTMLDivElement | null>(null);
  const stripRef = useRef<HTMLDivElement | null>(null);
  const bestFrameRef = useRef<HTMLDivElement | null>(null);
  const bestDirectionRef = useRef<BestDirection>("visible");
  const [bestDirection, setBestDirection] = useState<BestDirection>("visible");
  const [isVisible, setIsVisible] = useState(false);
  const bestScore = Math.max(...frames.map((f) => f.result.confidence));
  const bestFrame = frames.find((f) => f.rankInVideo === 1);

  // Lazy render observer to drop off-screen DOM weight and network load
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: "250px" }
    );
    if (elementRef.current) {
      observer.observe(elementRef.current);
    }
    return () => observer.disconnect();
  }, []);

  function updateBestDirection() {
    const strip = stripRef.current;
    const best = bestFrameRef.current;
    if (!strip || !best) return;

    const stripRect = strip.getBoundingClientRect();
    const bestRect = best.getBoundingClientRect();
    let nextDirection: BestDirection = "visible";
    if (bestRect.left < stripRect.left + 8) {
      nextDirection = "left";
    } else if (bestRect.right > stripRect.right - 8) {
      nextDirection = "right";
    }

    if (bestDirectionRef.current !== nextDirection) {
      bestDirectionRef.current = nextDirection;
      setBestDirection(nextDirection);
    }
  }

  function jumpToBest() {
    bestFrameRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
      inline: "center",
    });
  }

  useEffect(() => {
    if (!isVisible) return;
    const handle = setTimeout(updateBestDirection, 0);
    window.addEventListener("resize", updateBestDirection);
    return () => {
      clearTimeout(handle);
      window.removeEventListener("resize", updateBestDirection);
    };
  }, [frames, isVisible]);

  return (
    <div
      ref={elementRef}
      className="border border-slate-700 rounded-xl overflow-hidden bg-slate-800/50 min-h-[174px]"
    >
      {/* Video header */}
      <div className="px-4 py-2 bg-slate-800 border-b border-slate-700 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className="font-mono text-sm text-white font-semibold">
            {videoId}
          </span>
          <span className="text-xs text-slate-400">
            {frames.length} matched frame{frames.length !== 1 ? "s" : ""}
          </span>
        </div>
        <div className="flex items-center gap-3">
          {bestFrame && (
            <span className="hidden sm:inline text-xs text-slate-400 font-mono">
              best frame {bestFrame.result.frame_number}
            </span>
          )}
          <span className="text-xs text-emerald-400 font-semibold">
            best: {(bestScore * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Horizontal scrollable frame strip */}
      {isVisible ? (
        <div className="relative">
          {bestDirection !== "visible" && (
            <button
              type="button"
              onClick={jumpToBest}
              className="absolute right-3 top-1/2 z-20 -translate-y-1/2 rounded-full border border-emerald-300/60 bg-slate-950/90 px-3 py-1.5 text-xs font-semibold text-emerald-200 shadow-lg shadow-black/30 backdrop-blur hover:border-emerald-300 hover:bg-emerald-500 hover:text-white transition"
              title="Jump to the best frame in this video"
            >
              {bestDirection === "left" ? "← Best" : "Best →"}
            </button>
          )}
          <div
            ref={stripRef}
            onScroll={updateBestDirection}
            className="flex overflow-x-auto gap-2 p-3 scrollbar-thin"
          >
            {frames.map((df, i) => {
              const badge = frameBadge(df.rankInVideo);

              return (
                <div
                  key={df.result.frame_id}
                  ref={(node) => {
                    if (df.rankInVideo === 1) bestFrameRef.current = node;
                  }}
                  className={`relative shrink-0 w-44 rounded-lg overflow-hidden ${frameHighlightClass(df.rankInVideo)}`}
                >
                  {badge && (
                    <span
                      className={`pointer-events-none absolute right-1 top-1 z-10 rounded px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-white shadow ${df.rankInVideo === 1 ? "bg-emerald-500" : "bg-cyan-500"
                        }`}
                    >
                      {badge}
                    </span>
                  )}
                  <ResultCard
                    result={df.result}
                    rank={i + 1}
                    onClick={onCardClick}
                    hideBadge={df.rankInVideo <= 5}
                    compact
                  />
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <div className="h-28 flex items-center justify-center text-slate-500 text-xs font-mono select-none">
          Loading frames…
        </div>
      )}
    </div>
  );
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
          ...videoGroups.get(a)!.map((f) => f.result.confidence)
        );
        const bMax = Math.max(
          ...videoGroups.get(b)!.map((f) => f.result.confidence)
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

        return (
          <VideoGroupSection
            key={videoId}
            videoId={videoId}
            frames={frames}
            onCardClick={onCardClick}
          />
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
