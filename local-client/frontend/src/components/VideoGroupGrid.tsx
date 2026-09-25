import { useEffect, useMemo, useRef, useState } from "react";
import type { ContextFrame, SearchResult, TranscriptSegment } from "@/types";
import { fetchContextFrames, fetchFrameScores, fetchScoredContext, fetchTranscript, zipFrameImageUrl } from "@/lib/api";
import ResultCard from "./ResultCard";
import VerifyAction from "./VerifyAction";

interface Props {
  results: SearchResult[];
  total: number;
  executionTimeMs: number;
  onCardClick: (result: SearchResult) => void;
  onVerify?: (result: SearchResult) => void;
  showTranscript: boolean;
  // The query text(s) that produced `results` and the per-event weight the
  // active strategy config has (if any) — optional, since not every caller
  // (transcript search's Video view) has a query-events concept to score
  // against. Second-phase display score only; doesn't change ranking/badges.
  queryEvents?: string[];
  eventWeights?: number[];
  // Same slider the original search's own dedup used — kept in sync so
  // Video view's second-phase dedup (context frames) doesn't drift onto a
  // different, stale threshold when the user adjusts it. Defaults to the
  // backend's own default (0.98) when omitted, same as a plain search.
  duplicateThreshold?: number;
}

// ~15s each side of the best frame (≈30s total) — wide enough for context,
// narrow enough to stay readable and avoid pulling in unrelated content
// from outlier frames far from the actual match.
const BEST_FRAME_TRANSCRIPT_RADIUS_MS = 15000;

// Keep the video strip useful without turning a multi-hour video's sparse
// matches into thousands of cards. Every real search hit stays visible; these
// are only supplementary frames around the best hit. Roughly 24 cards gives a
// readable local clip while keeping decode, DOM and rescoring work bounded.
const CONTEXT_TARGET_TOTAL = 24;

// However many neighbors each side needs to bring `matchedCount` up to
// CONTEXT_TARGET_TOTAL, split evenly. A side that runs out (e.g. matches
// already sit near the video's start) can leave the strip slightly short.
function contextExpandPerSide(matchedCount: number): number {
  return Math.max(0, Math.ceil((CONTEXT_TARGET_TOTAL - matchedCount) / 2));
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
    return "ring-2 ring-orange-600 ring-offset-2 ring-offset-cream dark:ring-offset-stone-900";
  }
  if (rankInVideo >= 2 && rankInVideo <= 5) {
    return "ring-2 ring-teal-600/80 ring-offset-1 ring-offset-cream dark:ring-offset-stone-900";
  }
  return "ring-1 ring-stone-300 dark:ring-stone-700";
}

function frameBadge(rankInVideo: number): string | null {
  if (rankInVideo === 1) return "Best";
  if (rankInVideo >= 2 && rankInVideo <= 5) return `Top ${rankInVideo}`;
  return null;
}

// rankInVideo 0 — frameHighlightClass/frameBadge already render that as
// unhighlighted/no-badge, exactly the "this is context, not a match" look
// wanted here, no new styling needed.
function contextFrameToDisplay(videoId: string, fps: number, frame: ContextFrame): DisplayFrame {
  return {
    result: {
      video_id: videoId,
      youtube_id: frame.youtube_id || undefined,
      frame_id: frame.frame_id,
      frame_number: frame.frame_number,
      timestamp_ms: frame.timestamp_ms,
      confidence: 0,
      frame_image_url: zipFrameImageUrl(videoId, frame.timestamp_ms, frame.frame_number),
      fps,
    },
    rankInVideo: 0,
  };
}

type BestDirection = "left" | "right" | "visible";

interface VideoGroupSectionProps {
  videoId: string;
  frames: DisplayFrame[];
  onCardClick: (result: SearchResult) => void;
  onVerify?: (result: SearchResult) => void;
  showTranscript: boolean;
  queryEvents?: string[];
  eventWeights?: number[];
  duplicateThreshold?: number;
}

function VideoGroupSection({ videoId, frames, onCardClick, onVerify, showTranscript, queryEvents, eventWeights, duplicateThreshold }: VideoGroupSectionProps) {
  const elementRef = useRef<HTMLDivElement | null>(null);
  const stripRef = useRef<HTMLDivElement | null>(null);
  const bestFrameRef = useRef<HTMLDivElement | null>(null);
  const bestDirectionRef = useRef<BestDirection>("visible");
  const hasCenteredRef = useRef(false);
  const centeredForFrameIdRef = useRef<string | null>(null);
  const [bestDirection, setBestDirection] = useState<BestDirection>("visible");
  const [isVisible, setIsVisible] = useState(false);
  const userScrolledRef = useRef(false);
  const centeredStripRef = useRef("");
  const [transcriptSegments, setTranscriptSegments] = useState<TranscriptSegment[] | null>(null);
  const [transcriptError, setTranscriptError] = useState<string | null>(null);
  const bestScore = Math.max(...frames.map((f) => f.result.confidence));
  const bestFrame = frames.find((f) => f.rankInVideo === 1);

  // Fill a sparse strip around the best match. Using the earliest-to-latest
  // match span is unsafe for S videos: two hits hours apart used to pull every
  // indexed frame between them into the page. Real hits remain in `frames`.
  const framesKey = JSON.stringify(frames.map(f => [
    f.result.frame_id, f.result.timestamp_ms, f.result.frame_number,
  ]));
  const needsContext = isVisible && frames.length > 0 && frames.length < CONTEXT_TARGET_TOTAL;
  const contextKey = JSON.stringify([videoId, framesKey, queryEvents, eventWeights, duplicateThreshold]);
  const [contextState, setContextState] = useState<{ key: string; framesKey: string; frames: DisplayFrame[]; scores: Record<string, number> | null } | null>(null);
  useEffect(() => {
    if (!needsContext) return;
    const controller = new AbortController();
    const focusMs = bestFrame?.result.timestamp_ms ?? frames[0].result.timestamp_ms;
    const startMs = focusMs;
    const endMs = focusMs;
    const matchedIds = new Set(frames.map(f => f.result.frame_id));
    const applyContext = (res: Awaited<ReturnType<typeof fetchScoredContext>>) => {
      if (controller.signal.aborted) return;
      const additions = [...res.before, ...res.middle, ...res.after]
        .filter(f => !matchedIds.has(f.frame_id))
        .map(f => contextFrameToDisplay(videoId, res.fps, f));
      setContextState({ key: contextKey, framesKey, frames: additions, scores: res.scores });
    };
    const request = queryEvents?.length
      ? fetchScoredContext(videoId, startMs, endMs, contextExpandPerSide(frames.length),
          frames.map(f => f.result), queryEvents, eventWeights, duplicateThreshold, controller.signal, applyContext)
      : fetchContextFrames(videoId, startMs, endMs, contextExpandPerSide(frames.length), controller.signal)
          .then(res => ({ ...res, scores: null }));
    request
      .then(applyContext)
      .catch(async () => {
        if (controller.signal.aborted) return;
        const scores = queryEvents?.length
          ? await fetchFrameScores(queryEvents, frames.map(f => f.result.frame_id), eventWeights, duplicateThreshold, controller.signal)
          : null;
        if (!controller.signal.aborted) setContextState({ key: contextKey, framesKey, frames: [], scores });
      });
    return () => controller.abort();
    // The key tracks frame identities and times without depending on array identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [needsContext, contextKey]);

  const displayFrames = useMemo(() => {
    const additions = needsContext && contextState?.framesKey === framesKey ? contextState.frames : [];
    return additions.length ? [...frames, ...additions].sort((a, b) =>
      a.result.frame_number - b.result.frame_number
    ) : frames;
  }, [frames, needsContext, contextState, framesKey]);

  // Score the completed strip once, using the query that produced its matches.
  const frameIdsKey = displayFrames.map(f => f.result.frame_id).join(",");
  const scoreKey = JSON.stringify([videoId, frameIdsKey, queryEvents, eventWeights, duplicateThreshold]);
  const [scoreState, setScoreState] = useState<{ key: string; scores: Record<string, number> | null } | null>(null);
  const frameScores = needsContext
    ? (contextState?.key === contextKey ? contextState.scores : null)
    : (scoreState?.key === scoreKey ? scoreState.scores : null);
  useEffect(() => {
    if (!isVisible || needsContext || !queryEvents?.length || !frameIdsKey) return;
    const controller = new AbortController();
    fetchFrameScores(queryEvents, frameIdsKey.split(","), eventWeights, duplicateThreshold, controller.signal)
      .then(scores => {
        if (!controller.signal.aborted) setScoreState({ key: scoreKey, scores });
      });
    return () => controller.abort();
    // scoreKey includes the query, weights, threshold and complete frame set.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isVisible, needsContext, scoreKey]);

  // Duplicate filtering only ever removes *expanded* frames (rankInVideo
  // 0) — a real search match stays visible regardless, so badges/
  // bestFrame/centering (all keyed off rankInVideo) never have to deal
  // with their target disappearing out from under them. Before scores
  // arrive (frameScores === null), show everything optimistically.
  const visibleFrames = useMemo(() => {
    if (frameScores === null) return displayFrames;
    return displayFrames.filter(
      (f) => f.rankInVideo > 0 || frameScores[f.result.frame_id] !== undefined,
    );
  }, [displayFrames, frameScores]);

  // Fetch this video's full transcript once (cheap after the first call);
  // rangeSegments below narrows it down to just a window around the best frame.
  useEffect(() => {
    if (!showTranscript || !isVisible) return;
    let cancelled = false;
    setTranscriptSegments(null);
    setTranscriptError(null);
    const applyTranscript = (res: Awaited<ReturnType<typeof fetchTranscript>>) => {
      if (!cancelled) setTranscriptSegments(res.segments);
    };
    fetchTranscript(videoId, applyTranscript)
      .then(applyTranscript)
      .catch((err) => {
        if (!cancelled) {
          setTranscriptSegments(null);
          setTranscriptError(err.message || "No transcript available");
        }
      });
    return () => { cancelled = true; };
  }, [showTranscript, isVisible, videoId]);

  // Only a window around the best frame, not the full earliest-to-latest
  // span of every retrieved frame — a stray frame near the start/end that's
  // far from the best match is likely noise, and pulling in the transcript
  // all the way out to it made the excerpt too long/unfocused to read.
  const rangeSegments = useMemo(() => {
    if (!transcriptSegments || !bestFrame) return [];
    const centerMs = bestFrame.result.timestamp_ms;
    const minMs = centerMs - BEST_FRAME_TRANSCRIPT_RADIUS_MS;
    const maxMs = centerMs + BEST_FRAME_TRANSCRIPT_RADIUS_MS;
    return transcriptSegments.filter((s) => s.end_ms >= minMs && s.start_ms <= maxMs);
  }, [transcriptSegments, bestFrame]);

  function segmentHighlightRank(seg: TranscriptSegment): number | null {
    let best: number | null = null;
    for (const f of frames) {
      if (f.rankInVideo < 1 || f.rankInVideo > 3) continue;
      const t = f.result.timestamp_ms;
      if (t >= seg.start_ms && t < seg.end_ms && (best === null || f.rankInVideo < best)) {
        best = f.rankInVideo;
      }
    }
    return best;
  }

  // A fresh search can reuse this component instance (same videoId key) with
  // a different best frame — re-center for it instead of leaving the guard
  // permanently tripped from the first-ever search.
  if (bestFrame && centeredForFrameIdRef.current !== bestFrame.result.frame_id) {
    hasCenteredRef.current = false;
    userScrolledRef.current = false;
  }

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

  // Computes scrollLeft directly instead of using scrollIntoView, which is a
  // no-op when the target is already within the visible area (e.g. short
  // strips from a single search) — we want it centered, not just visible.
  function centerBest(smooth: boolean) {
    const strip = stripRef.current;
    const best = bestFrameRef.current;
    if (!strip || !best) return;
    const target = best.offsetLeft - (strip.clientWidth - best.clientWidth) / 2;
    strip.scrollTo({ left: Math.max(0, target), behavior: smooth ? "smooth" : "auto" });
  }

  function jumpToBest() {
    centerBest(true);
  }

  useEffect(() => {
    if (!isVisible) return;
    const handle = setTimeout(() => {
      // Center the best-match frame on first layout instead of leaving the
      // strip scrolled to its start — the user shouldn't have to notice the
      // "jump to best" affordance to find the top result.
      const stripKey = visibleFrames.map(f => f.result.frame_id).join(",");
      if (!hasCenteredRef.current || (!userScrolledRef.current && centeredStripRef.current !== stripKey)) {
        hasCenteredRef.current = true;
        centeredForFrameIdRef.current = bestFrame?.result.frame_id ?? null;
        centerBest(false);
        centeredStripRef.current = stripKey;
      }
      updateBestDirection();
    }, 0);
    window.addEventListener("resize", updateBestDirection);
    return () => {
      clearTimeout(handle);
      window.removeEventListener("resize", updateBestDirection);
    };
  }, [frames, visibleFrames, isVisible]);

  return (
    <div
      ref={elementRef}
      className="border-2 border-stone-800 dark:border-stone-600 rounded overflow-hidden bg-cream dark:bg-stone-800/50 min-h-[174px]"
    >
      {/* Video header */}
      <div className="px-4 py-2 bg-cream-card dark:bg-stone-800 border-b-2 border-stone-800 dark:border-stone-600 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className="font-mono text-sm text-stone-900 dark:text-white font-semibold">
            {videoId}
          </span>
          <span className="text-xs text-stone-500 dark:text-stone-400">
            {frames.length} matched frame{frames.length !== 1 ? "s" : ""}
            {visibleFrames.length > frames.length && ` + ${visibleFrames.length - frames.length} nearby`}
          </span>
        </div>
        <div className="flex items-center gap-3">
          {bestFrame && (
            <span className="hidden sm:inline text-xs text-stone-500 dark:text-stone-400 font-mono">
              best frame {bestFrame.result.frame_number}
            </span>
          )}
          <span className="text-xs text-teal-700 dark:text-teal-400 font-semibold">
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
              className="absolute right-3 top-1/2 z-20 -translate-y-1/2 rounded-full border-2 border-orange-700/70 bg-cream-card/90 dark:bg-stone-950/90 px-3 py-1.5 text-xs font-semibold text-orange-800 dark:text-orange-200 shadow-lg shadow-black/10 dark:shadow-black/30 backdrop-blur hover:border-orange-700 hover:bg-orange-700 hover:text-white transition"
              title="Jump to the best frame in this video"
            >
              {bestDirection === "left" ? "← Best" : "Best →"}
            </button>
          )}
          <div
            ref={stripRef}
            data-frame-strip
            onPointerDown={() => { userScrolledRef.current = true; }}
            onWheel={() => { userScrolledRef.current = true; }}
            onKeyDown={() => { userScrolledRef.current = true; }}
            onScroll={updateBestDirection}
            className="flex overflow-x-auto gap-2 p-3 scrollbar-thin"
          >
            {visibleFrames.map((df, i) => {
              const badge = frameBadge(df.rankInVideo);
              const eventScore = frameScores?.[df.result.frame_id];

              return (
                <div
                  key={df.result.frame_id}
                  ref={(node) => {
                    if (df.rankInVideo === 1) bestFrameRef.current = node;
                  }}
                  className={`relative shrink-0 w-44 rounded overflow-hidden ${frameHighlightClass(df.rankInVideo)}`}
                >
                  {badge && (
                    <span
                      className={`font-retro pointer-events-none absolute right-1 top-1 z-10 rounded px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-white shadow ${df.rankInVideo === 1 ? "bg-orange-700" : "bg-teal-700"
                        }`}
                    >
                      {badge}
                    </span>
                  )}
                  {eventScore !== undefined && (
                    <span
                      className="pointer-events-none absolute left-1 bottom-1 z-10 rounded bg-black/70 px-1 py-0.5 font-mono text-[9px] text-white"
                      title="Second-phase score: weighted sum of cosine similarity across every query event — same scale for matched and expanded frames alike"
                    >
                      {eventScore.toFixed(3)}
                    </span>
                  )}
                  <ResultCard
                    result={df.result}
                    rank={i + 1}
                    imageLoading="lazy"
                    imagePriority={df.rankInVideo === 1 ? "high" : "auto"}
                    onClick={onCardClick}
                    hideBadge={df.rankInVideo <= 5}
                    compact
                  />
                  {onVerify && df.rankInVideo > 0 && <VerifyAction onClick={() => onVerify(df.result)} />}
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <div className="h-28 flex items-center justify-center text-stone-400 dark:text-stone-500 text-xs font-mono select-none">
          Loading frames…
        </div>
      )}

      {/* Transcript for just this box's frame range — shown in full, no
          inner scroll (wraps naturally); top-1/2/3 frame moments highlighted. */}
      {showTranscript && isVisible && (
        <div className="px-4 py-2.5 border-t-2 border-stone-800 dark:border-stone-600 text-xs text-stone-600 dark:text-stone-400 leading-relaxed">
          {transcriptError && <span className="italic text-stone-500">{transcriptError}</span>}
          {!transcriptError && transcriptSegments === null && (
            <span className="italic text-stone-500">Loading transcript…</span>
          )}
          {!transcriptError && transcriptSegments !== null && rangeSegments.length === 0 && (
            <span className="italic text-stone-500">No transcript for this range.</span>
          )}
          {rangeSegments.map((seg, i) => {
            const rank = segmentHighlightRank(seg);
            return (
              <span
                key={i}
                className={
                  rank === 1
                    ? "bg-orange-700/25 text-orange-800 dark:text-orange-300 font-medium rounded px-0.5"
                    : rank === 2 || rank === 3
                      ? "bg-teal-700/20 text-teal-800 dark:text-teal-300 font-medium rounded px-0.5"
                      : ""
                }
              >
                {seg.text}{" "}
              </span>
            );
          })}
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
  onVerify,
  showTranscript,
  queryEvents,
  eventWeights,
  duplicateThreshold,
}: Props) {
  const videoGroups = useMemo(() => buildVideoGroups(results), [results]);

  // Sort-by-video-total toggle — a video whose several matched frames each
  // score decently can beat one whose single best frame barely edges out
  // everyone else's best, which the default (max confidence) can't
  // express. Only fetched once actually toggled on: this scores every
  // *matched* frame across every video group in one request (not the
  // lazily-expanded context frames each section fetches for itself —
  // those are still loading for off-screen videos, so summing them here
  // would be sorting on incomplete data), reusing the exact same
  // second-phase mechanism (frontend-only; nothing here calls anything
  // the backend doesn't already expose).
  const [sortByTotalScore, setSortByTotalScore] = useState(false);
  const [videoTotalScores, setVideoTotalScores] = useState<Record<string, number> | null>(null);
  const allMatchedFrameIdsKey = useMemo(
    () => results.map((r) => r.frame_id).join(","),
    [results],
  );
  useEffect(() => {
    setVideoTotalScores(null);
    if (!sortByTotalScore || !queryEvents || queryEvents.length === 0 || !allMatchedFrameIdsKey) return;
    let cancelled = false;
    fetchFrameScores(queryEvents, allMatchedFrameIdsKey.split(","), eventWeights, duplicateThreshold)
      .then((scores) => {
        if (cancelled || scores === null) return;
        const totals: Record<string, number> = {};
        for (const [videoId, frames] of Array.from(videoGroups.entries())) {
          totals[videoId] = frames.reduce((sum, f) => sum + (scores[f.result.frame_id] ?? 0), 0);
        }
        setVideoTotalScores(totals);
      });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sortByTotalScore, queryEvents, eventWeights, duplicateThreshold, allMatchedFrameIdsKey]);

  const sortedVideoIds = useMemo(() => {
    if (sortByTotalScore && videoTotalScores) {
      return Array.from(videoGroups.keys()).sort(
        (a, b) => (videoTotalScores[b] ?? 0) - (videoTotalScores[a] ?? 0),
      );
    }
    return Array.from(videoGroups.keys()).sort((a, b) => {
      const aMax = Math.max(...videoGroups.get(a)!.map((f) => f.result.confidence));
      const bMax = Math.max(...videoGroups.get(b)!.map((f) => f.result.confidence));
      return bMax - aMax;
    });
  }, [videoGroups, sortByTotalScore, videoTotalScores]);

  return (
    <div className="space-y-4">
      {/* Stats bar */}
      <div className="flex items-center gap-4 text-sm text-stone-500 dark:text-stone-400 font-mono">
        <span>
          <span className="text-stone-900 dark:text-white font-semibold">{total}</span> results
        </span>
        <span>·</span>
        <span>
          <span className="text-stone-900 dark:text-white font-semibold">{executionTimeMs}</span> ms
        </span>
        <span>·</span>
        <span>
          <span className="text-stone-900 dark:text-white font-semibold">{sortedVideoIds.length}</span> videos
        </span>
        {queryEvents && queryEvents.length > 0 && (
          <>
            <span>·</span>
            <button
              type="button"
              onClick={() => setSortByTotalScore((v) => !v)}
              className={`font-retro rounded border-2 px-2 py-0.5 text-xs font-bold uppercase tracking-wide transition ${
                sortByTotalScore
                  ? "border-orange-700 bg-orange-700 text-white"
                  : "border-stone-500 text-stone-600 dark:text-stone-300 hover:border-orange-700 hover:text-orange-700 dark:hover:text-orange-400"
              }`}
              title="Sort videos by the sum of every matched frame's second-phase score, instead of just the single best frame's confidence"
            >
              Sort: {sortByTotalScore ? "Video total" : "Best frame"}
            </button>
            {sortByTotalScore && !videoTotalScores && (
              <span className="italic text-stone-400">scoring…</span>
            )}
          </>
        )}
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
            onVerify={onVerify}
            showTranscript={showTranscript}
            queryEvents={queryEvents}
            eventWeights={eventWeights}
            duplicateThreshold={duplicateThreshold}
          />
        );
      })}

      {results.length === 0 && (
        <p className="text-stone-500 text-center py-16">
          No results. Try a different query.
        </p>
      )}
    </div>
  );
}
