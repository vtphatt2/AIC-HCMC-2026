import { forwardRef, useEffect, useRef, useState } from "react";
import type { SearchResult } from "@/types";
import { apiUrl } from "@/lib/api";
import { useFrameImage } from "@/lib/useFrameImage";
import { cardImageUrl, isConstrainedConnection, needsOriginalFrame, prefetchOriginalFrame, rememberCardImage } from "@/lib/frameImages";

interface Props {
  result: SearchResult;
  rank: number;
  onClick: (result: SearchResult) => void;
  hideBadge?: boolean;
  compact?: boolean;
  badgeLabel?: string;
  focused?: boolean;
  imageLoading?: "eager" | "lazy";
  imagePriority?: "high" | "auto";
}

function formatTimestamp(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60).toString().padStart(2, "0");
  const sec = (totalSec % 60).toString().padStart(2, "0");
  return `${min}:${sec}`;
}

function confidenceColor(score: number): string {
  if (score >= 0.8) return "text-teal-700 dark:text-teal-400";
  if (score >= 0.6) return "text-amber-700 dark:text-amber-400";
  return "text-rose-700 dark:text-rose-400";
}

const ResultCard = forwardRef<HTMLButtonElement, Props>(function ResultCard(
  { result, rank, onClick, hideBadge, compact, badgeLabel, focused, imageLoading,
    imagePriority },
  ref,
) {
  const serverImageUrl = result.frame_image_url.startsWith("http")
    ? result.frame_image_url
    : apiUrl(result.frame_image_url);
  // Decodes here instead of on the backend when NEXT_PUBLIC_FRAME_DECODE=client
  // and the browser has WebCodecs; otherwise this is serverImageUrl unchanged.
  const imageUrl = useFrameImage(serverImageUrl, result.frame_image_url);
  const cardUrl = imageUrl === serverImageUrl ? cardImageUrl(imageUrl) : null;
  const [failedCardUrl, setFailedCardUrl] = useState<string | null>(null);
  const [failedOriginalUrl, setFailedOriginalUrl] = useState<string | null>(null);
  const [needsOriginal, setNeedsOriginal] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [recovering, setRecovering] = useState(false);
  const [imageUnavailable, setImageUnavailable] = useState(false);
  const imageBoxRef = useRef<HTMLDivElement>(null);
  const retryTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const useCard = Boolean(cardUrl && failedCardUrl !== cardUrl);
  const constrainedConnection = isConstrainedConnection();
  const useOriginal = useCard && needsOriginal && !constrainedConnection && failedOriginalUrl !== imageUrl;
  const retryUrl = (url: string) => retryCount
    ? `${url}${url.includes("?") ? "&" : "?"}image_retry=${retryCount}` : url;
  const originalRequestUrl = retryUrl(imageUrl);
  const cardRequestUrl = cardUrl ? retryUrl(cardUrl) : null;
  const hoverTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  function cancelPrefetch() {
    if (hoverTimer.current !== null) clearTimeout(hoverTimer.current);
    hoverTimer.current = null;
  }
  useEffect(() => cancelPrefetch, [serverImageUrl]);
  useEffect(() => {
    setFailedCardUrl(null);
    setFailedOriginalUrl(null);
    setRetryCount(0);
    setRecovering(false);
    setImageUnavailable(false);
    return () => {
      if (retryTimer.current !== null) clearTimeout(retryTimer.current);
      retryTimer.current = null;
    };
  }, [imageUrl]);
  function retryLater() {
    if (retryCount >= 2) {
      setImageUnavailable(true);
      return;
    }
    if (retryTimer.current !== null) return;
    retryTimer.current = setTimeout(() => {
      retryTimer.current = null;
      setFailedCardUrl(null);
      setFailedOriginalUrl(null);
      setRetryCount((count) => count + 1);
    }, 300 * (retryCount + 1));
  }
  useEffect(() => {
    const box = imageBoxRef.current;
    if (!box) return;
    const measure = () => setNeedsOriginal(needsOriginalFrame(
      box.getBoundingClientRect().width,
      window.devicePixelRatio || 1,
    ));
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(box);
    return () => observer.disconnect();
  }, []);
  function prefetch() {
    if (useCard) prefetchOriginalFrame(serverImageUrl);
  }
  const previewUrl = result.frame_preview_url
    ? (result.frame_preview_url.startsWith("http") ? result.frame_preview_url : apiUrl(result.frame_preview_url))
    : null;
  const [loadedUrl, setLoadedUrl] = useState<string | null>(null);
  const showPreview = previewUrl && loadedUrl !== imageUrl;
  const loading = imageLoading ?? (rank >= 0 && rank <= 12 ? "eager" : "lazy");

  return (
    <button
      ref={ref}
      onClick={() => onClick(result)}
      onMouseEnter={() => { cancelPrefetch(); hoverTimer.current = setTimeout(prefetch, 150); }}
      onMouseLeave={cancelPrefetch}
      onFocus={prefetch}
      className={`group bg-cream-card dark:bg-stone-800 border-2 rounded overflow-hidden transition-all text-left w-full ${
        focused
          ? "border-orange-700 dark:border-orange-500 ring-2 ring-orange-600 ring-offset-2 ring-offset-cream dark:ring-offset-stone-900"
          : "border-stone-800 dark:border-stone-600 hover:border-orange-700 dark:hover:border-orange-500"
      }`}
    >
      {/* Frame image */}
      <div ref={imageBoxRef} className="relative aspect-video bg-stone-200 dark:bg-stone-700 overflow-hidden">
        {/* Fast blurry placeholder — shown until the sharp image finishes loading */}
        {showPreview && (
          <img
            src={previewUrl}
            alt=""
            aria-hidden="true"
            loading={loading}
            className="absolute inset-0 w-full h-full object-cover scale-110 blur-sm"
          />
        )}
        <picture>
          {useOriginal && (
            <source srcSet={originalRequestUrl} />
          )}
          {useCard && constrainedConnection && (
            <source type="image/webp" srcSet={retryUrl(cardImageUrl(imageUrl, "webp")!)} />
          )}
          <img
            key={imageUrl}
            src={useCard ? cardRequestUrl! : originalRequestUrl}
            alt={`Frame ${result.frame_number} of ${result.video_id}`}
            onLoad={(event) => {
              setRecovering(false);
              setImageUnavailable(false);
              setLoadedUrl(imageUrl);
              rememberCardImage(serverImageUrl, event.currentTarget.currentSrc || event.currentTarget.src);
            }}
            onError={(event) => {
              // A <picture> source can fail even when the <img> fallback works.
              // Try the other size, then retry transient tunnel/decode errors.
              setRecovering(true);
              const attemptedOriginal = event.currentTarget.currentSrc ===
                new URL(originalRequestUrl, document.baseURI).href;
              if (attemptedOriginal) {
                if (failedCardUrl === cardUrl || !cardUrl) retryLater();
                else setFailedOriginalUrl(imageUrl);
              } else if (failedOriginalUrl === imageUrl) retryLater();
              else setFailedCardUrl(cardUrl);
            }}
            className={`relative w-full h-full object-cover group-hover:scale-105 transition-all duration-300 ${
              showPreview || recovering ? "opacity-0" : "opacity-100"
            }`}
            loading={loading}
            fetchPriority={imagePriority ?? (loading === "eager" && rank >= 0 && rank <= 6 ? "high" : "auto")}
            decoding="async"
          />
        </picture>
        {imageUnavailable && <span className="absolute inset-0 flex items-center justify-center text-xs text-stone-600 dark:text-stone-300">Frame unavailable</span>}
        {/* Rank / step badge */}
        {!hideBadge && (
          <span className="font-retro absolute top-1 left-1 bg-stone-900/80 text-white text-[10px] font-bold px-1.5 py-0.5 rounded">
            {badgeLabel ?? `#${rank}`}
          </span>
        )}
        {/* Play overlay on hover */}
        <div className="absolute inset-0 bg-stone-900/40 opacity-0 group-hover:opacity-100 transition flex items-center justify-center">
          <svg className="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 24 24">
            <path d="M8 5v14l11-7z" />
          </svg>
        </div>
      </div>

      {/* Metadata */}
      <div className={compact ? "p-1 space-y-0" : "p-2 space-y-0.5"}>
        {!compact && (
          <p className="text-xs text-stone-500 dark:text-stone-400 truncate font-mono">{result.video_id}</p>
        )}
        <div className="flex items-center justify-between">
          <span className="text-sm text-stone-900 dark:text-white font-medium font-mono">
            {formatTimestamp(result.timestamp_ms)}
          </span>
          {result.confidence >= 0 && (
            <span className={`text-xs font-semibold ${confidenceColor(result.confidence)}`}>
              {(result.confidence * 100).toFixed(1)}%
            </span>
          )}
        </div>
        <p className="text-xs text-stone-500">frame {result.frame_number}</p>
      </div>
    </button>
  );
});

export default ResultCard;
