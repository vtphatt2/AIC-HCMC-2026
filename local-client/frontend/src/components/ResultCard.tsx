import { forwardRef, useState } from "react";
import type { SearchResult } from "@/types";
import { apiUrl } from "@/lib/api";

interface Props {
  result: SearchResult;
  rank: number;
  onClick: (result: SearchResult) => void;
  hideBadge?: boolean;
  compact?: boolean;
  badgeLabel?: string;
  focused?: boolean;
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
  { result, rank, onClick, hideBadge, compact, badgeLabel, focused },
  ref,
) {
  const imageUrl = result.frame_image_url.startsWith("http")
    ? result.frame_image_url
    : apiUrl(result.frame_image_url);
  const previewUrl = result.frame_preview_url
    ? (result.frame_preview_url.startsWith("http") ? result.frame_preview_url : apiUrl(result.frame_preview_url))
    : null;
  const [loaded, setLoaded] = useState(false);
  const showPreview = previewUrl && !loaded;

  return (
    <button
      ref={ref}
      onClick={() => onClick(result)}
      className={`group bg-cream-card dark:bg-stone-800 border-2 rounded overflow-hidden transition-all text-left w-full ${
        focused
          ? "border-orange-700 dark:border-orange-500 ring-2 ring-orange-600 ring-offset-2 ring-offset-cream dark:ring-offset-stone-900"
          : "border-stone-800 dark:border-stone-600 hover:border-orange-700 dark:hover:border-orange-500"
      }`}
    >
      {/* Frame image */}
      <div className="relative aspect-video bg-stone-200 dark:bg-stone-700 overflow-hidden">
        {/* Fast blurry placeholder — shown until the sharp image finishes loading */}
        {showPreview && (
          <img
            src={previewUrl}
            alt=""
            aria-hidden="true"
            className="absolute inset-0 w-full h-full object-cover scale-110 blur-sm"
          />
        )}
        <img
          src={imageUrl}
          alt={`Frame ${result.frame_number} of ${result.video_id}`}
          onLoad={() => setLoaded(true)}
          className={`relative w-full h-full object-cover group-hover:scale-105 transition-all duration-300 ${
            showPreview ? "opacity-0" : "opacity-100"
          }`}
          loading={rank >= 0 && rank <= 12 ? "eager" : "lazy"}
          fetchPriority={rank >= 0 && rank <= 6 ? "high" : "auto"}
          decoding="async"
        />
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
