import type { TranscriptResult } from "@/types";
import { apiUrl } from "@/lib/api";

interface Props {
  result: TranscriptResult;
  rank: number;
  onClick: (result: TranscriptResult) => void;
}

function formatTimestamp(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60).toString().padStart(2, "0");
  const sec = (totalSec % 60).toString().padStart(2, "0");
  return `${min}:${sec}`;
}

function confidenceColor(score: number): string {
  if (score >= 0.8) return "text-emerald-400";
  if (score >= 0.6) return "text-yellow-400";
  return "text-red-400";
}

export default function TranscriptResultCard({ result, rank, onClick }: Props) {
  const hasFrame = !!result.frame_image_url;
  const imageUrl = result.frame_image_url
    ? (result.frame_image_url.startsWith("http")
        ? result.frame_image_url
        : apiUrl(result.frame_image_url))
    : "";

  return (
    <button
      onClick={() => onClick(result)}
      className="group bg-slate-800 border border-slate-700 rounded-xl overflow-hidden hover:border-emerald-500 hover:shadow-lg hover:shadow-emerald-900/30 transition-all text-left w-full"
    >
      {/* Thumbnail + rank badge — only if nearest frame exists */}
      {hasFrame && (
        <div className="relative aspect-video bg-slate-700 overflow-hidden">
          <img
            src={imageUrl}
            alt={`Frame near transcript in ${result.video_id}`}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
            loading={rank <= 6 ? "eager" : "lazy"}
            fetchPriority={rank <= 3 ? "high" : "auto"}
            decoding="async"
          />
          <span className="absolute top-1 left-1 bg-black/70 text-white text-xs px-1.5 py-0.5 rounded">
            #{rank}
          </span>
          <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition flex items-center justify-center">
            <svg className="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          </div>
        </div>
      )}

      {/* Text content */}
      <div className="p-3 space-y-1.5">
        {/* Video + timestamp */}
        <div className="flex items-center justify-between gap-2">
          <p className="text-xs text-slate-400 truncate font-mono">{result.video_id}</p>
          <p className="text-xs text-slate-500 font-mono shrink-0">
            {formatTimestamp(result.start_time_ms)} — {formatTimestamp(result.end_time_ms)}
          </p>
        </div>

        {/* Transcript snippet */}
        <p className="text-sm text-slate-200 line-clamp-3 leading-relaxed">
          {result.text}
        </p>

        {/* Score */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">
            {result.nearest_frame_id ? `frame ${result.nearest_frame_id.split("_").pop()}` : ""}
          </span>
          <span className={`text-xs font-semibold ${confidenceColor(result.score)}`}>
            {(result.score * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </button>
  );
}
