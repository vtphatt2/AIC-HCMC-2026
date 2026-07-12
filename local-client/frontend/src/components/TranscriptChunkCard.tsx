import type { TranscriptChunkResult } from "@/types";
import { apiUrl } from "@/lib/api";

interface Props {
  result: TranscriptChunkResult;
  rank: number;
  onClick: (result: TranscriptChunkResult) => void;
  onFrameClick?: (
    videoId: string,
    youtubeId: string,
    frameNumber: number,
    timestampMs: number,
    frameImageUrl: string,
  ) => void;
}

const TOPIC_COLORS: Record<string, string> = {
  "Ẩm thực":     "bg-orange-700 text-orange-200",
  "Công nghệ":   "bg-blue-700 text-blue-200",
  "Du lịch":     "bg-cyan-700 text-cyan-200",
  "Thể thao":    "bg-red-700 text-red-200",
  "Giáo dục":    "bg-purple-700 text-purple-200",
  "Kinh tế":     "bg-amber-700 text-amber-200",
  "Sức khỏe":    "bg-emerald-700 text-emerald-200",
  "Giải trí":    "bg-pink-700 text-pink-200",
  "Thời sự":     "bg-slate-600 text-slate-200",
  "Văn hóa":     "bg-indigo-700 text-indigo-200",
  "Đời sống":    "bg-teal-700 text-teal-200",
  "Môi trường":  "bg-green-700 text-green-200",
  "Giao thông":  "bg-yellow-700 text-yellow-200",
  "Pháp luật":   "bg-stone-700 text-stone-200",
};

function formatTimestamp(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60)
    .toString()
    .padStart(2, "0");
  const s = (totalSec % 60).toString().padStart(2, "0");
  if (h > 0) return `${h}:${m}:${s}`;
  return `${m}:${s}`;
}

function formatTimestampCompact(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const m = Math.floor(totalSec / 60)
    .toString()
    .padStart(2, "0");
  const s = (totalSec % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function confidenceColor(score: number): string {
  if (score >= 0.8) return "text-emerald-400";
  if (score >= 0.6) return "text-yellow-400";
  return "text-red-400";
}

function topicColor(topic: string): string {
  return TOPIC_COLORS[topic] ?? "bg-slate-700 text-slate-300";
}

export default function TranscriptChunkCard({ result, rank, onClick, onFrameClick }: Props) {
  const imageUrl = result.frame_image_url
    ? result.frame_image_url.startsWith("http")
      ? result.frame_image_url
      : apiUrl(result.frame_image_url)
    : "";

  function handleImageError(e: React.SyntheticEvent<HTMLImageElement>) {
    (e.target as HTMLImageElement).style.display = "none";
  }

  function handleFrameClick(e: React.MouseEvent) {
    e.stopPropagation();
    const targetMs = result.nearest_timestamp_ms !== null
      ? result.nearest_timestamp_ms
      : result.start_time_ms;
    onFrameClick?.(
      result.video_id,
      result.youtube_id,
      result.frame_number,
      targetMs,
      result.frame_image_url,
    );
  }

  return (
    <div className="group bg-white dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden hover:border-emerald-500 transition-all flex flex-col h-full shadow-md shadow-black/5 dark:shadow-black/25">
      {/* Header bar */}
      <div className="px-3 py-2 bg-slate-100 dark:bg-slate-800/80 border-b border-slate-200 dark:border-slate-700/60 flex items-center gap-2 flex-wrap text-xs select-none">
        <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-slate-900/80 dark:bg-black/40 text-white font-mono">
          #{rank}
        </span>
        <span
          className={`text-[11px] font-semibold px-2 py-0.5 rounded-full ${topicColor(result.topic)}`}
        >
          {result.topic}
        </span>
        <span className="text-[10px] text-slate-500 font-mono">
          {result.video_id}
        </span>
        <div className="flex-1" />
        <span className="text-xs text-slate-700 dark:text-slate-300 font-mono bg-slate-200 dark:bg-slate-700/60 px-2 py-0.5 rounded">
          {formatTimestamp(result.start_time_ms)} – {formatTimestamp(result.end_time_ms)}
        </span>
        <span className={`text-xs font-semibold ${confidenceColor(result.score)}`}>
          {(result.score * 100).toFixed(1)}%
        </span>
      </div>

      {result.frame_image_url && (
        <button
          onClick={handleFrameClick}
          className="relative w-full aspect-video bg-slate-200 dark:bg-slate-700 overflow-hidden text-left border-b border-slate-200 dark:border-slate-700/60 group/btn shrink-0"
        >
          <img
            src={imageUrl}
            alt={`Nearest frame ${result.frame_number} for transcript segment`}
            className="w-full h-full object-cover group-hover/btn:scale-105 transition-transform duration-300"
            loading="lazy"
            decoding="async"
            onError={handleImageError}
          />
          {/* Timestamp badge */}
          {result.nearest_timestamp_ms !== null && (
            <span className="absolute bottom-1.5 left-1.5 bg-black/85 text-white text-[10px] px-1.5 py-0.5 rounded font-mono leading-tight shadow border border-slate-800/65">
              {formatTimestampCompact(result.nearest_timestamp_ms)} (Match)
            </span>
          )}
          {/* Play overlay on hover */}
          <div className="absolute inset-0 bg-black/40 opacity-0 group-hover/btn:opacity-100 transition flex items-center justify-center">
            <svg className="w-10 h-10 text-white drop-shadow-md" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          </div>
        </button>
      )}

      {/* Spoken Text Content */}
      <button
        onClick={() => onClick(result)}
        className="w-full text-left p-3.5 flex-1 flex flex-col justify-between gap-3 hover:bg-slate-100 dark:hover:bg-slate-800/35 transition"
      >
        <p className="text-sm text-slate-700 dark:text-slate-200 leading-relaxed font-sans font-normal line-clamp-4">
          {result.text}
        </p>
        <div className="flex items-center gap-1 text-[11px] text-slate-500 font-mono select-none">
          <svg className="w-3.5 h-3.5 text-slate-500" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>Click to play segment</span>
        </div>
      </button>
    </div>
  );
}
