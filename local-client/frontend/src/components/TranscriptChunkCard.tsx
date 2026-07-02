import { useMemo } from "react";
import type { TranscriptChunkResult } from "@/types";
import { apiUrl } from "@/lib/api";

interface FrameSample {
  frameNumber: number;
  timestampMs: number;
  imageUrl: string;
  isQueried: boolean;
}

interface Props {
  result: TranscriptChunkResult;
  rank: number;
  onClick: (result: TranscriptChunkResult) => void;
  onFrameClick?: (videoId: string, youtubeId: string, frameNumber: number, timestampMs: number) => void;
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

const FPS = 25;
const FILLER_INTERVAL = 10; // show filler frame every 10 frames

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

function computeSampleFrames(
  videoId: string,
  startMs: number,
  endMs: number,
  queriedFrameNumber: number,
): FrameSample[] {
  const startFrame = Math.floor((startMs / 1000) * FPS);
  const endFrame = Math.floor((endMs / 1000) * FPS);

  const frames: FrameSample[] = [];
  for (let fn = startFrame; fn <= endFrame; fn += FILLER_INTERVAL) {
    const isQueried = queriedFrameNumber > 0 && fn === queriedFrameNumber;
    frames.push({
      frameNumber: fn,
      timestampMs: Math.round((fn / FPS) * 1000),
      imageUrl: `/static/frames/${videoId}/${String(fn).padStart(6, "0")}.jpg`,
      isQueried,
    });
  }
  return frames;
}

export default function TranscriptChunkCard({ result, rank, onClick, onFrameClick }: Props) {
  const sampleFrames = useMemo(
    () =>
      computeSampleFrames(
        result.video_id,
        result.start_time_ms,
        result.end_time_ms,
        result.frame_number,
      ),
    [result.video_id, result.start_time_ms, result.end_time_ms, result.frame_number],
  );

  function handleImageError(e: React.SyntheticEvent<HTMLImageElement>) {
    (e.target as HTMLImageElement).style.display = "none";
  }

  function handleFrameClick(fn: number, ts: number, e: React.MouseEvent) {
    e.stopPropagation();
    onFrameClick?.(result.video_id, result.youtube_id, fn, ts);
  }

  return (
    <div className="group bg-slate-800/50 border border-slate-700 rounded-xl overflow-hidden hover:border-emerald-500 transition-all">
      {/* Header bar */}
      <div className="px-3 py-2 bg-slate-800/80 border-b border-slate-700/60 flex items-center gap-2 flex-wrap">
        <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-black/40 text-white font-mono">
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
        <span className="text-xs text-slate-300 font-mono bg-slate-700/60 px-2 py-0.5 rounded">
          {formatTimestamp(result.start_time_ms)} – {formatTimestamp(result.end_time_ms)}
        </span>
        <span className={`text-xs font-semibold ${confidenceColor(result.score)}`}>
          {(result.score * 100).toFixed(1)}%
        </span>
      </div>

      {/* Horizontal frame strip */}
      {sampleFrames.length > 0 && (
        <div className="flex overflow-x-auto gap-2 p-2 scrollbar-thin">
          {sampleFrames.map((sf) => (
            <button
              key={sf.frameNumber}
              onClick={(e) => handleFrameClick(sf.frameNumber, sf.timestampMs, e)}
              className={`relative shrink-0 w-36 rounded-lg overflow-hidden border transition-all hover:scale-[1.02] ${
                sf.isQueried
                  ? "ring-2 ring-yellow-400 ring-offset-1 ring-offset-slate-900"
                  : "opacity-60 ring-1 ring-slate-700 hover:opacity-100"
              }`}
              title={`Frame ${sf.frameNumber} — ${formatTimestamp(sf.timestampMs)}`}
            >
              <div className="aspect-video bg-slate-700">
                <img
                  src={apiUrl(sf.imageUrl)}
                  alt={`Frame ${sf.frameNumber}`}
                  className="w-full h-full object-cover"
                  loading="lazy"
                  decoding="async"
                  onError={handleImageError}
                />
              </div>
              {/* Timestamp badge */}
              <span className="absolute bottom-0.5 left-0.5 bg-black/80 text-white text-[10px] px-1 py-px rounded font-mono leading-tight">
                {formatTimestampCompact(sf.timestampMs)}
              </span>
              {/* Play overlay on hover */}
              <div className="absolute inset-0 bg-black/30 opacity-0 hover:opacity-100 transition flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
              </div>
            </button>
          ))}
        </div>
      )}

      {/* Text content */}
      <button
        onClick={() => onClick(result)}
        className="w-full text-left px-3 pb-3 pt-2 space-y-1"
      >
        <p className="text-xs text-slate-400 truncate font-mono hover:text-slate-300 transition">
          {result.video_id}
        </p>
        <p className="text-sm text-slate-200 line-clamp-3 leading-relaxed">
          {result.text}
        </p>
      </button>
    </div>
  );
}
