import type { TranscriptChunkResult } from "@/types";

interface Props {
  result: TranscriptChunkResult;
  rank: number;
  onClick: (result: TranscriptChunkResult) => void;
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

function confidenceColor(score: number): string {
  if (score >= 0.8) return "text-emerald-400";
  if (score >= 0.6) return "text-yellow-400";
  return "text-red-400";
}

function topicColor(topic: string): string {
  return TOPIC_COLORS[topic] ?? "bg-slate-700 text-slate-300";
}

export default function TranscriptChunkCard({ result, rank, onClick }: Props) {
  return (
    <button
      onClick={() => onClick(result)}
      className="group bg-slate-800 border border-slate-700 rounded-xl overflow-hidden hover:border-emerald-500 hover:shadow-lg hover:shadow-emerald-900/30 transition-all text-left w-full"
    >
      {/* Rank bar + topic badge */}
      <div className="flex items-center gap-2 px-3 pt-3 pb-0">
        <span className="bg-black/60 text-white text-xs px-1.5 py-0.5 rounded font-mono">
          #{rank}
        </span>
        <span
          className={`text-[11px] font-semibold px-2 py-0.5 rounded-full ${topicColor(result.topic)}`}
        >
          {result.topic}
        </span>
        <div className="flex-1" />
        <span className={`text-xs font-semibold ${confidenceColor(result.score)}`}>
          {(result.score * 100).toFixed(1)}%
        </span>
      </div>

      {/* Text content */}
      <div className="p-3 pt-1.5 space-y-2">
        {/* Video ID + clickable timestamp */}
        <div className="flex items-center justify-between gap-2">
          <p className="text-xs text-slate-400 truncate font-mono">{result.video_id}</p>
          <span className="text-xs text-slate-300 font-mono bg-slate-700/70 px-2 py-0.5 rounded group-hover:bg-emerald-700/50 transition-colors shrink-0">
            {formatTimestamp(result.start_time_ms)} – {formatTimestamp(result.end_time_ms)}
          </span>
        </div>

        {/* Text preview */}
        <p className="text-sm text-slate-200 line-clamp-3 leading-relaxed">
          {result.text}
        </p>
      </div>
    </button>
  );
}
