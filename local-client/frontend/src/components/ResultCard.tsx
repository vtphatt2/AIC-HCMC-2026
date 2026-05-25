import type { SearchResult } from "@/types";

interface Props {
  result: SearchResult;
  rank: number;
  onClick: (result: SearchResult) => void;
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

export default function ResultCard({ result, rank, onClick }: Props) {
  return (
    <button
      onClick={() => onClick(result)}
      className="group bg-slate-800 border border-slate-700 rounded-xl overflow-hidden hover:border-blue-500 hover:shadow-lg hover:shadow-blue-900/30 transition-all text-left w-full"
    >
      {/* Frame image */}
      <div className="relative aspect-video bg-slate-700 overflow-hidden">
        <img
          src={result.frame_image_url}
          alt={`Frame ${result.frame_number} of ${result.video_id}`}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
          loading="lazy"
        />
        {/* Rank badge */}
        <span className="absolute top-1 left-1 bg-black/70 text-white text-xs px-1.5 py-0.5 rounded">
          #{rank}
        </span>
        {/* Play overlay on hover */}
        <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition flex items-center justify-center">
          <svg className="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 24 24">
            <path d="M8 5v14l11-7z" />
          </svg>
        </div>
      </div>

      {/* Metadata */}
      <div className="p-2 space-y-0.5">
        <p className="text-xs text-slate-400 truncate font-mono">{result.video_id}</p>
        <div className="flex items-center justify-between">
          <span className="text-sm text-white font-medium">{formatTimestamp(result.timestamp_ms)}</span>
          <span className={`text-xs font-semibold ${confidenceColor(result.confidence)}`}>
            {(result.confidence * 100).toFixed(1)}%
          </span>
        </div>
        <p className="text-xs text-slate-500">frame {result.frame_number}</p>
      </div>
    </button>
  );
}
