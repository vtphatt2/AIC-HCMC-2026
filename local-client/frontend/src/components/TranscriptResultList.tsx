import type { TranscriptResult } from "@/types";
import TranscriptResultCard from "./TranscriptResultCard";

interface Props {
  results: TranscriptResult[];
  total: number;
  executionTimeMs: number;
  onCardClick: (result: TranscriptResult) => void;
}

export default function TranscriptResultList({ results, total, executionTimeMs, onCardClick }: Props) {
  return (
    <div className="space-y-4">
      {/* Stats bar */}
      <div className="flex items-center gap-4 text-sm text-slate-400">
        <span>
          <span className="text-white font-semibold">{total}</span> transcript matches
        </span>
        <span>·</span>
        <span title="Total time for transcript search">
          <span className="text-white font-semibold">{executionTimeMs}</span> ms
        </span>
      </div>

      {/* List — 1 col on mobile, 2 on md */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {results.map((r, i) => (
          <TranscriptResultCard
            key={`${r.video_id}_${r.start_time_ms}_${i}`}
            result={r}
            rank={i + 1}
            onClick={onCardClick}
          />
        ))}
      </div>

      {results.length === 0 && (
        <p className="text-slate-500 text-center py-16">
          No transcript matches. Try different keywords.
        </p>
      )}
    </div>
  );
}
