import type { SearchResult } from "@/types";
import ResultCard from "./ResultCard";

interface Props {
  results: SearchResult[];
  total: number;
  executionTimeMs: number;
  onCardClick: (result: SearchResult) => void;
}

export default function ResultGrid({ results, total, executionTimeMs, onCardClick }: Props) {
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
      </div>

      {/* Grid — 2 cols on mobile, 4 on md, 6 on xl */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6 gap-3">
        {results.map((r, i) => (
          <ResultCard
            key={r.frame_id}
            result={r}
            rank={i + 1}
            onClick={onCardClick}
          />
        ))}
      </div>

      {results.length === 0 && (
        <p className="text-slate-500 text-center py-16">No results. Try a different query.</p>
      )}
    </div>
  );
}
