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
      <div className="flex items-center gap-4 text-sm text-slate-500 dark:text-slate-400">
        <span>
          <span className="text-slate-900 dark:text-white font-semibold">{total}</span> results
        </span>
        <span>·</span>
        <span title="Total time including translation and search">
          <span className="text-slate-900 dark:text-white font-semibold">{executionTimeMs}</span> ms
        </span>
      </div>

      {/* Grid — sized for a half-width results pane, not the full viewport */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
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
