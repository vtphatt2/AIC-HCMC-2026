import { useEffect, useRef, useState } from "react";
import type { SearchResult, TranscriptChunkResult } from "@/types";
import {
  cancelAgentSearch,
  fetchAgentSearch,
  resetAgentSearch,
  startAgentSearch,
  type AgentSearchState,
} from "@/lib/agentSearch";
import { transcriptChunksToFrameResults } from "@/lib/transcriptSearch";
import ResultCard from "./ResultCard";
import TranscriptChunkCard from "./TranscriptChunkCard";
import VerifyAction from "./VerifyAction";

interface Props {
  manualQuery: string;
  onOpen: (result: SearchResult) => void;
  onVerify: (result: SearchResult | TranscriptChunkResult, query: string) => void;
  onQueryChange: (query: string) => void;
}

const STATUS: Record<AgentSearchState["status"], string> = {
  idle: "Idle",
  planning: "Planning",
  searching: "Searching",
  done: "Done",
  error: "Error",
  cancelled: "Cancelled",
};

export default function AgentSearchPanel({ manualQuery, onOpen, onVerify, onQueryChange }: Props) {
  const [query, setQuery] = useState("");
  const [state, setState] = useState<AgentSearchState | null>(null);
  const [connectionError, setConnectionError] = useState("");
  const editedRef = useRef(false);
  const requestSequence = useRef(0);

  useEffect(() => onQueryChange(state?.query ?? ""), [state?.query, onQueryChange]);

  useEffect(() => {
    let active = true;
    async function refresh() {
      const sequence = ++requestSequence.current;
      try {
        const next = await fetchAgentSearch();
        if (!active || sequence !== requestSequence.current) return;
        setState(next);
        setConnectionError("");
        if (!editedRef.current && next.query) setQuery(next.query);
      } catch (error) {
        if (!active || sequence !== requestSequence.current) return;
        setConnectionError(error instanceof Error ? error.message : "Search Agent unavailable");
      }
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), 1000);
    return () => { active = false; window.clearInterval(timer); requestSequence.current += 1; };
  }, []);

  async function apply(action: () => Promise<AgentSearchState>) {
    const sequence = ++requestSequence.current;
    try {
      const next = await action();
      if (sequence !== requestSequence.current) return;
      setState(next);
      setConnectionError("");
    } catch (error) {
      if (sequence === requestSequence.current) {
        setConnectionError(error instanceof Error ? error.message : "Search Agent unavailable");
      }
    }
  }

  const running = state?.status === "planning" || state?.status === "searching";
  const transcriptMode = state?.plan?.primary.mode === "transcript";
  const frameResults = transcriptMode ? [] : (state?.results ?? []) as SearchResult[];
  const transcriptResults = transcriptMode ? (state?.results ?? []) as TranscriptChunkResult[] : [];

  return (
    <section className="rounded border-2 border-stone-800 dark:border-stone-600 bg-cream-card dark:bg-stone-800/50 p-4 space-y-3" aria-label="Search Agent">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div>
          <h2 className="font-retro font-bold text-sm uppercase tracking-wide">Search Agent</h2>
          <p className="text-xs text-stone-500 dark:text-stone-400">Shared across operators · candidates for human review</p>
        </div>
        <span className="rounded border border-stone-500 px-2 py-0.5 text-xs font-mono" role="status" aria-live="polite">
          {connectionError ? "Unavailable" : STATUS[state?.status ?? "idle"]}
        </span>
      </div>

      <div className="flex gap-2 items-start flex-wrap">
        <textarea
          value={query}
          onChange={(event) => { editedRef.current = true; setQuery(event.target.value); }}
          placeholder="Paste the official query for Agent Search…"
          rows={2}
          maxLength={1000}
          className="min-w-56 flex-1 rounded border-2 border-stone-700 dark:border-stone-500 bg-cream dark:bg-stone-800 px-3 py-2 text-sm resize-y"
          aria-label="Official query for Search Agent"
        />
        <div className="flex gap-2 flex-wrap">
          <button
            type="button"
            onClick={() => { editedRef.current = true; setQuery(manualQuery); }}
            disabled={!manualQuery}
            className="rounded border-2 border-stone-700 dark:border-stone-500 px-2 py-1.5 text-xs disabled:opacity-40"
          >
            Use manual text
          </button>
          <button
            type="button"
            onClick={() => void apply(() => startAgentSearch(query.trim()))}
            disabled={!query.trim() || (running && query.trim() === state?.query)}
            className="rounded border-2 border-stone-900 bg-orange-700 px-3 py-1.5 text-xs font-bold text-white disabled:opacity-40"
          >
            Agent Search
          </button>
          {running && <button type="button" onClick={() => void apply(cancelAgentSearch)} className="rounded border-2 border-stone-700 px-2 py-1.5 text-xs">Cancel</button>}
          {state?.query_id && <button type="button" onClick={() => { editedRef.current = false; setQuery(""); void apply(resetAgentSearch); }} className="rounded border-2 border-stone-700 px-2 py-1.5 text-xs">Reset</button>}
        </div>
      </div>

      {connectionError && <p className="text-xs text-rose-700 dark:text-rose-400" role="alert">{connectionError}</p>}
      {state?.error && <p className="text-xs text-rose-700 dark:text-rose-400" role="alert">{state.error}</p>}
      {state?.query && <p className="text-xs text-stone-500 dark:text-stone-400">Current: {state.query}</p>}
      {state?.plan_summary && <p className="text-sm">{state.plan_summary}</p>}
      {state?.status === "done" && (
        <p className="text-xs text-stone-500 dark:text-stone-400">
          {state.total} candidates · {state.timing_ms?.total ?? 0} ms total · retrieval rank and scores are for browsing
        </p>
      )}

      {frameResults.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-2">
          {frameResults.map((result, index) => (
            <div key={`${result.frame_id}-${index}`} className="min-w-0">
              <ResultCard result={result} rank={index + 1} onClick={onOpen} compact />
              <VerifyAction onClick={() => onVerify(result, state?.query ?? "")} />
              {result.steps && result.steps.length > 1 && (
                <p className="mt-1 text-[11px] text-stone-500">{result.steps.length} ordered frames · open to inspect</p>
              )}
            </div>
          ))}
        </div>
      )}
      {transcriptResults.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          {transcriptResults.map((result, index) => (
            <div key={`${result.video_id}-${result.chunk_id}-${index}`}>
              <TranscriptChunkCard
              result={result}
              rank={index + 1}
              query={state?.query ?? ""}
              onClick={(chunk) => onOpen(transcriptChunksToFrameResults([chunk])[0])}
              onFrameClick={() => onOpen(transcriptChunksToFrameResults([result])[0])}
              />
              <VerifyAction onClick={() => onVerify(result, state?.query ?? "")} />
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
