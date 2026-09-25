import { useEffect, useRef, useState } from "react";
import {
  cancelVerify,
  fetchVerify,
  resetVerify,
  startVerify,
  type VerifyCandidate,
  type VerifyState,
  type VerifyStatus,
} from "@/lib/agentVerify";

export interface VerifyRequest {
  query: string;
  candidate: VerifyCandidate;
  token: number;
}

interface Props {
  request: VerifyRequest | null;
}

const STATUS: Record<VerifyState["status"], string> = {
  idle: "Idle", verifying: "Verifying", done: "Done", error: "Error", cancelled: "Cancelled",
};
const CHECK_COLOR: Record<VerifyStatus, string> = {
  MATCH: "text-teal-700 dark:text-teal-300",
  MISMATCH: "text-rose-700 dark:text-rose-300",
  UNKNOWN: "text-amber-700 dark:text-amber-300",
};

export default function VerifyAgentPanel({ request }: Props) {
  const [state, setState] = useState<VerifyState | null>(null);
  const [connectionError, setConnectionError] = useState("");
  const sequence = useRef(0);

  useEffect(() => {
    let active = true;
    async function refresh() {
      const current = ++sequence.current;
      try {
        const next = await fetchVerify();
        if (!active || current !== sequence.current) return;
        setState(next);
        setConnectionError("");
      } catch (error) {
        if (active && current === sequence.current) {
          setConnectionError(error instanceof Error ? error.message : "Verify Agent unavailable");
        }
      }
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), 1000);
    return () => { active = false; window.clearInterval(timer); sequence.current += 1; };
  }, []);

  useEffect(() => {
    if (!request) return;
    if (!request.query.trim()) {
      setConnectionError("Enter the official query before requesting verification.");
      return;
    }
    const current = ++sequence.current;
    startVerify(request.query, request.candidate)
      .then((next) => {
        if (current !== sequence.current) return;
        setState(next);
        setConnectionError("");
      })
      .catch((error) => {
        if (current === sequence.current) {
          setConnectionError(error instanceof Error ? error.message : "Verify Agent unavailable");
        }
      });
  }, [request]);

  async function act(action: () => Promise<VerifyState>) {
    const current = ++sequence.current;
    try {
      const next = await action();
      if (current !== sequence.current) return;
      setState(next);
      setConnectionError("");
    } catch (error) {
      if (current === sequence.current) {
        setConnectionError(error instanceof Error ? error.message : "Verify Agent unavailable");
      }
    }
  }

  const completed = Object.values(state?.results ?? {}).slice(-4).reverse();
  return (
    <section className="rounded border-2 border-stone-800 dark:border-stone-600 bg-cream-card dark:bg-stone-800/50 p-4 space-y-3" aria-label="Verify Agent">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div>
          <h2 className="font-retro font-bold text-sm uppercase tracking-wide">Verify Agent</h2>
          <p className="text-xs text-stone-500 dark:text-stone-400">Human selected candidates · operator makes the final decision</p>
        </div>
        <span className="rounded border border-stone-500 px-2 py-0.5 text-xs font-mono" role="status" aria-live="polite">
          {connectionError ? "Unavailable" : STATUS[state?.status ?? "idle"]}
        </span>
      </div>
      <p className="text-xs text-stone-500 dark:text-stone-400">Visual checks stay UNKNOWN until frame image transport is tested. Open a candidate to inspect its video.</p>
      {state?.query && <p className="text-xs">Query: {state.query}</p>}
      {state?.current_candidate && <p className="text-xs font-mono">Verifying {state.current_candidate.video_id} · {state.current_candidate.frame_id || state.current_candidate.chunk_id || "segment"}</p>}
      {(state?.queue.length ?? 0) > 0 && <p className="text-xs">Queued: {state?.queue.length}</p>}
      <div className="flex gap-2">
        {state?.status === "verifying" && <button type="button" onClick={() => void act(cancelVerify)} className="rounded border border-stone-600 px-2 py-1 text-xs">Cancel Verify</button>}
        {state && (state.query || Object.keys(state.results).length > 0) && <button type="button" onClick={() => void act(resetVerify)} className="rounded border border-stone-600 px-2 py-1 text-xs">Reset Verify</button>}
      </div>
      {connectionError && <p role="alert" className="text-xs text-rose-700 dark:text-rose-400">{connectionError}</p>}
      {state?.error && <p role="alert" className="text-xs text-rose-700 dark:text-rose-400">{state.error}</p>}
      {completed.map((result) => (
        <div key={result.candidate_id} className="rounded border border-stone-500 p-2 space-y-2 text-xs">
          <div className="flex justify-between gap-2 flex-wrap">
            <span className="font-mono">{result.candidate.video_id} · {result.candidate.frame_id || result.candidate.chunk_id || "segment"}</span>
            <strong>{result.status === "error" ? "Error" : result.overall}</strong>
          </div>
          {result.error && <p className="text-rose-700 dark:text-rose-400">{result.error}</p>}
          {(result.evidence?.errors.length ?? 0) > 0 && <p className="text-amber-700 dark:text-amber-300">Some VORTA evidence was unavailable; unresolved checks remain UNKNOWN.</p>}
          {result.checks?.map((check, index) => (
            <div key={`${result.candidate_id}-${index}`} className="flex gap-2 justify-between border-t border-stone-300 dark:border-stone-600 pt-1">
              <span>{check.requirement} <span className="text-stone-500">· {check.note}{check.evidence_quote ? ` · “${check.evidence_quote}”` : ""}</span></span>
              <strong className={`${CHECK_COLOR[check.status]} shrink-0`}>{check.status}</strong>
            </div>
          ))}
          {result.status === "done" && <p className="text-stone-500">Inspect the video and decide manually.</p>}
        </div>
      ))}
    </section>
  );
}
