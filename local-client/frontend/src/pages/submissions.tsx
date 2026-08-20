import Head from "next/head";
import { useCallback, useEffect, useState } from "react";

import type { SearchResult, SubmissionSessionSummary, SubmissionState } from "@/types";
import {
  buildSubmissionCsv,
  downloadCsv,
  editSubmissionEntryFrame,
  fetchSubmission,
  fetchSubmissionSessions,
  removeSubmissionEntry,
  submissionEntryToSearchResult,
} from "@/lib/submission";
import VideoModal from "@/components/VideoModal";

const BTN =
  "font-retro text-xs uppercase tracking-wide border-2 border-stone-800 dark:border-stone-500 rounded px-2.5 py-1.5 hover:bg-orange-600 hover:text-white hover:border-orange-600 transition";
const FRAME_INPUT =
  "w-20 bg-cream-card dark:bg-stone-800 border-2 border-stone-800 dark:border-stone-500 rounded px-1.5 py-0.5 text-sm font-mono text-stone-900 dark:text-stone-50 focus:outline-none focus:ring-2 focus:ring-orange-600";

export default function SubmissionsDashboard() {
  const [sessions, setSessions] = useState<SubmissionSessionSummary[]>([]);
  const [states, setStates] = useState<Record<string, SubmissionState>>({});
  const [activeResult, setActiveResult] = useState<SearchResult | null>(null);

  useEffect(() => {
    document.documentElement.classList.toggle(
      "dark",
      window.localStorage.getItem("aic2026-theme") !== "light",
    );
  }, []);

  const refreshAll = useCallback(async () => {
    const list = await fetchSubmissionSessions().catch(() => []);
    setSessions(list);
    const fetched = await Promise.all(list.map((s) => fetchSubmission(s.session).catch(() => null)));
    setStates((prev) => {
      const next: Record<string, SubmissionState> = {};
      list.forEach((s, i) => {
        next[s.session] = fetched[i] || prev[s.session];
      });
      return next;
    });
  }, []);

  useEffect(() => {
    refreshAll();
    const interval = setInterval(refreshAll, 5000);
    return () => clearInterval(interval);
  }, [refreshAll]);

  async function handleEditFrame(session: string, id: string, raw: string) {
    const frame = Number.parseInt(raw, 10);
    if (!Number.isFinite(frame) || frame < 0) return;
    const updated = await editSubmissionEntryFrame(session, id, frame);
    setStates((prev) => ({ ...prev, [session]: updated }));
  }

  async function handleRemove(session: string, id: string) {
    const updated = await removeSubmissionEntry(session, id);
    setStates((prev) => ({ ...prev, [session]: updated }));
  }

  function handleDownload(state: SubmissionState) {
    const { filename, content, rows } = buildSubmissionCsv(state);
    if (rows === 0) return window.alert("No entries to export yet.");
    downloadCsv(filename, content);
  }

  return (
    <>
      <Head><title>Submission Dashboard</title></Head>
      <main className="min-h-screen bg-cream dark:bg-stone-900 text-stone-900 dark:text-stone-50 p-5 md:p-8">
        <div className="max-w-4xl mx-auto space-y-6">
          <header className="flex items-start justify-between gap-4">
            <div>
              <h1 className="font-retro text-2xl font-bold uppercase">Submission Dashboard</h1>
              <p className="text-sm text-stone-500 dark:text-stone-400">
                Every session, live — click a frame to review it, edit the frame number inline.
              </p>
            </div>
            <a href="/" className="font-retro text-sm text-orange-700 dark:text-orange-400 hover:underline">Back to search</a>
          </header>

          {sessions.length === 0 && (
            <p className="text-sm text-stone-500 italic">No submission sessions yet.</p>
          )}

          {sessions.map((summary) => {
            const state = states[summary.session];
            if (!state) return null;
            const groupCount = new Set(state.entries.map((e) => e.groupIndex)).size;
            return (
              <section
                key={summary.session}
                className="border-2 border-stone-800 dark:border-stone-500 rounded p-4 bg-cream-card dark:bg-stone-800 space-y-3"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h2 className="font-retro text-lg font-bold">
                    {state.session}{" "}
                    <span className="text-xs font-normal text-stone-500 dark:text-stone-400">
                      {state.queryType.toUpperCase()} · query #{state.queryNumber} · {state.entries.length} row
                      {state.entries.length === 1 ? "" : "s"}
                      {state.queryType === "trake" && ` · ${groupCount} candidate${groupCount === 1 ? "" : "s"}`}
                    </span>
                  </h2>
                  <button className={BTN} onClick={() => handleDownload(state)}>⬇ Download CSV</button>
                </div>

                {state.queryType === "qa" && state.answer && (
                  <p className="text-sm text-stone-600 dark:text-stone-300">Answer: {state.answer}</p>
                )}

                <div className="space-y-1">
                  {state.entries.length === 0 && (
                    <p className="text-sm text-stone-500 italic">No frames added yet.</p>
                  )}
                  {state.entries.map((entry) => (
                    <div
                      key={entry.id}
                      className="flex items-center gap-2 border border-stone-300 dark:border-stone-700 rounded px-2 py-1"
                    >
                      {entry.imageUrl && (
                        <button type="button" onClick={() => setActiveResult(submissionEntryToSearchResult(entry))}>
                          <img src={entry.imageUrl} alt="" className="w-16 h-9 object-cover rounded shrink-0 hover:ring-2 hover:ring-orange-600 transition" />
                        </button>
                      )}
                      <span className="text-sm font-mono text-stone-800 dark:text-stone-200">{entry.videoId}</span>
                      <span className="text-xs text-stone-500">frame</span>
                      <input
                        key={entry.frame}
                        type="number"
                        min={0}
                        defaultValue={entry.frame}
                        className={FRAME_INPUT}
                        onBlur={(e) => handleEditFrame(summary.session, entry.id, e.target.value)}
                        onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
                      />
                      {state.queryType === "trake" && (
                        <span className="text-xs text-stone-500">candidate {entry.groupIndex + 1}</span>
                      )}
                      <button
                        onClick={() => handleRemove(summary.session, entry.id)}
                        className="ml-auto text-stone-400 hover:text-red-600 transition"
                        aria-label="Remove"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              </section>
            );
          })}
        </div>
      </main>

      {activeResult && (
        <VideoModal
          result={activeResult}
          onClose={() => setActiveResult(null)}
          showTranscript={false}
          onToggleTranscript={() => {}}
          onOpenSubmissionPanel={() => window.alert("Pick a working session first, from the search page.")}
        />
      )}
    </>
  );
}
