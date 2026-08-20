import Head from "next/head";
import { useCallback, useEffect, useState } from "react";

import type { SearchResult, SubmissionEntry, SubmissionSessionSummary, SubmissionState } from "@/types";
import {
  buildSubmissionCsv,
  downloadCsv,
  downloadSubmissionZip,
  editSubmissionEntryFrame,
  editSubmissionEntryGroup,
  fetchSubmission,
  fetchSubmissionSessions,
  orderedEntries,
  removeSubmissionEntry,
  reorderKeys,
  reorderSubmissionRows,
  submissionEntryToSearchResult,
  trakeCandidateSizeMismatch,
  trakeCandidateVideoMismatch,
  trakeGroupKey,
  trakeSortedGroups,
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
  const [activeContext, setActiveContext] = useState<{ session: string; groupIndex: number } | null>(null);
  const [dragOver, setDragOver] = useState<{ session: string; group: number } | null>(null);
  const [dragOverEntry, setDragOverEntry] = useState<string | null>(null);

  function openEntry(session: string, entry: SubmissionEntry) {
    setActiveResult(submissionEntryToSearchResult(entry));
    setActiveContext({ session, groupIndex: entry.groupIndex });
  }

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

  async function handleMoveGroup(session: string, id: string, groupIndex: number) {
    if (groupIndex < 0) return;
    const updated = await editSubmissionEntryGroup(session, id, groupIndex);
    setStates((prev) => ({ ...prev, [session]: updated }));
  }

  async function handleReorderRows(session: string, rowOrder: string[]) {
    const updated = await reorderSubmissionRows(session, rowOrder);
    setStates((prev) => ({ ...prev, [session]: updated }));
  }

  // A candidate box is a drop target for two different drags — a frame card
  // (reassigns its candidate) or another candidate's header (reorders the
  // ranking) — told apart by dataTransfer type.
  function handleCandidateDrop(e: React.DragEvent, session: string, targetGroupIndex: number) {
    e.preventDefault();
    setDragOver(null);
    const state = states[session];
    if (!state) return;
    if (e.dataTransfer.types.includes("application/x-candidate-key")) {
      const draggedKey = e.dataTransfer.getData("application/x-candidate-key");
      const targetKey = trakeGroupKey(targetGroupIndex);
      if (draggedKey === targetKey) return;
      const keys = trakeSortedGroups(state.entries, state.rowOrder).map((es) => trakeGroupKey(es[0].groupIndex));
      handleReorderRows(session, reorderKeys(keys, draggedKey, targetKey));
      return;
    }
    const frameId = e.dataTransfer.getData("application/x-frame-id");
    if (frameId) handleMoveGroup(session, frameId, targetGroupIndex);
  }

  function handleEntryDrop(e: React.DragEvent, session: string, targetId: string) {
    e.preventDefault();
    setDragOverEntry(null);
    const state = states[session];
    if (!state) return;
    const draggedId = e.dataTransfer.getData("application/x-entry-id");
    if (!draggedId) return;
    const ids = orderedEntries(state).map((entry) => entry.id);
    handleReorderRows(session, reorderKeys(ids, draggedId, targetId));
  }

  function handleDownload(state: SubmissionState) {
    const { filename, content, rows } = buildSubmissionCsv(state);
    if (rows === 0) return window.alert("No entries to export yet.");
    downloadCsv(filename, content);
  }

  function handleDownloadZip() {
    const ok = downloadSubmissionZip(Object.values(states));
    if (!ok) window.alert("No sessions with entries to bundle yet.");
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
            <div className="flex items-center gap-3 shrink-0">
              <button className={BTN} onClick={handleDownloadZip}>⬇ Download submission.zip</button>
              <a href="/" className="font-retro text-sm text-orange-700 dark:text-orange-400 hover:underline">Back to search</a>
            </div>
          </header>

          {sessions.length === 0 && (
            <p className="text-sm text-stone-500 italic">No submission sessions yet.</p>
          )}

          {sessions.map((summary) => {
            const state = states[summary.session];
            if (!state) return null;
            const groupCount = new Set(state.entries.map((e) => e.groupIndex)).size;
            const sizeMismatch = trakeCandidateSizeMismatch(state);
            const videoMismatch = trakeCandidateVideoMismatch(state);
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

                {state.queryType === "trake" && (
                  <p className="text-sm font-bold text-orange-700 dark:text-orange-400">
                    ▸ Adding to candidate {state.nextGroupIndex + 1}
                  </p>
                )}

                {sizeMismatch && (
                  <p className="text-xs text-red-600">
                    ⚠ Candidates have different frame counts ({sizeMismatch.join(", ")}) — organizer scoring
                    requires every candidate to match the query's event count exactly.
                  </p>
                )}

                {videoMismatch && (
                  <p className="text-xs text-red-600">
                    ⚠ Candidate{videoMismatch.length === 1 ? "" : "s"} {videoMismatch.join(", ")} mix frames from
                    different videos — every frame in a TRAKE candidate must come from the same video.
                  </p>
                )}

                {state.entries.length === 0 && (
                  <p className="text-sm text-stone-500 italic">No frames added yet.</p>
                )}

                {state.entries.length > 0 && state.queryType === "trake" && (
                  <div className="space-y-2">
                    {trakeSortedGroups(state.entries, state.rowOrder).map((groupEntries) => {
                      const gi = groupEntries[0].groupIndex;
                      const isOver = dragOver?.session === summary.session && dragOver.group === gi;
                      return (
                        <div
                          key={gi}
                          onDragOver={(e) => { e.preventDefault(); setDragOver({ session: summary.session, group: gi }); }}
                          onDragLeave={() => setDragOver((d) => (d?.session === summary.session && d.group === gi ? null : d))}
                          onDrop={(e) => handleCandidateDrop(e, summary.session, gi)}
                          className={`border-2 rounded p-2 transition-colors ${
                            isOver ? "border-orange-600 bg-orange-50 dark:bg-orange-950/30" : "border-stone-400 dark:border-stone-600"
                          }`}
                        >
                          <p
                            draggable
                            onDragStart={(e) => e.dataTransfer.setData("application/x-candidate-key", trakeGroupKey(gi))}
                            className="text-xs font-bold uppercase text-stone-500 mb-1.5 cursor-grab active:cursor-grabbing w-fit"
                            title="Drag to reorder this candidate's rank"
                          >
                            ⠿ Candidate {gi + 1} · {groupEntries.length} frame{groupEntries.length === 1 ? "" : "s"}
                          </p>
                          <div className="flex flex-wrap gap-2">
                            {groupEntries.map((entry) => (
                              <div
                                key={entry.id}
                                draggable
                                onDragStart={(e) => e.dataTransfer.setData("application/x-frame-id", entry.id)}
                                className="flex flex-col items-center gap-1 w-28 border border-stone-300 dark:border-stone-700 rounded p-2 cursor-grab active:cursor-grabbing bg-cream-card dark:bg-stone-900"
                              >
                                {entry.imageUrl && (
                                  <button type="button" onClick={() => openEntry(summary.session, entry)}>
                                    <img
                                      src={entry.imageUrl}
                                      alt=""
                                      className="w-24 h-14 object-cover rounded hover:ring-2 hover:ring-orange-600 transition pointer-events-none"
                                    />
                                  </button>
                                )}
                                <span className="text-xs font-mono text-stone-500 truncate max-w-full">{entry.videoId}</span>
                                <input
                                  key={entry.frame}
                                  type="number"
                                  min={0}
                                  defaultValue={entry.frame}
                                  className={`${FRAME_INPUT} w-full text-center px-1`}
                                  onBlur={(e) => handleEditFrame(summary.session, entry.id, e.target.value)}
                                  onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
                                />
                                <button
                                  onClick={() => handleRemove(summary.session, entry.id)}
                                  className="text-stone-400 hover:text-red-600 transition text-xs"
                                  aria-label="Remove"
                                >
                                  ✕ remove
                                </button>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                    <p className="text-xs text-stone-500 italic">
                      Drag a frame onto a different candidate box to move it; drag a candidate's ⠿ title to rank it.
                    </p>
                  </div>
                )}

                {state.entries.length > 0 && state.queryType !== "trake" && (
                  <div className="space-y-1">
                    {orderedEntries(state).map((entry) => (
                      <div
                        key={entry.id}
                        draggable
                        onDragStart={(e) => e.dataTransfer.setData("application/x-entry-id", entry.id)}
                        onDragOver={(e) => { e.preventDefault(); setDragOverEntry(entry.id); }}
                        onDragLeave={() => setDragOverEntry((id) => (id === entry.id ? null : id))}
                        onDrop={(e) => handleEntryDrop(e, summary.session, entry.id)}
                        className={`flex items-center gap-2 border rounded px-2 py-1 cursor-grab active:cursor-grabbing transition-colors ${
                          dragOverEntry === entry.id
                            ? "border-orange-600 bg-orange-50 dark:bg-orange-950/30"
                            : "border-stone-300 dark:border-stone-700"
                        }`}
                      >
                        <span className="text-stone-400 select-none">⠿</span>
                        {entry.imageUrl && (
                          <button type="button" onClick={() => openEntry(summary.session, entry)}>
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
                        <button
                          onClick={() => handleRemove(summary.session, entry.id)}
                          className="ml-auto text-stone-400 hover:text-red-600 transition"
                          aria-label="Remove"
                        >
                          ✕
                        </button>
                      </div>
                    ))}
                    <p className="text-xs text-stone-500 italic">Drag a row to rank it — export order matches this list.</p>
                  </div>
                )}
              </section>
            );
          })}
        </div>
      </main>

      {activeResult && (
        <VideoModal
          result={activeResult}
          onClose={() => { setActiveResult(null); setActiveContext(null); }}
          showTranscript={false}
          onToggleTranscript={() => {}}
          onOpenSubmissionPanel={() => window.alert("Pick a working session first, from the search page.")}
          overrideSession={activeContext?.session}
          overrideGroupIndex={activeContext?.groupIndex}
        />
      )}
    </>
  );
}
