import Head from "next/head";
import { useCallback, useEffect, useRef, useState } from "react";

import type {
  SearchResult,
  SubmissionQueryType,
  SubmissionSessionSummary,
  SubmissionState,
} from "@/types";
import {
  addSubmissionRow,
  addSubmissionRowFrame,
  buildSubmissionCsv,
  createSubmissionSession,
  deleteSubmissionSession,
  downloadCsv,
  downloadSubmissionZip,
  editRowFrame,
  editRowVideoId,
  fetchSubmission,
  fetchSubmissionSessions,
  getVideoInfo,
  moveItem,
  newTrakeCandidate,
  removeRow,
  removeRowFrame,
  renameSubmissionSession,
  reorderSubmissionRows,
  replaceSubmissionCsv,
  rowFrameToSearchResult,
  rowThumbUrl,
  setQueryType,
  setRowAnswer,
  trakeSizeMismatch,
  useVideoInfo,
} from "@/lib/submission";
import VideoModal from "@/components/VideoModal";

const QUERY_TYPES: SubmissionQueryType[] = ["kis", "qa", "trake"];

const BTN =
  "font-retro text-xs uppercase tracking-wide border-2 border-stone-800 dark:border-stone-500 rounded px-2.5 py-1.5 hover:bg-orange-600 hover:text-white hover:border-orange-600 transition";
const DANGER_BTN =
  "font-retro text-xs uppercase tracking-wide border-2 border-stone-800 dark:border-stone-500 rounded px-2.5 py-1.5 hover:bg-red-600 hover:text-white hover:border-red-600 transition";
const SMALL_INPUT =
  "bg-cream-card dark:bg-stone-800 border-2 border-stone-800 dark:border-stone-500 rounded px-1.5 py-0.5 text-sm font-mono text-stone-900 dark:text-stone-50 focus:outline-none focus:ring-2 focus:ring-orange-600";
const ANSWER_INPUT =
  "w-full bg-cream-card dark:bg-stone-800 border-2 border-stone-800 dark:border-stone-500 rounded px-2.5 py-1.5 text-sm text-stone-900 dark:text-stone-50 focus:outline-none focus:ring-2 focus:ring-orange-600";

// Local draft while typing, saved on blur — see SubmissionPanel.tsx for why
// (saving on every keystroke round-trips to the server before React
// re-renders the controlled value, which breaks Vietnamese IME composition).
function RowAnswerField({ answer, onSave }: { answer: string; onSave: (value: string) => void }) {
  const [draft, setDraft] = useState(answer);
  const focused = useRef(false);

  useEffect(() => {
    if (!focused.current) setDraft(answer);
  }, [answer]);

  return (
    <div className="space-y-0.5">
      <textarea
        className={ANSWER_INPUT}
        placeholder="Answer for this candidate"
        value={draft}
        maxLength={100}
        rows={2}
        onFocus={() => { focused.current = true; }}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={(e) => {
          focused.current = false;
          if (e.target.value !== answer) onSave(e.target.value);
        }}
      />
      <p className="text-xs text-stone-500 text-right">{draft.length}/100</p>
    </div>
  );
}

// Raw CSV textarea — a second way to edit a session besides the frame grid.
// The on-disk file already IS this exact text (see buildSubmissionCsv), so
// this is a direct view of the stored data, not a derived one. Doesn't
// overwrite the draft while `dirty`, so a slow save or a background poll
// tick never clobbers text the user hasn't saved yet; a failed save keeps
// what they typed and shows why instead of discarding it.
function RawCsvEditor({
  session,
  content,
  onSaved,
}: {
  session: string;
  content: string;
  onSaved: (state: SubmissionState) => void;
}) {
  const [draft, setDraft] = useState(content);
  const [dirty, setDirty] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    if (!dirty) setDraft(content);
  }, [content, dirty]);

  async function handleSave() {
    try {
      onSaved(await replaceSubmissionCsv(session, draft));
      setDirty(false);
      setSaveError(null);
    } catch (err: any) {
      setSaveError(err.message || "Failed to save");
    }
  }

  return (
    <div className="space-y-1.5">
      <textarea
        className={`${ANSWER_INPUT} font-mono text-xs`}
        rows={Math.min(20, Math.max(4, draft.split("\n").length + 1))}
        value={draft}
        spellCheck={false}
        onChange={(e) => { setDraft(e.target.value); setDirty(true); }}
      />
      <div className="flex items-center gap-2">
        <button className={BTN} onClick={handleSave} disabled={!dirty}>💾 Save CSV</button>
        {dirty && !saveError && <span className="text-xs text-orange-600">Unsaved changes</span>}
        {saveError && <span className="text-xs text-red-600">{saveError}</span>}
      </div>
    </div>
  );
}

export default function SubmissionsDashboard() {
  const [sessions, setSessions] = useState<SubmissionSessionSummary[]>([]);
  const [states, setStates] = useState<Record<string, SubmissionState>>({});
  const [activeResult, setActiveResult] = useState<SearchResult | null>(null);
  const [activeContext, setActiveContext] = useState<{ session: string; rowIndex: number } | null>(null);
  const [showTranscript, setShowTranscript] = useState(true);
  const [sortByName, setSortByName] = useState(false);
  const [filterType, setFilterType] = useState<SubmissionQueryType | "all">("all");
  const [editMode, setEditMode] = useState<Record<string, "grid" | "raw">>({});
  const [newRowDraft, setNewRowDraft] = useState<Record<string, { videoId: string; frames: string }>>({});
  const [dragOverRow, setDragOverRow] = useState<{ session: string; row: number } | null>(null);

  const videoInfo = useVideoInfo(Object.values(states).flatMap((s) => s.rows.map((r) => r.videoId)));

  // Awaits getVideoInfo directly rather than reading the videoInfo state —
  // that state is populated by useVideoInfo's background prefetch below, and
  // a click landing before that fetch resolves would otherwise snapshot a
  // wrong/missing fps and youtubeId into the SearchResult VideoModal opens
  // with (a static object — it never re-reads videoInfo once set), forcing
  // the zip-video fallback and a wrong frame counter even for a video with a
  // real YouTube embed. getVideoInfo's own cache makes this instant once
  // useVideoInfo has already warmed it, which is the common case.
  async function openEntry(session: string, rowIndex: number, videoId: string, frame: number) {
    const info = await getVideoInfo(videoId);
    setActiveResult(rowFrameToSearchResult(videoId, frame, info.fps, info.youtubeId));
    setActiveContext({ session, rowIndex });
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

  function updateState(session: string, updated: SubmissionState) {
    setStates((prev) => ({ ...prev, [session]: updated }));
  }

  async function handleEditVideoId(session: string, rowIndex: number, videoId: string) {
    if (!videoId.trim()) return;
    try {
      updateState(session, await editRowVideoId(session, rowIndex, videoId.trim()));
    } catch (err: any) {
      window.alert(err.message || "Invalid video id");
    }
  }

  async function handleEditFrame(session: string, rowIndex: number, frameIndex: number, raw: string) {
    const frame = Number.parseInt(raw, 10);
    if (!Number.isFinite(frame) || frame < 0) return;
    updateState(session, await editRowFrame(session, rowIndex, frameIndex, frame));
  }

  async function handleRemoveFrame(session: string, rowIndex: number, frameIndex: number) {
    updateState(session, await removeRowFrame(session, rowIndex, frameIndex));
  }

  async function handleRemoveRow(session: string, rowIndex: number) {
    updateState(session, await removeRow(session, rowIndex));
  }

  async function handleAddFrameToRow(session: string, rowIndex: number, videoId: string) {
    const raw = window.prompt("Add frame number to this candidate:");
    if (!raw) return;
    const frame = Number.parseInt(raw, 10);
    if (!Number.isFinite(frame) || frame < 0) return;
    try {
      updateState(session, await addSubmissionRowFrame(session, videoId, frame, rowIndex));
    } catch (err: any) {
      window.alert(err.message || "Failed to add frame");
    }
  }

  async function handleAddRow(session: string, queryType: SubmissionQueryType) {
    const draft = newRowDraft[session] || { videoId: "", frames: "" };
    const videoId = draft.videoId.trim();
    if (!videoId) return;
    const frames = draft.frames
      .split(",")
      .map((s) => Number.parseInt(s.trim(), 10))
      .filter((n) => Number.isFinite(n) && n >= 0);
    if (frames.length === 0) return;
    if (queryType !== "trake" && frames.length > 1) {
      window.alert("kis/qa rows take exactly one frame.");
      return;
    }
    try {
      updateState(session, await addSubmissionRow(session, videoId, frames));
      setNewRowDraft((prev) => ({ ...prev, [session]: { videoId: "", frames: "" } }));
    } catch (err: any) {
      window.alert(err.message || "Failed to add row");
    }
  }

  async function handleCreateSession() {
    // The session name IS the exported filename (session.csv) — name it
    // exactly what BTC's query file is called, e.g. "query-p1-11-kis" for
    // their query-p1-11-kis.txt, so the export always matches without
    // needing a separate number field.
    const name = window.prompt("Session name — must exactly match BTC's query filename (e.g. query-p1-11-kis):")?.trim();
    if (!name) return;

    const typeRaw = window.prompt("Query type — kis, qa, or trake:", "kis")?.trim().toLowerCase();
    if (!typeRaw) return;
    if (!QUERY_TYPES.includes(typeRaw as SubmissionQueryType)) {
      window.alert(`Invalid query type "${typeRaw}" — must be kis, qa, or trake.`);
      return;
    }
    const queryType = typeRaw as SubmissionQueryType;

    try {
      await createSubmissionSession(name, queryType);
      refreshAll();
    } catch (err: any) {
      if (err.status === 409) window.alert("That session name is already taken — pick another.");
      else window.alert(err.message || "Failed to create session");
    }
  }

  async function handleSetQueryType(session: string, queryType: SubmissionQueryType) {
    updateState(session, await setQueryType(session, queryType));
  }

  async function handleSetAnswer(session: string, rowIndex: number, answer: string) {
    updateState(session, await setRowAnswer(session, rowIndex, answer));
  }

  async function handleNewCandidate(session: string) {
    updateState(session, await newTrakeCandidate(session));
  }

  async function handleRenameSession(session: string) {
    const newSession = window.prompt("New session name (also becomes the exported filename):", session)?.trim();
    if (!newSession || newSession === session) return;
    try {
      await renameSubmissionSession(session, newSession);
      refreshAll();
    } catch (err: any) {
      if (err.status === 409) window.alert("A session with that name already exists — pick another.");
      else window.alert(err.message || "Failed to rename session");
    }
  }

  async function handleDeleteSession(session: string) {
    if (!window.confirm(`Delete session "${session}"? This cannot be undone.`)) return;
    await deleteSubmissionSession(session);
    setSessions((prev) => prev.filter((s) => s.session !== session));
    setStates((prev) => {
      const next = { ...prev };
      delete next[session];
      return next;
    });
  }

  function handleRowDrop(e: React.DragEvent, session: string, targetIndex: number) {
    e.preventDefault();
    setDragOverRow(null);
    const state = states[session];
    if (!state) return;
    const from = Number(e.dataTransfer.getData("application/x-row-index"));
    if (!Number.isFinite(from) || from === targetIndex) return;
    const order = state.rows.map((_, i) => i);
    reorderSubmissionRows(session, moveItem(order, from, targetIndex)).then((updated) => updateState(session, updated));
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

  // Array.prototype.sort is stable, so returning 0 when sortByName is off
  // just preserves the fetch order (createdAt, from the API) unchanged.
  const visibleSessions = sessions
    .filter((s) => filterType === "all" || s.queryType === filterType)
    .sort((a, b) => (sortByName ? a.session.localeCompare(b.session) : 0));

  return (
    <>
      <Head><title>Submission Dashboard</title></Head>
      <main className="min-h-screen bg-cream dark:bg-stone-900 text-stone-900 dark:text-stone-50 p-5 md:p-8">
        <div className="max-w-4xl mx-auto space-y-6">
          <header className="flex items-start justify-between gap-4">
            <div>
              <h1 className="font-retro text-2xl font-bold uppercase">Submission Dashboard</h1>
              <p className="text-sm text-stone-500 dark:text-stone-400">
                Every session, live — click a frame to review it, edit inline, or switch to raw CSV.
              </p>
            </div>
            <div className="flex items-center gap-3 shrink-0">
              <button className={BTN} onClick={handleCreateSession}>+ New session</button>
              <button className={BTN} onClick={handleDownloadZip}>⬇ Download submission.zip</button>
              <a href="/" className="font-retro text-sm text-orange-700 dark:text-orange-400 hover:underline">Back to search</a>
            </div>
          </header>

          {sessions.length === 0 && (
            <p className="text-sm text-stone-500 italic">No submission sessions yet.</p>
          )}

          {sessions.length > 0 && (
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-1.5">
                <span className="font-retro text-xs uppercase tracking-wide text-stone-500">Type:</span>
                {(["all", ...QUERY_TYPES] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setFilterType(t)}
                    className={`font-retro text-xs uppercase tracking-wide px-2.5 py-1 rounded border-2 transition ${
                      filterType === t
                        ? "bg-orange-600 text-white border-orange-600"
                        : "border-stone-800 dark:border-stone-500 hover:bg-orange-100 dark:hover:bg-stone-700"
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </div>
              <button className={BTN} onClick={() => setSortByName((v) => !v)}>
                Sort: {sortByName ? "Name A-Z" : "Created"}
              </button>
              <span className="text-xs text-stone-500 dark:text-stone-400">
                Showing {visibleSessions.length} of {sessions.length} session{sessions.length === 1 ? "" : "s"}
                {" — "}
                {QUERY_TYPES.map((t) => `${t}: ${sessions.filter((s) => s.queryType === t).length}`).join(" · ")}
              </span>
            </div>
          )}

          {sessions.length > 0 && visibleSessions.length === 0 && (
            <p className="text-sm text-stone-500 italic">No sessions match this filter.</p>
          )}

          {visibleSessions.map((summary) => {
            const state = states[summary.session];
            if (!state) return null;
            const mode = editMode[summary.session] ?? "grid";
            const draft = newRowDraft[summary.session] || { videoId: "", frames: "" };
            const sizeMismatch = trakeSizeMismatch(state);
            return (
              <section
                key={summary.session}
                className="border-2 border-stone-800 dark:border-stone-500 rounded p-4 bg-cream-card dark:bg-stone-800 space-y-3"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h2 className="font-retro text-lg font-bold">
                    {state.session}{" "}
                    <span className="text-xs font-normal text-stone-500 dark:text-stone-400">
                      {state.queryType.toUpperCase()} · {state.rows.length} row{state.rows.length === 1 ? "" : "s"}
                    </span>
                  </h2>
                  <div className="flex items-center gap-2">
                    <button
                      className={BTN}
                      onClick={() => setEditMode((prev) => ({ ...prev, [summary.session]: mode === "grid" ? "raw" : "grid" }))}
                    >
                      {mode === "grid" ? "📝 Raw CSV" : "▦ Frame grid"}
                    </button>
                    <button className={BTN} onClick={() => handleDownload(state)}>⬇ Download CSV</button>
                    <button className={BTN} onClick={() => handleRenameSession(summary.session)}>✎ Rename</button>
                    <button className={DANGER_BTN} onClick={() => handleDeleteSession(summary.session)}>
                      ✕ Delete
                    </button>
                  </div>
                </div>

                {mode === "raw" ? (
                  <RawCsvEditor
                    session={summary.session}
                    content={buildSubmissionCsv(state).content}
                    onSaved={(updated) => updateState(summary.session, updated)}
                  />
                ) : (
                  <>
                    <div className="flex flex-wrap items-center gap-3">
                      <div className="flex border-2 border-stone-800 dark:border-stone-500 rounded overflow-hidden w-fit">
                        {QUERY_TYPES.map((qt) => (
                          <button
                            key={qt}
                            onClick={() => handleSetQueryType(summary.session, qt)}
                            className={`font-retro text-xs uppercase tracking-wide px-2.5 py-1 transition ${
                              state.queryType === qt
                                ? "bg-orange-600 text-white"
                                : "bg-cream-card dark:bg-stone-800 text-stone-700 dark:text-stone-300 hover:bg-orange-100 dark:hover:bg-stone-700"
                            }`}
                          >
                            {qt.toUpperCase()}
                          </button>
                        ))}
                      </div>
                      {state.queryType === "trake" && (
                        <>
                          <span className="text-sm font-bold text-orange-700 dark:text-orange-400">
                            ▸ Adding to candidate {state.draftRowIndex + 1}
                          </span>
                          <button className={BTN} onClick={() => handleNewCandidate(summary.session)}>+ New candidate</button>
                        </>
                      )}
                    </div>

                    {/* Manual add-row — type a video_id + frame(s) without going through search */}
                    <div className="flex items-center gap-2 flex-wrap border-2 border-dashed border-stone-400 dark:border-stone-600 rounded p-2">
                      <input
                        className={`${SMALL_INPUT} w-32`}
                        placeholder="video_id"
                        value={draft.videoId}
                        onChange={(e) => setNewRowDraft((prev) => ({ ...prev, [summary.session]: { ...draft, videoId: e.target.value } }))}
                      />
                      <input
                        className={`${SMALL_INPUT} w-32`}
                        placeholder={state.queryType === "trake" ? "frame,frame,…" : "frame"}
                        value={draft.frames}
                        onChange={(e) => setNewRowDraft((prev) => ({ ...prev, [summary.session]: { ...draft, frames: e.target.value } }))}
                        onKeyDown={(e) => { if (e.key === "Enter") handleAddRow(summary.session, state.queryType); }}
                      />
                      <button className={BTN} onClick={() => handleAddRow(summary.session, state.queryType)}>+ Add row</button>
                    </div>

                    {sizeMismatch && (
                      <p className="text-xs text-red-600">
                        ⚠ Candidates have different frame counts ({sizeMismatch.join(", ")}) — organizer scoring
                        requires every candidate to match the query's event count exactly.
                      </p>
                    )}

                    {state.rows.length === 0 && (
                      <p className="text-sm text-stone-500 italic">No rows added yet.</p>
                    )}

                    {state.rows.length > 0 && (
                      <div className="space-y-1.5">
                        {state.rows.map((row, i) => {
                          const fps = videoInfo[row.videoId]?.fps ?? 25;
                          const isOver = dragOverRow?.session === summary.session && dragOverRow.row === i;
                          return (
                            <div
                              key={i}
                              draggable
                              onDragStart={(e) => e.dataTransfer.setData("application/x-row-index", String(i))}
                              onDragOver={(e) => { e.preventDefault(); setDragOverRow({ session: summary.session, row: i }); }}
                              onDragLeave={() => setDragOverRow((d) => (d?.session === summary.session && d.row === i ? null : d))}
                              onDrop={(e) => handleRowDrop(e, summary.session, i)}
                              className={`border rounded p-2 space-y-1.5 cursor-grab active:cursor-grabbing transition-colors ${
                                isOver ? "border-orange-600 bg-orange-50 dark:bg-orange-950/30" : "border-stone-300 dark:border-stone-700"
                              }`}
                            >
                              <div className="flex items-center gap-2">
                                <span className="text-stone-400 select-none">⠿</span>
                                <input
                                  key={row.videoId}
                                  defaultValue={row.videoId}
                                  className={`${SMALL_INPUT} w-32`}
                                  onBlur={(e) => handleEditVideoId(summary.session, i, e.target.value)}
                                />
                                {state.queryType === "trake" && (
                                  <button
                                    className="text-xs text-stone-500 hover:text-orange-700 dark:hover:text-orange-400 transition"
                                    onClick={() => handleAddFrameToRow(summary.session, i, row.videoId)}
                                  >
                                    + frame
                                  </button>
                                )}
                                <button
                                  onClick={() => handleRemoveRow(summary.session, i)}
                                  className="ml-auto text-stone-400 hover:text-red-600 transition"
                                  aria-label="Remove row"
                                >
                                  ✕
                                </button>
                              </div>
                              <div className="flex flex-wrap gap-1.5 pl-6">
                                {row.frames.map((frame, fi) => (
                                  <div key={fi} className="flex items-center gap-1 border border-stone-300 dark:border-stone-700 rounded p-1">
                                    <button type="button" onClick={() => openEntry(summary.session, i, row.videoId, frame)}>
                                      <img
                                        src={rowThumbUrl(row.videoId, frame, fps)}
                                        alt=""
                                        className="w-16 h-9 object-cover rounded hover:ring-2 hover:ring-orange-600 transition"
                                      />
                                    </button>
                                    <input
                                      key={frame}
                                      type="number"
                                      min={0}
                                      defaultValue={frame}
                                      className={`${SMALL_INPUT} w-16 text-center`}
                                      onBlur={(e) => handleEditFrame(summary.session, i, fi, e.target.value)}
                                      onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
                                    />
                                    {row.frames.length > 1 && (
                                      <button
                                        onClick={() => handleRemoveFrame(summary.session, i, fi)}
                                        className="text-stone-400 hover:text-red-600 transition text-xs"
                                        aria-label="Remove frame"
                                      >
                                        ✕
                                      </button>
                                    )}
                                  </div>
                                ))}
                              </div>
                              {state.queryType === "qa" && (
                                <div className="pl-6">
                                  <RowAnswerField
                                    answer={row.answer ?? ""}
                                    onSave={(answer) => handleSetAnswer(summary.session, i, answer)}
                                  />
                                </div>
                              )}
                            </div>
                          );
                        })}
                        <p className="text-xs text-stone-500 italic">Drag a row to rank it — export order matches this list.</p>
                      </div>
                    )}
                  </>
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
          showTranscript={showTranscript}
          onToggleTranscript={() => setShowTranscript((v) => !v)}
          onOpenSubmissionPanel={() => window.alert("Pick a working session first, from the search page.")}
          overrideSession={activeContext?.session}
          overrideRowIndex={activeContext?.rowIndex}
        />
      )}
    </>
  );
}
