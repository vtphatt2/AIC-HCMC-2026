import { parsePosition } from "@/lib/submission/format";
import { useCallback, useEffect, useRef, useState } from "react";
import type { SubmissionQueryType, SubmissionSessionSummary, SubmissionState } from "@/types";
import {
  addSubmissionRow,
  addSubmissionRowFrame,
  buildSubmissionCsv,
  createSubmissionSession,
  downloadCsv,
  editRowFrame,
  editRowVideoId,
  fetchSubmission,
  fetchSubmissionSessions,
  moveItem,
  newTrakeCandidate,
  removeRow,
  removeRowFrame,
  reorderSubmissionRows,
  resetSubmission,
  rowThumbUrl,
  setQueryType,
  setRowAnswer,
  trakeSizeMismatch,
  useVideoInfo,
} from "@/lib/submission";

export const SUBMISSION_SESSION_KEY = "aic2026-submission-session";

interface Props {
  onClose: () => void;
}

const INPUT =
  "bg-cream-card dark:bg-stone-800 border-2 border-stone-800 dark:border-stone-500 rounded px-2.5 py-1.5 text-sm text-stone-900 dark:text-stone-50 focus:outline-none focus:ring-2 focus:ring-orange-600";
const BTN =
  "font-retro text-xs uppercase tracking-wide border-2 border-stone-800 dark:border-stone-500 rounded px-2.5 py-1.5 hover:bg-orange-600 hover:text-white hover:border-orange-600 transition";
const SMALL_INPUT =
  "bg-cream-card dark:bg-stone-800 border-2 border-stone-800 dark:border-stone-500 rounded px-1.5 py-0.5 text-sm font-mono text-stone-900 dark:text-stone-50 focus:outline-none focus:ring-2 focus:ring-orange-600";

const QUERY_TYPES: SubmissionQueryType[] = ["kis", "qa", "trake"];

// Local draft while typing, saved on blur — saving on every keystroke (the
// pattern every other field used to follow) round-trips to the server
// before React re-renders the controlled value, which stomps on an
// in-progress Vietnamese IME composition — garbled/dropped characters and
// visible lag.
function DraftField({
  value,
  onSave,
  placeholder,
  className,
  maxLength,
  multiline,
}: {
  value: string;
  onSave: (value: string) => void;
  placeholder?: string;
  className: string;
  maxLength?: number;
  multiline?: boolean;
}) {
  const [draft, setDraft] = useState(value);
  const focused = useRef(false);

  useEffect(() => {
    if (!focused.current) setDraft(value);
  }, [value]);

  const shared = {
    className,
    value: draft,
    placeholder,
    maxLength,
    onFocus: () => { focused.current = true; },
    onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => setDraft(e.target.value),
    onBlur: (e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      focused.current = false;
      if (e.target.value !== value) onSave(e.target.value);
    },
  };
  return multiline ? <textarea {...shared} rows={2} /> : <input {...shared} />;
}

export default function SubmissionPanel({ onClose }: Props) {
  const [sessions, setSessions] = useState<SubmissionSessionSummary[]>([]);
  const [session, setSession] = useState<string>("");
  const [state, setState] = useState<SubmissionState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOverRow, setDragOverRow] = useState<number | null>(null);
  const [newVideoId, setNewVideoId] = useState("");
  const [newFrames, setNewFrames] = useState("");

  const videoInfo = useVideoInfo(state?.rows.map((r) => r.videoId) ?? []);

  useEffect(() => {
    const saved = window.localStorage.getItem(SUBMISSION_SESSION_KEY);
    if (saved) setSession(saved);
  }, []);

  useEffect(() => {
    if (session) window.localStorage.setItem(SUBMISSION_SESSION_KEY, session);
  }, [session]);

  const refreshSessions = useCallback(() => {
    fetchSubmissionSessions().then(setSessions).catch(() => {});
  }, []);

  useEffect(() => {
    refreshSessions();
    const interval = setInterval(refreshSessions, 5000);
    return () => clearInterval(interval);
  }, [refreshSessions]);

  const refreshState = useCallback(() => {
    if (!session) { setState(null); return; }
    fetchSubmission(session)
      .then((s) => { setState(s); setError(null); })
      .catch((err) => setError(err.message));
  }, [session]);

  useEffect(() => {
    refreshState();
    const interval = setInterval(refreshState, 5000);
    return () => clearInterval(interval);
  }, [refreshState]);

  async function handleCreate() {
    // The session name IS the exported filename (session.csv) — name it
    // exactly what BTC's query file is called, e.g. "query-p1-11-kis" for
    // their query-p1-11-kis.txt (see docs/SUBMISSION_RULES.md).
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
      setSession(name);
      refreshSessions();
    } catch (err: any) {
      if (err.status === 409) window.alert("That session name is already taken — pick another.");
      else window.alert(err.message || "Failed to create session");
    }
  }

  async function handleSetQueryType(qt: SubmissionQueryType) {
    if (!session) return;
    setState(await setQueryType(session, qt));
  }

  async function handleAddRow() {
    if (!session || !state) return;
    const videoId = newVideoId.trim();
    if (!videoId) return;
    let frames: number[];
    try { frames = newFrames.split(",").map(parsePosition); }
    catch (error: any) { window.alert(error.message); return; }
    if (frames.length === 0) return;
    if (state.queryType !== "trake" && frames.length > 1) {
      window.alert("kis/qa rows take exactly one frame.");
      return;
    }
    try {
      setState(await addSubmissionRow(session, videoId, frames));
      setNewVideoId("");
      setNewFrames("");
    } catch (err: any) {
      window.alert(err.message || "Failed to add row");
    }
  }

  async function handleAddFrameToRow(rowIndex: number, videoId: string) {
    if (!session) return;
    const raw = window.prompt("Add frame number to this candidate:");
    if (!raw) return;
    let frame: number;
    try { frame = parsePosition(raw); } catch (error: any) { window.alert(error.message); return; }
    if (!Number.isFinite(frame) || frame < 0) return;
    try {
      setState(await addSubmissionRowFrame(session, videoId, frame, rowIndex));
    } catch (err: any) {
      window.alert(err.message || "Failed to add frame");
    }
  }

  async function handleEditVideoId(rowIndex: number, videoId: string) {
    if (!session || !videoId.trim()) return;
    try {
      setState(await editRowVideoId(session, rowIndex, videoId.trim()));
    } catch (err: any) {
      window.alert(err.message || "Invalid video id");
    }
  }

  async function handleEditFrame(rowIndex: number, frameIndex: number, raw: string) {
    let frame: number;
    try { frame = parsePosition(raw); } catch (error: any) { window.alert(error.message); return; }
    if (!session || !Number.isFinite(frame) || frame < 0) return;
    setState(await editRowFrame(session, rowIndex, frameIndex, frame));
  }

  async function handleRemoveFrame(rowIndex: number, frameIndex: number) {
    if (!session) return;
    setState(await removeRowFrame(session, rowIndex, frameIndex));
  }

  async function handleRemoveRow(rowIndex: number) {
    if (!session) return;
    setState(await removeRow(session, rowIndex));
  }

  async function handleSetAnswer(rowIndex: number, answer: string) {
    if (!session) return;
    setState(await setRowAnswer(session, rowIndex, answer));
  }

  async function handleNewCandidate() {
    if (!session) return;
    setState(await newTrakeCandidate(session));
  }

  function handleRowDrop(e: React.DragEvent, targetIndex: number) {
    e.preventDefault();
    setDragOverRow(null);
    if (!session || !state) return;
    const from = Number(e.dataTransfer.getData("application/x-row-index"));
    if (!Number.isFinite(from) || from === targetIndex) return;
    const order = state.rows.map((_, i) => i);
    reorderSubmissionRows(session, moveItem(order, from, targetIndex)).then(setState);
  }

  async function handleReset() {
    if (!session || !window.confirm("Clear all entries in this session?")) return;
    setState(await resetSubmission(session));
  }

  function handleDownload() {
    if (!state) return;
    const { filename, content, rows } = buildSubmissionCsv(state);
    if (rows === 0) { window.alert("No entries to export yet."); return; }
    downloadCsv(filename, content);
  }

  const sizeMismatch = state ? trakeSizeMismatch(state) : null;

  return (
    <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4" onClick={onClose}>
      <div
        className="relative w-full max-w-2xl max-h-[85vh] flex flex-col bg-cream dark:bg-stone-900 border-2 border-stone-800 dark:border-stone-500 rounded shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-3 border-b-2 border-stone-800 dark:border-stone-600 shrink-0">
          <h2 className="font-retro text-lg font-bold text-stone-900 dark:text-stone-50">
            🗳 Submission
          </h2>
          <div className="flex items-center gap-3">
            <a
              href="/submissions"
              target="_blank"
              rel="noreferrer"
              className="font-retro text-stone-500 hover:text-orange-700 dark:hover:text-orange-400 text-xs uppercase tracking-wide transition"
            >
              ↗ Full dashboard
            </a>
            <button onClick={onClose} className="font-retro text-stone-500 hover:text-orange-700 dark:hover:text-orange-400 text-xs uppercase tracking-wide transition">
              ✕ Close
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          {/* Session picker */}
          <div className="flex items-center gap-2">
            <select
              className={INPUT}
              value={session}
              onChange={(e) => setSession(e.target.value)}
            >
              <option value="">— Select a session —</option>
              {sessions.map((s) => (
                <option key={s.session} value={s.session}>
                  {s.session} ({s.queryType}, {s.rowCount})
                </option>
              ))}
            </select>
            <button className={BTN} onClick={handleCreate}>+ New session</button>
          </div>

          {error && <p className="text-sm text-red-600">{error}</p>}

          {state && (
            <>
              {/* Query type */}
              <div className="flex border-2 border-stone-800 dark:border-stone-500 rounded overflow-hidden w-fit">
                {QUERY_TYPES.map((qt) => (
                  <button
                    key={qt}
                    onClick={() => handleSetQueryType(qt)}
                    className={`font-retro text-xs uppercase tracking-wide px-3 py-1.5 transition ${
                      state.queryType === qt
                        ? "bg-orange-600 text-white"
                        : "bg-cream-card dark:bg-stone-800 text-stone-700 dark:text-stone-300 hover:bg-orange-100 dark:hover:bg-stone-700"
                    }`}
                  >
                    {qt.toUpperCase()}
                  </button>
                ))}
              </div>

              {/* Manual add-row — type a video_id + frame(s) without going through search */}
              <div className="flex items-center gap-2 flex-wrap border-2 border-dashed border-stone-400 dark:border-stone-600 rounded p-2">
                <input
                  className={`${SMALL_INPUT} w-32`}
                  placeholder="video_id"
                  value={newVideoId}
                  onChange={(e) => setNewVideoId(e.target.value)}
                />
                <input
                  className={`${SMALL_INPUT} w-32`}
                  placeholder={newVideoId.startsWith("N") ? "source milliseconds" : state.queryType === "trake" ? "frame,frame,…" : "frame"}
                  value={newFrames}
                  onChange={(e) => setNewFrames(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") handleAddRow(); }}
                />
                <button className={BTN} onClick={handleAddRow}>+ Add row</button>
              </div>

              {state.queryType === "trake" && (
                <div className="space-y-1">
                  <p className="text-sm font-bold text-orange-700 dark:text-orange-400">
                    ▸ Adding to candidate {state.draftRowIndex + 1}
                  </p>
                  <button className={BTN} onClick={handleNewCandidate}>+ New candidate</button>
                  {sizeMismatch && (
                    <p className="text-xs text-red-600">
                      ⚠ Candidates have different frame counts ({sizeMismatch.join(", ")}) — organizer scoring
                      requires every candidate to match the query's event count exactly.
                    </p>
                  )}
                </div>
              )}

              {/* Rows */}
              {state.rows.length === 0 && (
                <p className="text-sm text-stone-500 italic">No rows added yet.</p>
              )}

              {state.rows.length > 0 && (
                <div className="space-y-1.5">
                  {state.rows.map((row, i) => {
                    const fps = videoInfo[row.videoId]?.fps ?? 25;
                    return (
                      <div
                        key={i}
                        draggable
                        onDragStart={(e) => e.dataTransfer.setData("application/x-row-index", String(i))}
                        onDragOver={(e) => { e.preventDefault(); setDragOverRow(i); }}
                        onDragLeave={() => setDragOverRow((d) => (d === i ? null : d))}
                        onDrop={(e) => handleRowDrop(e, i)}
                        className={`border rounded p-2 space-y-1.5 cursor-grab active:cursor-grabbing transition-colors ${
                          dragOverRow === i
                            ? "border-orange-600 bg-orange-50 dark:bg-orange-950/30"
                            : "border-stone-300 dark:border-stone-700"
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-stone-400 select-none">⠿</span>
                          <input
                            key={row.videoId}
                            defaultValue={row.videoId}
                            className={`${SMALL_INPUT} w-32`}
                            onBlur={(e) => handleEditVideoId(i, e.target.value)}
                          />
                          {state.queryType === "trake" && (
                            <button
                              className="text-xs text-stone-500 hover:text-orange-700 dark:hover:text-orange-400 transition"
                              onClick={() => handleAddFrameToRow(i, row.videoId)}
                            >
                              + frame
                            </button>
                          )}
                          <button
                            onClick={() => handleRemoveRow(i)}
                            className="ml-auto text-stone-400 hover:text-red-600 transition"
                            aria-label="Remove row"
                          >
                            ✕
                          </button>
                        </div>
                        <div className="flex flex-wrap gap-1.5 pl-6">
                          <span className="text-xs">{row.unit === "milliseconds" ? "ms" : "frames"}{row.timingStatus === "unresolved" ? " · timing unresolved" : ""}</span>
                          {row.frames.map((frame, fi) => (
                            <div key={fi} className="flex items-center gap-1 border border-stone-300 dark:border-stone-700 rounded p-1">
                              <img src={rowThumbUrl(row.videoId, frame, fps, row.sourceFrames?.[row.frames.indexOf(frame)])} alt="" className="w-14 h-8 object-cover rounded" />
                              <input
                                key={frame}
                                type="number"
                                min={0}
                                defaultValue={frame}
                                className={`${SMALL_INPUT} w-16 text-center`}
                                onBlur={(e) => handleEditFrame(i, fi, e.target.value)}
                                onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
                              />
                              {row.frames.length > 1 && (
                                <button
                                  onClick={() => handleRemoveFrame(i, fi)}
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
                          <div className="pl-6 space-y-0.5">
                            <DraftField
                              value={row.answer ?? ""}
                              onSave={(v) => handleSetAnswer(i, v)}
                              placeholder="Answer for this candidate"
                              className={`${INPUT} w-full`}
                              maxLength={100}
                              multiline
                            />
                            <p className="text-xs text-stone-500 text-right">{(row.answer ?? "").length}/100</p>
                          </div>
                        )}
                      </div>
                    );
                  })}
                  <p className="text-xs text-stone-500 italic">Drag a row to rank it — export order matches this list.</p>
                </div>
              )}

              <p className="text-xs text-stone-500">
                {state.rows.length} row{state.rows.length === 1 ? "" : "s"} (organizer cap: 100 rows/query).
              </p>
            </>
          )}

          {!state && !error && session && <p className="text-sm text-stone-500 italic">Loading…</p>}
          {!session && <p className="text-sm text-stone-500 italic">Pick or create a session to start.</p>}
        </div>

        {state && (
          <div className="flex items-center gap-2 px-5 py-3 border-t-2 border-stone-800 dark:border-stone-600 shrink-0">
            <button className={BTN} onClick={handleDownload}>⬇ Download CSV</button>
            <button className={BTN} onClick={handleReset}>🗑 Reset</button>
          </div>
        )}
      </div>
    </div>
  );
}
