import { useCallback, useEffect, useState } from "react";
import type { SubmissionQueryType, SubmissionSessionSummary, SubmissionState } from "@/types";
import {
  buildSubmissionCsv,
  createSubmissionSession,
  downloadCsv,
  fetchSubmission,
  fetchSubmissionSessions,
  newSubmissionCandidate,
  removeSubmissionEntry,
  resetSubmission,
  setSubmissionMeta,
  trakeCandidateSizeMismatch,
} from "@/lib/submission";

export const SUBMISSION_SESSION_KEY = "aic2026-submission-session";

interface Props {
  onClose: () => void;
}

const INPUT =
  "bg-cream-card dark:bg-stone-800 border-2 border-stone-800 dark:border-stone-500 rounded px-2.5 py-1.5 text-sm text-stone-900 dark:text-stone-50 focus:outline-none focus:ring-2 focus:ring-orange-600";
const BTN =
  "font-retro text-xs uppercase tracking-wide border-2 border-stone-800 dark:border-stone-500 rounded px-2.5 py-1.5 hover:bg-orange-600 hover:text-white hover:border-orange-600 transition";

const QUERY_TYPES: SubmissionQueryType[] = ["kis", "qa", "trake"];

export default function SubmissionPanel({ onClose }: Props) {
  const [sessions, setSessions] = useState<SubmissionSessionSummary[]>([]);
  const [session, setSession] = useState<string>("");
  const [state, setState] = useState<SubmissionState | null>(null);
  const [error, setError] = useState<string | null>(null);

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
    const name = window.prompt("Session name (e.g. query-3-kis):")?.trim();
    if (!name) return;
    try {
      await createSubmissionSession(name);
      setSession(name);
      refreshSessions();
    } catch (err: any) {
      if (err.status === 409) window.alert("That session name is already taken — pick another.");
      else window.alert(err.message || "Failed to create session");
    }
  }

  async function handleMeta(patch: Partial<Pick<SubmissionState, "queryType" | "queryNumber" | "answer">>) {
    if (!session) return;
    setState(await setSubmissionMeta(session, patch));
  }

  async function handleRemove(id: string) {
    if (!session) return;
    setState(await removeSubmissionEntry(session, id));
  }

  async function handleNewCandidate() {
    if (!session) return;
    setState(await newSubmissionCandidate(session));
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

  const groupCount = state ? new Set(state.entries.map((e) => e.groupIndex)).size : 0;
  const sizeMismatch = state ? trakeCandidateSizeMismatch(state) : null;

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
                  {s.session} ({s.queryType}, {s.entryCount})
                </option>
              ))}
            </select>
            <button className={BTN} onClick={handleCreate}>+ New session</button>
          </div>

          {error && <p className="text-sm text-red-600">{error}</p>}

          {state && (
            <>
              {/* Query type + number */}
              <div className="flex flex-wrap items-center gap-3">
                <div className="flex border-2 border-stone-800 dark:border-stone-500 rounded overflow-hidden">
                  {QUERY_TYPES.map((qt) => (
                    <button
                      key={qt}
                      onClick={() => handleMeta({ queryType: qt })}
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
                <label className="flex items-center gap-1.5 text-sm text-stone-600 dark:text-stone-400">
                  Query #
                  <input
                    type="number"
                    min={1}
                    className={`${INPUT} w-20`}
                    value={state.queryNumber}
                    onChange={(e) => handleMeta({ queryNumber: Number.parseInt(e.target.value, 10) || 1 })}
                  />
                </label>
              </div>

              {state.queryType === "qa" && (
                <div className="space-y-1">
                  <textarea
                    className={`${INPUT} w-full`}
                    placeholder="Answer text (applied to every row on export)"
                    value={state.answer}
                    maxLength={100}
                    onChange={(e) => handleMeta({ answer: e.target.value })}
                    rows={2}
                  />
                  <p className="text-xs text-stone-500 text-right">{state.answer.length}/100</p>
                </div>
              )}

              {state.queryType === "trake" && (
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <button className={BTN} onClick={handleNewCandidate}>+ New candidate</button>
                    <span className="text-xs text-stone-500">
                      {groupCount} candidate{groupCount === 1 ? "" : "s"} so far — "Add to submission" from the
                      video modal appends to the current one.
                    </span>
                  </div>
                  {sizeMismatch && (
                    <p className="text-xs text-red-600">
                      ⚠ Candidates have different frame counts ({sizeMismatch.join(", ")}) — organizer scoring
                      requires every candidate to match the query's event count exactly.
                    </p>
                  )}
                </div>
              )}

              {/* Entries */}
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
                      <img src={entry.imageUrl} alt="" className="w-14 h-8 object-cover rounded shrink-0" />
                    )}
                    <span className="text-sm font-mono text-stone-800 dark:text-stone-200 truncate">
                      {entry.videoId} · frame {entry.frame}
                      {state.queryType === "trake" && ` · candidate ${entry.groupIndex + 1}`}
                    </span>
                    <button
                      onClick={() => handleRemove(entry.id)}
                      className="ml-auto text-stone-400 hover:text-red-600 transition"
                      aria-label="Remove"
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>

              <p className="text-xs text-stone-500">
                {state.entries.length} row candidate{state.entries.length === 1 ? "" : "s"} added (organizer cap: 100 rows/query).
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
