import { useEffect, useMemo, useState } from "react";

import {
  createSubmissionSession,
  deleteSubmissionSession,
  renameSubmissionSession,
} from "@/lib/submission";
import { parseSessionNames, sessionNameError } from "@/lib/submission/sessionImport";
import type { SubmissionQueryType, SubmissionSessionSummary } from "@/types";

const QUERY_TYPES: SubmissionQueryType[] = ["kis", "qa", "trake"];
const INPUT =
  "bg-cream-card dark:bg-stone-800 border-2 border-stone-800 dark:border-stone-500 rounded px-2.5 py-1.5 text-sm font-mono text-stone-900 dark:text-stone-50 focus:outline-none focus:ring-2 focus:ring-orange-600";
const BTN =
  "font-retro text-xs uppercase tracking-wide border-2 border-stone-800 dark:border-stone-500 rounded px-2.5 py-1.5 hover:bg-orange-600 hover:text-white hover:border-orange-600 transition disabled:opacity-40 disabled:cursor-not-allowed";
const PRIMARY_BTN =
  "font-retro text-xs uppercase tracking-wide border-2 border-orange-600 rounded px-2.5 py-1.5 bg-orange-600 text-white hover:bg-orange-700 hover:border-orange-700 transition disabled:opacity-40 disabled:cursor-not-allowed";
const DANGER_BTN =
  "font-retro text-xs uppercase tracking-wide border-2 border-stone-800 dark:border-stone-500 rounded px-2.5 py-1.5 hover:bg-red-600 hover:text-white hover:border-red-600 transition disabled:opacity-40 disabled:cursor-not-allowed";

interface ImportReport {
  queryType: SubmissionQueryType;
  created: string[];
  skipped: string[];
  duplicates: string[];
  invalid: string[];
  failed: Array<{ name: string; message: string }>;
}

interface SessionManagerModalProps {
  sessions: SubmissionSessionSummary[];
  onClose: () => void;
  onChanged: () => Promise<void>;
}

function apiError(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function apiStatus(error: unknown): number | undefined {
  return typeof error === "object" && error !== null && "status" in error
    ? Number((error as { status?: unknown }).status)
    : undefined;
}

export default function SessionManagerModal({
  sessions,
  onClose,
  onChanged,
}: SessionManagerModalProps) {
  const [bulkNames, setBulkNames] = useState("");
  const [queryType, setQueryType] = useState<SubmissionQueryType>("kis");
  const [newName, setNewName] = useState("");
  const [renameDrafts, setRenameDrafts] = useState<Record<string, string>>({});
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [message, setMessage] = useState<{ kind: "error" | "success"; text: string } | null>(null);
  const [report, setReport] = useState<ImportReport | null>(null);

  const existingNames = useMemo(() => sessions.map((session) => session.session), [sessions]);
  const parsed = useMemo(
    () => parseSessionNames(bulkNames, existingNames),
    [bulkNames, existingNames],
  );

  useEffect(() => {
    setRenameDrafts((current) => Object.fromEntries(
      sessions.map((session) => [session.session, current[session.session] ?? session.session]),
    ));
  }, [sessions]);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape" && !busyAction) onClose();
    }
    window.addEventListener("keydown", handleKeyDown);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [busyAction, onClose]);

  async function handleImport() {
    if (parsed.valid.length === 0) {
      setMessage({ kind: "error", text: "Không có tên session hợp lệ và chưa tồn tại để import." });
      return;
    }

    setBusyAction("import");
    setMessage(null);
    const nextReport: ImportReport = {
      queryType,
      created: [],
      skipped: [...parsed.existing],
      duplicates: [...parsed.duplicates],
      invalid: [...parsed.invalid],
      failed: [],
    };

    for (const name of parsed.valid) {
      try {
        await createSubmissionSession(name, queryType);
        nextReport.created.push(name);
      } catch (error) {
        if (apiStatus(error) === 409) nextReport.skipped.push(name);
        else nextReport.failed.push({ name, message: apiError(error, "Tạo session thất bại") });
      }
    }

    setReport(nextReport);
    if (nextReport.created.length > 0) {
      setBulkNames([
        ...nextReport.invalid,
        ...nextReport.failed.map(({ name }) => name),
      ].join("\n"));
    }
    setMessage({
      kind: nextReport.failed.length > 0 || nextReport.invalid.length > 0 ? "error" : "success",
      text: `Đã import ${nextReport.created.length}; đã tồn tại ${nextReport.skipped.length}; trùng input ${nextReport.duplicates.length}; không hợp lệ ${nextReport.invalid.length}; lỗi ${nextReport.failed.length}.`,
    });
    await onChanged();
    setBusyAction(null);
  }

  async function handleNew() {
    const name = newName.trim();
    const validationError = sessionNameError(name);
    if (validationError) {
      setMessage({ kind: "error", text: validationError });
      return;
    }
    if (existingNames.includes(name)) {
      setMessage({ kind: "error", text: `Session “${name}” đã tồn tại.` });
      return;
    }

    setBusyAction("new");
    setMessage(null);
    try {
      await createSubmissionSession(name, queryType);
      setNewName("");
      setMessage({ kind: "success", text: `Đã tạo session “${name}” (${queryType.toUpperCase()}).` });
      await onChanged();
    } catch (error) {
      setMessage({ kind: "error", text: apiError(error, "Tạo session thất bại") });
    } finally {
      setBusyAction(null);
    }
  }

  async function handleRename(session: string) {
    const nextName = (renameDrafts[session] ?? session).trim();
    const validationError = sessionNameError(nextName);
    if (validationError) {
      setMessage({ kind: "error", text: validationError });
      return;
    }
    if (nextName === session) return;

    setBusyAction(`rename:${session}`);
    setMessage(null);
    try {
      await renameSubmissionSession(session, nextName);
      setMessage({ kind: "success", text: `Đã đổi “${session}” thành “${nextName}”.` });
      await onChanged();
    } catch (error) {
      setMessage({ kind: "error", text: apiError(error, "Đổi tên session thất bại") });
    } finally {
      setBusyAction(null);
    }
  }

  async function handleDelete(session: string) {
    if (!window.confirm(`Xóa session “${session}”? Dữ liệu trong session sẽ bị xóa vĩnh viễn.`)) return;

    setBusyAction(`delete:${session}`);
    setMessage(null);
    try {
      await deleteSubmissionSession(session);
      setMessage({ kind: "success", text: `Đã xóa session “${session}”.` });
      await onChanged();
    } catch (error) {
      setMessage({ kind: "error", text: apiError(error, "Xóa session thất bại") });
    } finally {
      setBusyAction(null);
    }
  }

  const isBusy = busyAction !== null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-3 md:p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="session-manager-title"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !isBusy) onClose();
      }}
    >
      <div className="flex max-h-[94vh] w-full max-w-5xl flex-col overflow-hidden rounded border-2 border-stone-800 bg-cream-card text-stone-900 shadow-2xl dark:border-stone-500 dark:bg-stone-900 dark:text-stone-50">
        <header className="flex items-start justify-between gap-4 border-b-2 border-stone-800 p-4 dark:border-stone-500">
          <div>
            <h2 id="session-manager-title" className="font-retro text-xl font-bold uppercase">
              Session manager
            </h2>
            <p className="text-sm text-stone-500 dark:text-stone-400">
              Import nhiều session, hoặc tạo, đổi tên và xóa session tại đây.
            </p>
          </div>
          <button className={BTN} onClick={onClose} disabled={isBusy} aria-label="Close session manager">
            ✕ Close
          </button>
        </header>

        <div className="overflow-y-auto p-4 md:p-5 space-y-5">
          <section className="grid gap-4 md:grid-cols-[minmax(0,1fr)_220px]">
            <div className="space-y-2">
              <label htmlFor="bulk-session-names" className="font-retro text-sm font-bold uppercase">
                Paste session names
              </label>
              <textarea
                id="bulk-session-names"
                className={`${INPUT} min-h-36 w-full resize-y`}
                placeholder={"query-p1-11-kis\nquery-p1-12-kis\nquery-p1-13-kis"}
                value={bulkNames}
                disabled={isBusy}
                onChange={(event) => {
                  setBulkNames(event.target.value);
                  setMessage(null);
                }}
              />
              <p className="text-xs text-stone-500 dark:text-stone-400">
                Mỗi dòng là một tên. Tên chỉ chứa chữ, số, dấu gạch dưới hoặc gạch ngang; tối đa 64 ký tự.
              </p>
            </div>

            <div className="space-y-3 rounded border-2 border-stone-300 p-3 dark:border-stone-700">
              <label htmlFor="session-query-type" className="block font-retro text-sm font-bold uppercase">
                Query type
              </label>
              <select
                id="session-query-type"
                className={`${INPUT} w-full uppercase`}
                value={queryType}
                disabled={isBusy}
                onChange={(event) => setQueryType(event.target.value as SubmissionQueryType)}
              >
                {QUERY_TYPES.map((type) => <option key={type} value={type}>{type.toUpperCase()}</option>)}
              </select>
              <div className="space-y-1 text-xs text-stone-600 dark:text-stone-300">
                <p><strong>{parsed.valid.length}</strong> sẵn sàng import</p>
                <p><strong>{parsed.existing.length}</strong> đã tồn tại</p>
                <p><strong>{parsed.duplicates.length}</strong> trùng trong input</p>
                <p className={parsed.invalid.length ? "text-red-600" : ""}>
                  <strong>{parsed.invalid.length}</strong> không hợp lệ
                </p>
              </div>
              <button
                className={`${PRIMARY_BTN} w-full`}
                onClick={handleImport}
                disabled={isBusy || parsed.valid.length === 0}
              >
                {busyAction === "import" ? "Importing…" : `Import ${parsed.valid.length} session${parsed.valid.length === 1 ? "" : "s"}`}
              </button>
            </div>
          </section>

          {(parsed.invalid.length > 0 || parsed.duplicates.length > 0 || parsed.existing.length > 0) && (
            <div className="grid gap-2 text-xs md:grid-cols-3">
              {parsed.invalid.length > 0 && (
                <p className="rounded border border-red-400 bg-red-50 p-2 text-red-800 dark:bg-red-950/30 dark:text-red-300">
                  <strong>Không hợp lệ:</strong> {parsed.invalid.join(", ")}
                </p>
              )}
              {parsed.duplicates.length > 0 && (
                <p className="rounded border border-amber-400 bg-amber-50 p-2 text-amber-800 dark:bg-amber-950/30 dark:text-amber-300">
                  <strong>Trùng input:</strong> {parsed.duplicates.join(", ")}
                </p>
              )}
              {parsed.existing.length > 0 && (
                <p className="rounded border border-stone-400 p-2 text-stone-600 dark:text-stone-300">
                  <strong>Đã tồn tại:</strong> {parsed.existing.join(", ")}
                </p>
              )}
            </div>
          )}

          {message && (
            <p
              className={`rounded border p-2 text-sm ${
                message.kind === "success"
                  ? "border-green-500 bg-green-50 text-green-800 dark:bg-green-950/30 dark:text-green-300"
                  : "border-red-500 bg-red-50 text-red-800 dark:bg-red-950/30 dark:text-red-300"
              }`}
            >
              {message.text}
            </p>
          )}

          {report && (
            <section className="space-y-2 rounded border-2 border-green-600/60 bg-green-50 p-3 dark:bg-green-950/20">
              <h3 className="font-retro text-sm font-bold uppercase">
                Kết quả import · {report.queryType.toUpperCase()}
              </h3>
              {report.created.length > 0 && (
                <div>
                  <p className="text-xs font-bold text-green-800 dark:text-green-300">Đã import ({report.created.length})</p>
                  <div className="mt-1 flex flex-wrap gap-1.5">
                    {report.created.map((name) => (
                      <span key={name} className="rounded bg-green-700 px-2 py-1 font-mono text-xs text-white">{name}</span>
                    ))}
                  </div>
                </div>
              )}
              {report.skipped.length > 0 && (
                <p className="text-xs text-stone-600 dark:text-stone-300">
                  <strong>Bỏ qua vì đã tồn tại:</strong> {report.skipped.join(", ")}
                </p>
              )}
              {report.duplicates.length > 0 && (
                <p className="text-xs text-amber-700 dark:text-amber-300">
                  <strong>Trùng trong input:</strong> {report.duplicates.join(", ")}
                </p>
              )}
              {report.invalid.length > 0 && (
                <p className="text-xs text-red-700 dark:text-red-300">
                  <strong>Tên không hợp lệ (đã giữ lại trong ô nhập):</strong> {report.invalid.join(", ")}
                </p>
              )}
              {report.failed.length > 0 && (
                <div className="text-xs text-red-700 dark:text-red-300">
                  <strong>Lỗi:</strong>
                  <ul className="list-disc pl-5">
                    {report.failed.map(({ name, message: failure }) => <li key={name}>{name}: {failure}</li>)}
                  </ul>
                </div>
              )}
            </section>
          )}

          <section className="space-y-3 border-t-2 border-stone-300 pt-4 dark:border-stone-700">
            <div>
              <h3 className="font-retro text-sm font-bold uppercase">New session</h3>
              <p className="text-xs text-stone-500 dark:text-stone-400">
                Session mới dùng query type đang chọn ở phía trên: <strong>{queryType.toUpperCase()}</strong>.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <input
                className={`${INPUT} min-w-64 flex-1`}
                placeholder="query-p1-14-qa"
                value={newName}
                disabled={isBusy}
                onChange={(event) => setNewName(event.target.value)}
                onKeyDown={(event) => { if (event.key === "Enter") handleNew(); }}
              />
              <button className={PRIMARY_BTN} onClick={handleNew} disabled={isBusy || !newName.trim()}>
                {busyAction === "new" ? "Creating…" : "+ New"}
              </button>
            </div>
          </section>

          <section className="space-y-3 border-t-2 border-stone-300 pt-4 dark:border-stone-700">
            <div className="flex items-center justify-between gap-3">
              <h3 className="font-retro text-sm font-bold uppercase">All sessions</h3>
              <span className="text-xs text-stone-500">{sessions.length} total</span>
            </div>
            {sessions.length === 0 ? (
              <p className="text-sm italic text-stone-500">Chưa có session nào.</p>
            ) : (
              <div className="divide-y divide-stone-300 overflow-hidden rounded border-2 border-stone-300 dark:divide-stone-700 dark:border-stone-700">
                {sessions.map((summary) => {
                  const draft = renameDrafts[summary.session] ?? summary.session;
                  const rowBusy = busyAction === `rename:${summary.session}` || busyAction === `delete:${summary.session}`;
                  return (
                    <div key={summary.session} className="flex flex-wrap items-center gap-2 p-2.5">
                      <input
                        className={`${INPUT} min-w-56 flex-1`}
                        value={draft}
                        disabled={isBusy}
                        aria-label={`Rename ${summary.session}`}
                        onChange={(event) => setRenameDrafts((current) => ({
                          ...current,
                          [summary.session]: event.target.value,
                        }))}
                        onKeyDown={(event) => { if (event.key === "Enter") handleRename(summary.session); }}
                      />
                      <span className="min-w-16 rounded bg-stone-200 px-2 py-1 text-center font-retro text-xs uppercase dark:bg-stone-700">
                        {summary.queryType}
                      </span>
                      <span className="min-w-16 text-right text-xs text-stone-500">
                        {summary.rowCount} row{summary.rowCount === 1 ? "" : "s"}
                      </span>
                      <button
                        className={BTN}
                        onClick={() => handleRename(summary.session)}
                        disabled={isBusy || draft.trim() === summary.session || Boolean(sessionNameError(draft))}
                      >
                        {busyAction === `rename:${summary.session}` ? "Saving…" : "Rename"}
                      </button>
                      <button className={DANGER_BTN} onClick={() => handleDelete(summary.session)} disabled={isBusy}>
                        {busyAction === `delete:${summary.session}` ? "Deleting…" : "Delete"}
                      </button>
                      {rowBusy && <span className="sr-only">Updating {summary.session}</span>}
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
