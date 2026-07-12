import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import type { QueryGroup, Strategy } from "@/types";

type FieldKey = "semantic" | "text" | "offset" | "translate";
const FIELD_ORDER: FieldKey[] = ["semantic", "text", "offset", "translate"];
const SLASH_COMMANDS = [
  "/step add", "/step del", "/step clear", "/text", "/translate", "/mode", "/topk",
  "/genre", "/strategy", "/view", "/transcript", "/search", "/clear", "/help",
];
const COMMAND_USAGE: Record<string, string> = {
  "/step add": "/step add [x] — insert a new step at position x (default: end); steps at x.. shift +1",
  "/step del": "/step del [x] — remove step at position x (default: last step)",
  "/step clear": "/step clear [x] — clear semantic/text/translate of step x (default: active step)",
  "/text": "/text <value> — set the OCR/transcript text field of the active step",
  "/translate": "/translate on|off — toggle VI→EN translation for the active step",
  "/mode": "/mode frames|transcripts — switch search mode",
  "/topk": "/topk <n> — set number of results to fetch",
  "/genre": "/genre <name> — filter results by genre",
  "/strategy": "/strategy <id|name> — select the search strategy",
  "/view": "/view score|video — switch results view (grouped by video vs. flat score list)",
  "/transcript": "/transcript on|off — show/hide the transcript panel in the video modal",
  "/search": "/search — run the search immediately",
  "/clear": "/clear — reset to a single empty step (or clear transcript query)",
  "/help": "/help — list all commands",
};

interface Props {
  searchMode: "frames" | "transcripts";
  defaultGroup: QueryGroup;
  strategies: Strategy[];
  selectedStrategy: string;
  setSelectedStrategy: (id: string) => void;
  queryGroups: QueryGroup[];
  setQueryGroups: (updater: QueryGroup[] | ((prev: QueryGroup[]) => QueryGroup[])) => void;
  topKInput: string;
  setTopKInput: (v: string) => void;
  videoGenre: string;
  setVideoGenre: (v: string) => void;
  transcriptQuery: string;
  setTranscriptQuery: (v: string) => void;
  transcriptTopK: string;
  setTranscriptTopK: (v: string) => void;
  transcriptGenre: string;
  setTranscriptGenre: (v: string) => void;
  allGenres: string[];
  viewMode: "score" | "video";
  setViewMode: (v: "score" | "video") => void;
  transcriptViewMode: "score" | "video";
  setTranscriptViewMode: (v: "score" | "video") => void;
  onSearch: () => void;
  onSetMode: (mode: "frames" | "transcripts") => void;
  onEscapeToResults: () => void;
  showTranscript: boolean;
  setShowTranscript: (v: boolean) => void;
}

export interface CommandPanelHandle {
  focus: () => void;
}

function getFieldDisplay(g: QueryGroup, field: FieldKey): string {
  switch (field) {
    case "semantic": return g.semanticQuery;
    case "text": return g.textQuery;
    case "offset": return String(g.temporalOffsetMs);
    case "translate": return g.translateSemantic ? "on" : "off";
  }
}

function applyFieldValue(g: QueryGroup, field: FieldKey, raw: string): QueryGroup {
  switch (field) {
    case "semantic": return { ...g, semanticQuery: raw, translatedQuery: "" };
    case "text": return { ...g, textQuery: raw };
    case "offset": {
      const n = Number.parseInt(raw, 10);
      return Number.isNaN(n) ? g : { ...g, temporalOffsetMs: Math.max(0, n) };
    }
    case "translate": return { ...g, translateSemantic: /^(on|true|y|yes|1)$/i.test(raw), translatedQuery: "" };
  }
}

const CommandPanel = forwardRef<CommandPanelHandle, Props>(function CommandPanel({
  searchMode, defaultGroup, strategies, selectedStrategy, setSelectedStrategy,
  queryGroups, setQueryGroups, topKInput, setTopKInput, videoGenre, setVideoGenre,
  transcriptQuery, setTranscriptQuery, transcriptTopK, setTranscriptTopK,
  transcriptGenre, setTranscriptGenre, allGenres,
  viewMode, setViewMode, transcriptViewMode, setTranscriptViewMode,
  onSearch, onSetMode, onEscapeToResults,
  showTranscript, setShowTranscript,
}: Props, ref) {
  const [activeStepIndex, setActiveStepIndex] = useState(0);
  const [activeField, setActiveField] = useState<FieldKey | null>(null);
  const [correctionMode, setCorrectionMode] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [status, setStatusState] = useState<{ text: string; kind: "info" | "error" } | null>(null);
  const [undoStack, setUndoStack] = useState<QueryGroup[][]>([]);
  const [history, setHistory] = useState<string[]>([]);
  const [historyPointer, setHistoryPointer] = useState<number | null>(null);
  const [tabCycle, setTabCycle] = useState<{ base: string; options: string[]; index: number; isArg: boolean } | null>(null);
  const [searchTick, setSearchTick] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const statusTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isFirstSearchTick = useRef(true);

  useImperativeHandle(ref, () => ({
    focus: () => inputRef.current?.focus(),
  }));

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Runs after the state update that preceded it has committed, so `onSearch`
  // (which reads the parent's queryGroups/transcriptQuery via closure) always
  // sees fresh data instead of the stale snapshot from the keystroke's own render.
  useEffect(() => {
    if (isFirstSearchTick.current) { isFirstSearchTick.current = false; return; }
    onSearch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchTick]);

  function requestSearch() {
    setSearchTick((t) => t + 1);
  }

  useEffect(() => {
    setActiveStepIndex((i) => Math.min(i, Math.max(0, queryGroups.length - 1)));
  }, [queryGroups.length]);

  function setStatus(text: string, kind: "info" | "error" = "info") {
    if (statusTimer.current) clearTimeout(statusTimer.current);
    setStatusState({ text, kind });
    if (kind === "info") {
      statusTimer.current = setTimeout(() => setStatusState(null), 2500);
    }
  }

  function pushUndo() {
    setUndoStack((prev) => {
      const next = [...prev, queryGroups.map((g) => ({ ...g }))];
      return next.length > 20 ? next.slice(next.length - 20) : next;
    });
  }

  function undo() {
    if (undoStack.length === 0) {
      setStatus("Nothing to undo", "error");
      return;
    }
    const last = undoStack[undoStack.length - 1];
    setQueryGroups(last);
    setUndoStack((prev) => prev.slice(0, -1));
    setStatus("↺ undo");
  }

  // Resolves the enum-like argument options for whatever command the user is
  // currently typing (e.g. "/view " → ["score","video"]), so Up/Down can
  // cycle them without the user needing to type/remember exact values.
  function resolveArgOptions(inputVal: string): { base: string; options: string[] } | null {
    const argSources: Record<string, string[]> = {
      "/step": ["add", "del", "clear"],
      "/translate": ["on", "off"],
      "/mode": ["frames", "transcripts"],
      "/view": ["score", "video"],
      "/transcript": ["on", "off"],
      "/genre": allGenres,
      "/strategy": strategies.map((s) => s.id),
    };
    for (const [base, options] of Object.entries(argSources)) {
      if (inputVal === base || inputVal.startsWith(base + " ")) {
        const rest = inputVal.slice(base.length).trimStart();
        if (!rest.includes(" ")) return { base, options };
      }
    }
    return null;
  }

  function pushHistory(raw: string) {
    setHistory((prev) => [...prev, raw].slice(-50));
    setHistoryPointer(null);
  }

  function recallHistory(direction: number) {
    if (history.length === 0) return;
    const base = historyPointer === null ? history.length : historyPointer;
    const next = Math.min(history.length - 1, Math.max(0, base + direction));
    setHistoryPointer(next);
    setInputValue(history[next]);
  }

  // ── Frames mode: step/field sequence ──────────────────────────────────────
  function advance() {
    const fi = activeField ? FIELD_ORDER.indexOf(activeField) : -1;
    if (fi >= 0 && fi < FIELD_ORDER.length - 1) {
      setActiveField(FIELD_ORDER[fi + 1]);
      return;
    }
    if (activeStepIndex >= queryGroups.length - 1) {
      pushUndo();
      setQueryGroups((prev) => [...prev, { ...defaultGroup }]);
    }
    setActiveStepIndex((i) => i + 1);
    setActiveField(FIELD_ORDER[0]);
  }

  function confirmField(raw: string) {
    pushUndo();
    const idx = activeStepIndex;
    const field = activeField!;
    const trimmed = raw.trim();
    // During auto-advance, an empty submission means "skip, keep existing value".
    // During F2 correction, an empty submission is a deliberate clear.
    const shouldApply = correctionMode || trimmed !== "";
    if (shouldApply) {
      setQueryGroups((prev) => {
        const updated = [...prev];
        updated[idx] = applyFieldValue(updated[idx], field, trimmed);
        return updated;
      });
      setStatus(trimmed === "" ? `✓ step ${idx + 1} ${field} cleared` : `✓ step ${idx + 1} ${field} set`);
    }
    if (correctionMode) {
      setCorrectionMode(false);
      setActiveField(null);
    } else {
      advance();
    }
    setInputValue("");
    requestSearch();
  }

  function startCorrection() {
    const g = queryGroups[activeStepIndex];
    if (!g) return;
    setCorrectionMode(true);
    setActiveField("semantic");
    setInputValue(getFieldDisplay(g, "semantic"));
  }

  function applyOneShot(field: FieldKey, raw: string) {
    if (!raw.trim()) {
      setStatus("Missing value", "error");
      return;
    }
    pushUndo();
    const idx = activeStepIndex;
    setQueryGroups((prev) => {
      const updated = [...prev];
      updated[idx] = applyFieldValue(updated[idx], field, raw.trim());
      return updated;
    });
    setStatus(`✓ step ${idx + 1} ${field} set`);
  }

  function runSlashCommand(raw: string) {
    const [cmd, ...rest] = raw.slice(1).trim().split(/\s+/);
    const arg = rest.join(" ");
    switch (cmd) {
      case "step": {
        if (searchMode !== "frames") { setStatus("/step only applies in Frames mode", "error"); return; }
        const sub = rest[0];
        if (sub === "add") {
          const posArg = rest[1];
          const pos = posArg && /^\d+$/.test(posArg)
            ? Math.min(Math.max(1, Number.parseInt(posArg, 10)), queryGroups.length + 1)
            : queryGroups.length + 1;
          pushUndo();
          setQueryGroups((prev) => {
            const updated = [...prev];
            updated.splice(pos - 1, 0, { ...defaultGroup });
            return updated;
          });
          setActiveStepIndex(pos - 1);
          setActiveField(null);
          setStatus(`✓ step inserted at ${pos}`);
        } else if (sub === "del" || sub === "rm") {
          const posArg = rest[1];
          const n = posArg && /^\d+$/.test(posArg) ? Number.parseInt(posArg, 10) : queryGroups.length;
          if (queryGroups.length <= 1) {
            setStatus("Cannot remove the only step", "error");
          } else if (n >= 1 && n <= queryGroups.length) {
            pushUndo();
            setQueryGroups((prev) => prev.filter((_, i) => i !== n - 1));
            setActiveStepIndex(Math.max(0, n - 2));
            setActiveField(null);
            setStatus(`✗ step ${n} removed`);
          } else {
            setStatus("Invalid step number", "error");
          }
        } else if (sub === "clear") {
          const n = rest[1] ? Number.parseInt(rest[1], 10) : activeStepIndex + 1;
          if (!Number.isNaN(n) && n >= 1 && n <= queryGroups.length) {
            pushUndo();
            setQueryGroups((prev) => {
              const updated = [...prev];
              updated[n - 1] = {
                ...updated[n - 1],
                semanticQuery: "", textQuery: "", translateSemantic: false, translatedQuery: "",
              };
              return updated;
            });
            setStatus(`✓ step ${n} content cleared`);
          } else {
            setStatus("Usage: /step clear [n]", "error");
          }
        } else if (sub && /^\d+$/.test(sub)) {
          const n = Number.parseInt(sub, 10);
          if (n >= 1 && n <= queryGroups.length) {
            setActiveStepIndex(n - 1);
            setActiveField(null);
            setStatus(`→ step ${n}`);
          } else {
            setStatus("Invalid step number", "error");
          }
        } else {
          setStatus("Usage: /step add [x] | /step del [x] | /step clear [x] | /step <n>", "error");
        }
        break;
      }
      case "text":
        if (searchMode === "frames") applyOneShot("text", arg);
        else setStatus("/text only applies in Frames mode", "error");
        break;
      case "translate":
        if (searchMode === "frames") applyOneShot("translate", arg || (queryGroups[activeStepIndex]?.translateSemantic ? "off" : "on"));
        else setStatus("/translate only applies in Frames mode", "error");
        break;
      case "mode":
        if (arg === "frames" || arg === "transcripts") {
          onSetMode(arg);
          setStatus(`→ mode ${arg}`);
        } else {
          setStatus("Usage: /mode frames|transcripts", "error");
        }
        break;
      case "topk": {
        const n = Number.parseInt(arg, 10);
        if (!Number.isNaN(n) && n > 0) {
          if (searchMode === "frames") setTopKInput(String(n));
          else setTranscriptTopK(String(n));
          setStatus(`✓ topK=${n}`);
        } else {
          setStatus("Usage: /topk <n>", "error");
        }
        break;
      }
      case "genre": {
        const match = allGenres.find((g) => g.toLowerCase() === arg.toLowerCase());
        if (match) {
          if (searchMode === "frames") setVideoGenre(match);
          else setTranscriptGenre(match);
          setStatus(`✓ genre=${match}`);
        } else {
          setStatus(`Unknown genre. Options: ${allGenres.join(", ")}`, "error");
        }
        break;
      }
      case "strategy": {
        if (searchMode !== "frames") { setStatus("/strategy only applies in Frames mode", "error"); return; }
        const match = strategies.find((s) => s.id === arg || s.name.toLowerCase() === arg.toLowerCase());
        if (match) {
          setSelectedStrategy(match.id);
          setStatus(`✓ strategy=${match.name}`);
        } else {
          setStatus("Unknown strategy", "error");
        }
        break;
      }
      case "view": {
        if (arg === "score" || arg === "video") {
          if (searchMode === "frames") setViewMode(arg);
          else setTranscriptViewMode(arg);
          setStatus(`✓ view=${arg}`);
        } else {
          setStatus("Usage: /view score|video", "error");
        }
        break;
      }
      case "transcript": {
        if (arg === "on" || arg === "off") {
          setShowTranscript(arg === "on");
          setStatus(`✓ transcript=${arg}`);
        } else {
          setStatus("Usage: /transcript on|off", "error");
        }
        break;
      }
      case "search":
        onSearch();
        break;
      case "clear":
        if (searchMode === "frames") {
          pushUndo();
          setQueryGroups([{ ...defaultGroup, temporalOffsetMs: 0 }]);
          setActiveStepIndex(0);
        } else {
          setTranscriptQuery("");
        }
        setActiveField(null);
        setCorrectionMode(false);
        setStatus("✓ cleared");
        break;
      case "help":
        setStatus("Commands: /step add|del|clear|<n>, /text, /translate, /mode, /topk, /genre, /strategy, /view, /transcript, /search, /clear, /help");
        break;
      default:
        setStatus(`Unknown command: /${cmd}`, "error");
    }
  }

  // ── Transcripts mode: single flat query, no step sequence ─────────────────
  function handleTranscriptEnter(raw: string) {
    if (raw.startsWith("/")) {
      pushHistory(raw);
      runSlashCommand(raw);
      setInputValue("");
      return;
    }
    pushHistory(raw);
    if (raw.trim()) {
      setTranscriptQuery(raw.trim());
      setInputValue("");
    }
    requestSearch();
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      e.preventDefault();
      e.stopPropagation();
      if (e.ctrlKey) { onSearch(); return; }
      const raw = inputValue;

      if (searchMode === "transcripts") {
        handleTranscriptEnter(raw);
        return;
      }
      if (raw.startsWith("/")) {
        pushHistory(raw);
        runSlashCommand(raw);
        setInputValue("");
        return;
      }
      if (activeField) {
        pushHistory(raw);
        confirmField(raw);
        return;
      }
      // idle
      if (raw.trim()) {
        pushHistory(raw);
        pushUndo();
        const idx = activeStepIndex;
        setQueryGroups((prev) => {
          const updated = [...prev];
          updated[idx] = { ...updated[idx], semanticQuery: raw.trim(), translatedQuery: "" };
          return updated;
        });
        setInputValue("");
      }
      requestSearch();
      return;
    }

    if (e.key === "Escape" && e.shiftKey) {
      // Dedicated combo for switching panels — plain Escape stays free for
      // clearing input / closing the video modal without ambiguity.
      e.preventDefault();
      onEscapeToResults();
      return;
    }

    if (e.key === "Escape") {
      if (inputValue) { setInputValue(""); return; }
      if (activeField) {
        setActiveField(null);
        setCorrectionMode(false);
        setStatus("→ idle");
      }
      return;
    }

    if (e.key === "F2" && searchMode === "frames" && !inputValue && !activeField) {
      e.preventDefault();
      startCorrection();
      return;
    }

    if (e.key === "Tab") {
      e.preventDefault();
      // Already cycling (either a command name or an argument value) — advance it.
      if (tabCycle) {
        const nextIndex = (tabCycle.index + 1) % tabCycle.options.length;
        setInputValue(tabCycle.isArg ? `${tabCycle.base} ${tabCycle.options[nextIndex]}` : `${tabCycle.options[nextIndex]} `);
        setTabCycle({ ...tabCycle, index: nextIndex });
        return;
      }
      // Typing a known command's argument — cycle its enum options.
      const argCtx = resolveArgOptions(inputValue);
      if (argCtx && argCtx.options.length > 0) {
        const rest = inputValue.slice(argCtx.base.length).trimStart();
        const filtered = argCtx.options.filter((o) => o.toLowerCase().startsWith(rest.toLowerCase()));
        const pool = filtered.length > 0 ? filtered : argCtx.options;
        setInputValue(`${argCtx.base} ${pool[0]}`);
        setTabCycle({ base: argCtx.base, options: pool, index: 0, isArg: true });
        return;
      }
      // Typing a command name — cycle matching commands.
      if (inputValue.startsWith("/")) {
        const matches = SLASH_COMMANDS.filter((c) => c.startsWith(inputValue));
        if (matches.length === 0) { setStatus(`No command matches "${inputValue}"`, "error"); return; }
        setInputValue(matches[0] + " ");
        setTabCycle({ base: inputValue, options: matches, index: 0, isArg: false });
        return;
      }
      if (searchMode === "frames" && activeField) {
        const fi = FIELD_ORDER.indexOf(activeField);
        const nextField = FIELD_ORDER[(fi + 1) % FIELD_ORDER.length];
        setActiveField(nextField);
        const g = queryGroups[activeStepIndex];
        if (g) setInputValue(getFieldDisplay(g, nextField));
      }
      return;
    }

    if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      const dir = e.key === "ArrowUp" ? -1 : 1;

      // Already cycling — move the highlighted option up/down.
      if (tabCycle) {
        e.preventDefault();
        const nextIndex = (tabCycle.index + dir + tabCycle.options.length) % tabCycle.options.length;
        setInputValue(tabCycle.isArg ? `${tabCycle.base} ${tabCycle.options[nextIndex]}` : `${tabCycle.options[nextIndex]} `);
        setTabCycle({ ...tabCycle, index: nextIndex });
        return;
      }

      // Typing a known command's argument — start cycling its options, like Claude Code's menu navigation.
      const argCtx = resolveArgOptions(inputValue);
      if (argCtx && argCtx.options.length > 0) {
        e.preventDefault();
        const rest = inputValue.slice(argCtx.base.length).trimStart();
        const filtered = argCtx.options.filter((o) => o.toLowerCase().startsWith(rest.toLowerCase()));
        const pool = filtered.length > 0 ? filtered : argCtx.options;
        const currentIndex = pool.findIndex((o) => o.toLowerCase() === rest.toLowerCase());
        const nextIndex = currentIndex >= 0
          ? (currentIndex + dir + pool.length) % pool.length
          : (dir === 1 ? 0 : pool.length - 1);
        setInputValue(`${argCtx.base} ${pool[nextIndex]}`);
        setTabCycle({ base: argCtx.base, options: pool, index: nextIndex, isArg: true });
        return;
      }

      if (searchMode === "frames" && !inputValue && !activeField) {
        e.preventDefault();
        setActiveStepIndex((i) => Math.min(queryGroups.length - 1, Math.max(0, i + dir)));
        return;
      }
      e.preventDefault();
      recallHistory(dir);
      return;
    }

    if ((e.key === "z" || e.key === "Z") && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      undo();
      return;
    }
  }

  // Suggestion list: either "which command" (typing the name) or "which
  // argument" (typing a known command's value) — Tab/↑↓ cycle whichever is active.
  const liveArgContext = resolveArgOptions(inputValue);
  const suggestionIsArg = tabCycle ? tabCycle.isArg : !!liveArgContext;
  const suggestionBase = tabCycle ? tabCycle.base : (liveArgContext ? liveArgContext.base : null);
  const suggestionOptions: string[] = tabCycle
    ? tabCycle.options
    : liveArgContext
      ? (() => {
          const rest = inputValue.slice(liveArgContext.base.length).trimStart();
          const filtered = liveArgContext.options.filter((o) => o.toLowerCase().startsWith(rest.toLowerCase()));
          return (filtered.length > 0 ? filtered : liveArgContext.options).slice(0, 8);
        })()
      : inputValue.startsWith("/")
        ? SLASH_COMMANDS.filter((c) => c.startsWith(inputValue)).slice(0, 6)
        : [];
  const highlightedOption = tabCycle ? suggestionOptions[tabCycle.index % suggestionOptions.length] : suggestionOptions[0];
  const combinedUsageKey = suggestionIsArg && suggestionBase && highlightedOption ? `${suggestionBase} ${highlightedOption}` : highlightedOption;
  const usageCommand = (combinedUsageKey && COMMAND_USAGE[combinedUsageKey])
    ? combinedUsageKey
    : (suggestionBase && COMMAND_USAGE[suggestionBase])
      ? suggestionBase
      : SLASH_COMMANDS.find((c) => inputValue.trim() === c || inputValue.startsWith(c + " "));
  const usageText = usageCommand ? COMMAND_USAGE[usageCommand] : undefined;

  return (
    <div className="flex flex-col h-full">
      {/* ── Inspector ── */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-2 text-sm font-mono">
        <div className="flex items-center gap-3 text-stone-500 dark:text-stone-400 flex-wrap text-sm">
          <span>Mode: <span className="text-stone-900 dark:text-stone-100 font-semibold">{searchMode}</span></span>
          {searchMode === "frames" && (
            <>
              <span>Strategy: <span className="text-stone-900 dark:text-stone-100 font-semibold">{selectedStrategy || "—"}</span></span>
              <span>TopK: <span className="text-stone-900 dark:text-stone-100 font-semibold">{topKInput}</span></span>
              <span>Genre: <span className="text-stone-900 dark:text-stone-100 font-semibold">{videoGenre}</span></span>
            </>
          )}
          {searchMode === "transcripts" && (
            <>
              <span>TopK: <span className="text-stone-900 dark:text-stone-100 font-semibold">{transcriptTopK}</span></span>
              <span>Genre: <span className="text-stone-900 dark:text-stone-100 font-semibold">{transcriptGenre || "Auto"}</span></span>
            </>
          )}
          <button
            onClick={() => runSlashCommand(`/view ${(searchMode === "frames" ? viewMode : transcriptViewMode) === "score" ? "video" : "score"}`)}
            title="Toggle results view (grouped by video vs. flat score list)"
            className="hover:text-orange-700 dark:hover:text-orange-400 transition"
          >
            View: <span className="text-stone-900 dark:text-stone-100 font-semibold">{searchMode === "frames" ? viewMode : transcriptViewMode}</span>
          </button>
        </div>

        {searchMode === "frames" ? (
          <div className="space-y-1">
            {queryGroups.map((g, i) => (
              <div
                key={i}
                onClick={() => { setActiveStepIndex(i); setActiveField(null); inputRef.current?.focus(); }}
                className={`group flex items-center gap-1.5 px-2 py-2 rounded border-2 cursor-pointer transition text-sm ${
                  i === activeStepIndex
                    ? "border-orange-700 bg-orange-700/10 dark:bg-orange-600/15 text-stone-900 dark:text-stone-50"
                    : "border-transparent text-stone-600 dark:text-stone-400 hover:border-stone-400 dark:hover:border-stone-600"
                }`}
              >
                <span className="truncate flex-1">
                  <span className="text-orange-700 dark:text-orange-400 mr-1">{i === activeStepIndex ? "▸" : " "}</span>
                  [{i + 1}] sem:&quot;{g.semanticQuery || "—"}&quot;  text:&quot;{g.textQuery || "—"}&quot;  off:{g.temporalOffsetMs}ms  tr:{g.translateSemantic ? "on" : "off"}
                </span>
                {queryGroups.length > 1 && (
                  <button
                    onClick={(e) => { e.stopPropagation(); runSlashCommand(`/step del ${i + 1}`); inputRef.current?.focus(); }}
                    title={`Remove step ${i + 1}`}
                    className="shrink-0 opacity-0 group-hover:opacity-100 text-rose-700 dark:text-rose-400 hover:text-rose-500 transition px-1"
                  >
                    ×
                  </button>
                )}
              </div>
            ))}
            <button
              onClick={() => { runSlashCommand("/step add"); inputRef.current?.focus(); }}
              className="w-full text-left px-2 py-1.5 rounded border-2 border-dashed border-stone-400 dark:border-stone-600 text-stone-500 dark:text-stone-400 hover:border-orange-700 hover:text-orange-700 dark:hover:border-orange-400 dark:hover:text-orange-400 transition text-sm"
            >
              + add step
            </button>
          </div>
        ) : (
          <div className="px-2 py-2 rounded border-2 border-orange-700 bg-orange-700/10 dark:bg-orange-600/15 text-stone-900 dark:text-stone-50 truncate text-sm">
            query:&quot;{transcriptQuery || "—"}&quot;
          </div>
        )}
      </div>

      {/* ── Status line + command bar ── */}
      <div className="shrink-0 border-t-2 border-stone-800 dark:border-stone-600 px-4 py-2 space-y-1.5">
        {status && (
          <p className={`text-sm font-mono ${status.kind === "error" ? "text-rose-700 dark:text-rose-400" : "text-stone-500 dark:text-stone-400"}`}>
            {status.text}
          </p>
        )}
        {suggestionOptions.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {suggestionOptions.map((opt, i) => (
              <button
                key={opt}
                onClick={() => {
                  setInputValue(suggestionIsArg ? `${suggestionBase} ${opt}` : `${opt} `);
                  setTabCycle({ base: suggestionBase ?? opt, options: suggestionOptions, index: i, isArg: suggestionIsArg });
                  inputRef.current?.focus();
                }}
                className={`font-mono text-xs px-2 py-1 rounded border-2 transition ${
                  opt === highlightedOption
                    ? "border-orange-700 bg-orange-700 text-white dark:border-orange-400 dark:bg-orange-600"
                    : "border-stone-400 dark:border-stone-600 text-stone-500 dark:text-stone-400 hover:border-orange-700 hover:text-orange-700 dark:hover:border-orange-400 dark:hover:text-orange-400"
                }`}
              >
                {opt}
              </button>
            ))}
          </div>
        )}
        {usageText && (
          <p className="text-xs text-stone-500 dark:text-stone-400 italic">{usageText}</p>
        )}
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm text-stone-500 dark:text-stone-400">
            {activeField ? `${activeStepIndex + 1}.${activeField}>` : ">"}
          </span>
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => { setInputValue(e.target.value); setTabCycle(null); }}
            onKeyDown={handleKeyDown}
            placeholder={activeField ? `type ${activeField} value…` : "type a query or /command…"}
            className="flex-1 bg-cream-card dark:bg-stone-800 border-2 border-stone-800 dark:border-stone-500 rounded px-3 py-2 text-base font-mono text-stone-900 dark:text-stone-50 placeholder-stone-400 dark:placeholder-stone-500 focus:outline-none focus:ring-2 focus:ring-orange-600"
          />
        </div>
        <p className="text-xs text-stone-500 leading-normal">
          {searchMode === "frames"
            ? "↑↓ step or cycle suggestions · Tab autocomplete · Enter confirm/advance · Esc idle · Shift+Esc results · F2 edit · Ctrl+Z undo · Ctrl+Enter search"
            : "↑↓/Tab cycle suggestions · Enter search · /topk /genre /mode /view /clear · Ctrl+Enter search"}
        </p>
      </div>
    </div>
  );
});

export default CommandPanel;
