import { useEffect, useRef, useState } from "react";
import { translateTexts } from "@/lib/api";
import type { QueryGroup as QueryGroupType } from "@/types";

interface Props {
  group: QueryGroupType;
  index: number;
  isFirst: boolean;
  onChange: (updated: QueryGroupType) => void;
  onRemove: () => void;
  onSubmit?: () => void;
}

export default function QueryGroup({ group, index, isFirst, onChange, onRemove, onSubmit }: Props) {
  const [translating, setTranslating] = useState(false);
  const [translationError, setTranslationError] = useState("");
  const translationRequestId = useRef(0);

  function update(patch: Partial<QueryGroupType>) {
    onChange({ ...group, ...patch });
  }

  async function translateQuery() {
    const text = group.semanticQuery.trim();
    if (!text) return;

    const requestId = ++translationRequestId.current;
    setTranslating(true);
    setTranslationError("");
    try {
      const response = await translateTexts([text]);
      if (requestId === translationRequestId.current) {
        update({ translatedSemanticQuery: response.translations[0] });
      }
    } catch (error) {
      if (requestId === translationRequestId.current) {
        setTranslationError(error instanceof Error ? error.message : "Translation failed");
      }
    } finally {
      if (requestId === translationRequestId.current) setTranslating(false);
    }
  }

  // Enabling translation immediately produces an English query. Subsequent
  // edits refresh it after a short pause, avoiding a request on every keystroke.
  useEffect(() => {
    if (
      !group.translationEnabled ||
      !group.semanticQuery.trim() ||
      group.translatedSemanticQuery ||
      translationError
    ) return;
    const timeout = window.setTimeout(() => void translateQuery(), 350);
    return () => window.clearTimeout(timeout);
  }, [group.semanticQuery, group.translatedSemanticQuery, group.translationEnabled]);

  function handleTranslationToggle() {
    const enabled = !group.translationEnabled;
    translationRequestId.current += 1;
    setTranslationError("");
    setTranslating(false);
    update({
      translationEnabled: enabled,
      // Keep the original text untouched; force a new result for this source.
      translatedSemanticQuery: undefined,
    });
  }

  // Enter in any of this step's inputs runs the search. Previously this
  // relied on the app root's onKeyDown catching the bubbled event, which
  // also fired search from unrelated places (results grid, video modal).
  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter") {
      e.preventDefault();
      onSubmit?.();
    }
  }

  return (
    <div className="bg-cream-card dark:bg-stone-800 border-2 border-stone-800 dark:border-stone-500 rounded p-4 space-y-3">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <span className="font-retro text-xs font-bold uppercase tracking-wide text-stone-500 dark:text-stone-400">
          {isFirst ? "Search" : `Temporal Step ${index}`}
        </span>
        {!isFirst && (
          <button
            onClick={onRemove}
            className="text-xs text-rose-700 dark:text-rose-400 hover:text-rose-500 dark:hover:text-rose-300 transition"
          >
            Remove
          </button>
        )}
      </div>

      {/* Temporal offset — only for non-first groups */}
      {!isFirst && (
        <div className="flex items-center gap-2 text-sm text-stone-600 dark:text-stone-400">
          <span>Occurs</span>
          <input
            type="number"
            min={0}
            step={1}
            value={Math.round(group.temporalOffsetMs / 1000)}
            onChange={(e) =>
              update({ temporalOffsetMs: Math.max(0, parseInt(e.target.value) || 0) * 1000 })
            }
            onKeyDown={handleKeyDown}
            className="w-20 bg-cream-card dark:bg-stone-700 border-2 border-stone-700 dark:border-stone-500 rounded px-2 py-1 text-stone-900 dark:text-white text-center"
          />
          <span>seconds after the previous step</span>
        </div>
      )}

      {/* Semantic search box */}
      <div className="flex gap-2">
        <div className="flex-1 relative">
          <input
            type="text"
            placeholder="Describe what you want to find…"
            value={group.semanticQuery}
            disabled={translating}
            onChange={(e) => {
              // Ignore an in-flight response for the text that was just replaced.
              translationRequestId.current += 1;
              setTranslating(false);
              setTranslationError("");
              update({ semanticQuery: e.target.value, translatedSemanticQuery: undefined });
            }}
            onKeyDown={handleKeyDown}
            className="w-full bg-cream-card dark:bg-stone-700 border-2 border-stone-700 dark:border-stone-500 rounded px-3 py-2 text-stone-900 dark:text-white placeholder-stone-400 dark:placeholder-stone-500 focus:outline-none focus:ring-2 focus:ring-orange-600"
          />
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={Boolean(group.translationEnabled)}
          title="Use an English translation for search"
          disabled={!group.semanticQuery.trim()}
          onClick={handleTranslationToggle}
          className={`flex items-center gap-2 rounded px-2.5 py-2 text-xs font-bold border-2 transition disabled:opacity-50 disabled:cursor-not-allowed ${
            group.translationEnabled
              ? "bg-orange-700 border-orange-700 text-white"
              : "bg-cream-card dark:bg-stone-700 border-stone-700 dark:border-stone-500 text-stone-500 dark:text-stone-400 hover:text-orange-700 dark:hover:text-orange-400"
          }`}
        >
          <span>VI→EN</span>
          <span className={`relative h-4 w-7 rounded-full ${group.translationEnabled ? "bg-orange-200/80" : "bg-stone-300 dark:bg-stone-500"}`}>
            <span className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-transform ${group.translationEnabled ? "translate-x-3.5" : "translate-x-0.5"}`} />
          </span>
        </button>
      </div>

      {group.translationEnabled && (
        <div className="border-l-2 border-orange-700/70 pl-3 text-sm text-stone-700 dark:text-stone-200" aria-live="polite">
          <span className="text-xs font-semibold text-orange-800 dark:text-orange-300">English query</span>
          <p className="mt-0.5 break-words">
            {translating ? "Translating…" : group.translatedSemanticQuery || "Translation will appear here."}
          </p>
        </div>
      )}
      {translationError && <p className="text-xs text-rose-700 dark:text-rose-400">{translationError}</p>}

    </div>
  );
}
