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
  function update(patch: Partial<QueryGroupType>) {
    onChange({ ...group, ...patch });
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
            onChange={(e) => update({
              semanticQuery: e.target.value,
              translatedQuery: "",
            })}
            onKeyDown={handleKeyDown}
            className="w-full bg-cream-card dark:bg-stone-700 border-2 border-stone-700 dark:border-stone-500 rounded px-3 py-2 text-stone-900 dark:text-white placeholder-stone-400 dark:placeholder-stone-500 focus:outline-none focus:ring-2 focus:ring-orange-600"
          />
        </div>
        {/* Translation toggle */}
        <button
          title={group.translateSemantic ? "Translate ON (query will be translated to English)" : "Translate OFF"}
          onClick={() => update({
            translateSemantic: !group.translateSemantic,
            translatedQuery: "",
          })}
          className={`font-retro px-3 py-2 rounded text-xs font-bold border-2 transition ${
            group.translateSemantic
              ? "bg-orange-700 border-orange-800 text-white"
              : "bg-cream-card dark:bg-stone-700 border-stone-700 dark:border-stone-500 text-stone-500 dark:text-stone-400 hover:text-orange-700 dark:hover:text-orange-400"
          }`}
        >
          VI→EN
        </button>
      </div>

      {group.translateSemantic && group.translatedQuery && (
        <div className="border-l-4 border-orange-700 pl-3">
          <p className="text-xs text-stone-500">English query</p>
          <p className="text-sm text-orange-800 dark:text-orange-300 break-words">{group.translatedQuery}</p>
        </div>
      )}

    </div>
  );
}
