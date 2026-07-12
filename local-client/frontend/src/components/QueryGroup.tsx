import type { QueryGroup as QueryGroupType } from "@/types";

interface Props {
  group: QueryGroupType;
  index: number;
  isFirst: boolean;
  onChange: (updated: QueryGroupType) => void;
  onRemove: () => void;
}

export default function QueryGroup({ group, index, isFirst, onChange, onRemove }: Props) {
  function update(patch: Partial<QueryGroupType>) {
    onChange({ ...group, ...patch });
  }

  return (
    <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl p-4 space-y-3">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-slate-500 dark:text-slate-400">
          {isFirst ? "Search" : `Temporal Step ${index}`}
        </span>
        {!isFirst && (
          <button
            onClick={onRemove}
            className="text-xs text-red-500 dark:text-red-400 hover:text-red-600 dark:hover:text-red-300 transition"
          >
            Remove
          </button>
        )}
      </div>

      {/* Temporal offset — only for non-first groups */}
      {!isFirst && (
        <div className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
          <span>Occurs</span>
          <input
            type="number"
            min={0}
            step={1}
            value={Math.round(group.temporalOffsetMs / 1000)}
            onChange={(e) =>
              update({ temporalOffsetMs: Math.max(0, parseInt(e.target.value) || 0) * 1000 })
            }
            className="w-20 bg-slate-100 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-slate-900 dark:text-white text-center"
          />
          <span>seconds after the previous step</span>
        </div>
      )}

      {/* Semantic search box */}
      <div className="flex gap-2">
        <div className="flex-1 relative">
          <input
            type="text"
            placeholder="Semantic search (visual scene description)…"
            value={group.semanticQuery}
            onChange={(e) => update({
              semanticQuery: e.target.value,
              translatedQuery: "",
            })}
            className="w-full bg-slate-100 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg px-3 py-2 text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        {/* Translation toggle */}
        <button
          title={group.translateSemantic ? "Translate ON (query will be translated to English)" : "Translate OFF"}
          onClick={() => update({
            translateSemantic: !group.translateSemantic,
            translatedQuery: "",
          })}
          className={`px-3 py-2 rounded-lg text-sm font-medium border transition ${
            group.translateSemantic
              ? "bg-blue-600 border-blue-500 text-white"
              : "bg-slate-100 dark:bg-slate-700 border-slate-300 dark:border-slate-600 text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
          }`}
        >
          VI→EN
        </button>
      </div>

      {group.translateSemantic && group.translatedQuery && (
        <div className="border-l-2 border-blue-500 pl-3">
          <p className="text-xs text-slate-500">English query</p>
          <p className="text-sm text-blue-700 dark:text-blue-200 break-words">{group.translatedQuery}</p>
        </div>
      )}

      {/* OCR / Transcript text search box */}
      <input
        type="text"
        placeholder="Text search (OCR / transcript keywords)…"
        value={group.textQuery}
        onChange={(e) => update({ textQuery: e.target.value })}
        className="w-full bg-slate-100 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg px-3 py-2 text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
      />
    </div>
  );
}
